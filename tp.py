# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from enum import Enum
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from torch import nn
if os.uname().sysname != "Darwin":
    from torch.distributed import _functional_collectives as funcol
else:
    # Distributed is not supported on MacOS
    funcol = None

from model import Attention, FeedForward, Transformer
from quantize import WeightOnlyInt4Linear, WeightOnlyInt8Linear


def _get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def is_local():
    return _get_rank() == 0

def local_break():
    if is_local():
        breakpoint()
    dist.barrier()

def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

global device_mesh

def _get_tp_mesh():
    # device_mesh has only TP dimension for now
    return device_mesh

def maybe_init_dist() -> Optional[int]:
    try:
        # provided by torchrun
        rank = _get_rank()
        world_size = _get_world_size()

        if world_size < 2:
            # too few gpus to parallelize, tp is no-op
            return None
    except KeyError:
        # not run via torchrun, no-op
        return None

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    global device_mesh
    device_mesh = dist.init_device_mesh(
        "cuda",
        (world_size,),  # Only TP dimension for now
    )
    return rank

class TPMode(Enum):
    MANUAL = 0
    DTENSOR = 1

def _apply_tp_linear(linear: nn.Linear, style: str) -> None:
    rank = _get_rank()
    world_size = _get_world_size()
    tp_mesh = _get_tp_mesh()

    # Linear's weight matrix is transposed, and is of shape
    # (linear.out_features, linear.in_features)
    dim_lookup = {
        "colwise": (0, "out_features", ColwiseParallel()),
        "rowwise": (1, "in_features", RowwiseParallel()),
    }
    assert style in dim_lookup
    shard_dim, size_attr, tp_plan = dim_lookup[style]

    # ensure we can shard evenly
    assert getattr(linear, size_attr) % world_size == 0
    def shard(x, dim):
        assert x.size(dim=dim) % world_size == 0
        return torch.tensor_split(x, world_size, dim=dim)[rank]

    def shard_scale(linear, shard_dim):
        if hasattr(linear, "scales_and_zeros"):
            linear.scales_and_zeros = shard(linear.scales_and_zeros, 1 - shard_dim)
            if style == "rowwise":
                assert linear.scales_and_zeros.shape[0] * 32 == sharded_weight.shape[1] * sharded_weight.shape[2] * sharded_weight.shape[3]
                assert linear.scales_and_zeros.shape[1] == sharded_weight.shape[0] * 8
        elif hasattr(linear, "scale"):
            if style == "colwise":
                linear.scales = shard(linear.scales, 0)

    # shard
    tp_mode: TPMode
    if isinstance(linear, (WeightOnlyInt4Linear, WeightOnlyInt8Linear)):
        # TODO: DTensor doesn't have a way to distribute quantized tensor yet.
        # Should revisit when that capability is added.
        sharded_weight = shard(linear.weight, shard_dim)
        linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
        shard_scale(linear, shard_dim)
        tp_mode = TPMode.MANUAL
    else:
        # Use DTensor based TP
        parallelize_module(linear, tp_mesh, tp_plan)
        tp_mode = TPMode.DTENSOR

    # local_break()
    setattr(linear, size_attr, getattr(linear, size_attr) // world_size)

    # shape info should still be synced
    # assert linear.weight.shape == (linear.out_features, linear.in_features)
    return tp_mode


def _apply_tp_ffn(mlp: FeedForward) -> None:
    assert hasattr(mlp, "w1")
    assert hasattr(mlp, "w3")
    assert hasattr(mlp, "w2")

    tp_mode = _apply_tp_linear(mlp.w1, "colwise")
    tp_mode = _apply_tp_linear(mlp.w3, "colwise")
    tp_mode = _apply_tp_linear(mlp.w2, "rowwise")

    if tp_mode == TPMode.MANUAL:
        # In manual mode, we need to manually add an all-reduce at the end
        world_size = _get_world_size()
        mlp.register_forward_hook(lambda _module, _input, output: funcol.all_reduce(
            output, "sum", list(range(world_size))))


def _apply_tp_attn(attn: Attention) -> None:
    assert hasattr(attn, "wq")
    assert hasattr(attn, "wk")
    assert hasattr(attn, "wv")
    assert hasattr(attn, "wo")

    kv_size = attn.n_local_heads * attn.head_dim
    tp_mode = _apply_tp_linear(attn.wq, "colwise")
    tp_mode = _apply_tp_linear(attn.wk, "colwise")
    tp_mode = _apply_tp_linear(attn.wv, "colwise")
    tp_mode = _apply_tp_linear(attn.wo, "rowwise")

    # overwrite
    world_size = _get_world_size()
    attn.n_head = attn.n_head // world_size
    attn.dim = attn.dim // world_size
    attn.head_dim = attn.dim // attn.n_head
    attn.n_local_heads = attn.n_local_heads // world_size

    if tp_mode == TPMode.MANUAL:
        # In manual mode, we need to manually add an all-reduce at the end
        attn.register_forward_hook(lambda _module, _input, output: funcol.all_reduce(
            output[0], "sum", list(range(world_size))))


def _apply_tp_Transformer(Transformer: Transformer) -> None:
    # overwrite config before Transformer.setup_cache is called
    world_size = _get_world_size()
    Transformer.config.n_head = Transformer.config.n_head // world_size
    Transformer.config.dim = Transformer.config.dim // world_size
    Transformer.config.n_local_heads = Transformer.config.n_local_heads // world_size


def apply_tp(model: Transformer) -> None:
    _apply_tp_Transformer(model)
    for block in model.layers:
        # Apply to MLP
        _apply_tp_ffn(block.feed_forward)
        _apply_tp_attn(block.attention)

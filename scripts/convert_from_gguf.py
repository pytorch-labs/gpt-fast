# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Usage:
#
# 0. Setup
#   pip install gguf
#   git@github.com:ggerganov/llama.cpp.git
#
# 1. Preparation: convert existing hf model to fp16
#   `python llama.cpp/convert.py [HF-dir] --outtype f16``
#
#    which will generate [HF-dir]/ggml-model-f16.gguf
#
# 2. Convert GGUF file to a checkpoint
#    `python scripts/convert_from_gguf.py --gguf_file [HF-dir]/ggml-model-f16.gguf --checkpoint_file [HF-dir]/model_gguf.pth`
#
# 3. Validate that it works:
#    `python generate.py --checkpoint_path [HF-dir]/model_gguf.pth --device=cpu --prompt "Hello, my name is" --max_new_tokens 20`
#
# NOTE: Only works for fp32 and fp16 types so that means, Steps 1-3 
# isn't providing much value right now because `convert_hf_checkpoint.py`
# can directly generate an equivalent .pth checkpoint file without gguf format indirection. 
# In the future, we will support running the quantized version of the graph.

import argparse

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import gguf

import torch
import torch.nn as nn

from gguf import GGUFValueType, ReaderTensor

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from model import ModelArgs, Transformer


@dataclass
class AttentionArgs:
    head_count: int
    head_count_kv: int
    layer_norm_rms_epsilon: float


@dataclass
class RopeArgs:
    dimension_count: int | None = None
    freq_base: float | None = None


@dataclass
class GGUFModelArgs:
    arch: str
    embedding_length: int
    block_count: int
    feed_forward_length: int
    vocab_size: int
    attention: AttentionArgs
    rope: RopeArgs


@dataclass
class GGUFWeights:
    tensors: list[ReaderTensor]


def _create_pt_model(
    gguf_model_args: GGUFModelArgs,
) -> nn.Module:
    llama_model_args = ModelArgs(
        dim=gguf_model_args.embedding_length,
        n_layer=gguf_model_args.block_count,
        n_head=gguf_model_args.attention.head_count,
        n_local_heads=gguf_model_args.attention.head_count_kv,
        vocab_size=gguf_model_args.vocab_size,
        norm_eps=gguf_model_args.attention.layer_norm_rms_epsilon,
        intermediate_size=gguf_model_args.feed_forward_length,
    )
    pt_model = Transformer(llama_model_args)
    pt_model.eval()
    return pt_model


_name_replacements = [
    ("blk", "layers"),
    ("token_embd", "tok_embeddings"),
    ("attn_q", "attention.wq"),
    ("attn_k", "attention.wk"),
    ("attn_v", "attention.wv"),
    ("attn_output", "attention.wo"),
    ("attn_norm", "attention_norm"),
    ("output_norm.weight", "norm.weight"),
    ("ffn_down", "feed_forward.w2"),
    ("ffn_gate", "feed_forward.w1"),
    ("ffn_up", "feed_forward.w3"),
]


def _convert_gguf_tensor_name_to_llama_nn(gguf_name: str) -> str:
    result = copy.deepcopy(gguf_name)
    for gguf_string, replacement in _name_replacements:
        result = result.replace(gguf_string, replacement)
    return result


def _convert_to_state_dict(gguf_weights: GGUFWeights) -> Mapping[str, Any]:
    state_dict = {}

    for tensor in gguf_weights.tensors:
        gguf_tensor_name = tensor.name
        nn_tensor_name = _convert_gguf_tensor_name_to_llama_nn(gguf_tensor_name)
        # gguf is reversed
        reversed_shape = tensor.shape[::-1]
        new_tensor = tensor.data.reshape(reversed_shape)
        state_dict[nn_tensor_name] = torch.from_numpy(new_tensor)

    return state_dict


def _get_metadata(reader: gguf.GGUFReader) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    for idx, field in enumerate(reader.fields.values()):
        val = None
        if field.types[:1] == [GGUFValueType.ARRAY]:
            itype = field.types[-1]
            if itype == GGUFValueType.STRING:
                val = [
                    str(bytes(field.parts[idx]), encoding="utf-8") for idx in field.data
                ]
            else:
                val = [pv for idx in field.data for pv in field.parts[idx].tolist()]
        elif field.types[0] == GGUFValueType.STRING:
            val = str(bytes(field.parts[-1]), encoding="utf-8")
        else:
            val = field.parts[-1].tolist()[0]

        metadata[field.name] = val

    return metadata


def _build_model_args(metadata: dict[str, Any]) -> GGUFModelArgs:
    arch = metadata["general.architecture"]
    assert arch == "llama", "Only LLaMa models are supported by this converter."

    gguf_ft = metadata["general.file_type"]
    # ALL_F32 or MOSTLY_F16
    assert (
        gguf_ft == 0 or gguf_ft == 1
    ), "Only fp32 or fp16 are supported by this converter."

    return GGUFModelArgs(
        arch=arch,
        embedding_length=metadata[f"{arch}.embedding_length"],
        block_count=metadata[f"{arch}.block_count"],
        feed_forward_length=metadata[f"{arch}.feed_forward_length"],
        vocab_size=len(metadata["tokenizer.ggml.tokens"]),
        attention=AttentionArgs(
            head_count=metadata[f"{arch}.attention.head_count"],
            head_count_kv=metadata[f"{arch}.attention.head_count_kv"],
            layer_norm_rms_epsilon=metadata[f"{arch}.attention.layer_norm_rms_epsilon"],
        ),
        rope=RopeArgs(
            freq_base=metadata.get(f"{arch}.rope.freq_base", None),
            dimension_count=metadata.get(f"{arch}.rope.dimension_count", None),
        ),
    )


def convert_to_checkpoint(
    gguf_model_args: GGUFModelArgs, gguf_weights: GGUFWeights
) -> Mapping[str, Any]:
    assert (
        gguf_model_args.arch == "llama"
    ), "Only LLaMa models are supported by this converter."

    # Step 1: Create the PyTorch model
    print("Create the PyTorch model")
    pt_model = _create_pt_model(gguf_model_args)

    # Step 2: Converting gguf weights into state dict
    print("Converting gguf weights into state dict")
    state_dict: Mapping[str, Any] = _convert_to_state_dict(gguf_weights)

    # Step 3: Verify by loading state dict
    print("Loading state dict")
    pt_model.load_state_dict(state_dict)
    return state_dict


def load_gguf_file(gguf_file: str) -> (GGUFModelArgs, GGUFWeights):
    """
    Load a GGUF file and return the model arguments and weights.
    """
    if not Path(gguf_file).is_file():
        raise ValueError(f"Could not find file {gguf_file}")

    reader = gguf.GGUFReader(gguf_file, "r")

    # Step 1: Build GGUFModelArgs
    metadata = _get_metadata(reader)
    model_args = _build_model_args(metadata)

    # Step 2: Build GGUFWeights
    gguf_weights = GGUFWeights(tensors=reader.tensors)

    return (model_args, gguf_weights)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gguf_file",
        type=str,
        required=True,
        help="The GGUF file to load.",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        required=True,
        help="The path to save the PyTorch checkpoint file.",
    )
    args = parser.parse_args()

    gguf_model_args, gguf_weights = load_gguf_file(args.gguf_file)
    state_dict = convert_to_checkpoint(gguf_model_args, gguf_weights)

    torch.save(state_dict, args.checkpoint_file)


if __name__ == "__main__":
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field, replace
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

import math
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

# Inefficient SDPA
def naive_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(1, 1, L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5

    sdpa: str = None

    prune_layer: int = 15
    chai_activate: bool = True
    chai_layers: list = field(
        default_factory=lambda: [
            28,
            28,
            28,
            18,
            18,
            18,
            18,
            18,
            18,
            18,
            18,
            18,
            18,
            18,
            18,
            8,
            8,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
        ]
    )

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config in str(name).upper() or config in str(name)]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match

        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    "Mistral-7B": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),
    "Llama-3-8B": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

class EarlyExitQCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, max_seq_length, dim)
        self.register_buffer('q_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, q_val):
        # input_pos: [S], k_val: [B, S, D]
        assert input_pos.shape[0] == q_val.shape[1]

        q_out = self.q_cache
        q_out[:, input_pos] = q_val

        return q_out

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs, early_exit: int = -1) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.prune_layer = config.prune_layer
        self.chai_activate = config.chai_activate
        self.chai_layers = config.chai_layers
        self.layers = nn.ModuleList(TransformerBlock(i, config) for i in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        assert early_exit < len(self.layers), f"Early exit ({early_exit}) is more than Num layers ({len(self.num_layers)})"
        self.num_layers = len(self.layers)
        self.early_num_layers = len(self.layers) if early_exit == -1 else early_exit
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.q_cache = None

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base, dtype)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.q_cache = EarlyExitQCache(max_batch_size, max_seq_length, self.config.dim, dtype)

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i in range(self.num_layers):
            layer = self.layers[i]
            x = layer(x, input_pos, freqs_cis, mask)
            if i == (self.early_num_layers - 1):
                # update the early exit cache with the input
                # pos of the output layer to make sure
                # q cache is filled from prefill step
                self.q_cache.update(input_pos, x)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    def forward_early(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i in range(self.early_num_layers):
            layer = self.layers[i]
            x = layer(x, input_pos, freqs_cis, mask)
        self.q_cache.update(input_pos, x)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    # TODO: def forward_remainder (updates KVQ cache to use KV)
    def forward_remainder(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        # iterate over first N layers and predict only the last token
        # iterate over the last N layers and do the full remainder pass
        # make sure we update the Q cache in this process
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx) # (bsz, seq, hidden_dim)

        # early exit should only be the last token
        early_input_pos = input_pos[-1:]
        early_input_x = x[:, -1:] # only embeding for the last token
        early_freq_cis = freqs_cis[-1:]
        early_mask = mask[:, :, -1:]

        for i in range(self.early_num_layers):
            # early exit pass
            layer = self.layers[i]
            early_input_x = layer(early_input_x, early_input_pos, early_freq_cis, early_mask)

        # save the query for the latest token
        self.q_cache.update(early_input_pos, early_input_x)

        # concatenate early exit query cache with the newly computed token
        # for the remaining layers
        cached_queries = self.q_cache.q_cache[:, input_pos[:-1]]
        x = torch.cat((cached_queries, early_input_x), dim=1)
        for i in range(self.early_num_layers, self.num_layers):
            # remainder pass
            layer = self.layers[i]
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str, early_exit: int = -1, **kwargs):
        model_args = ModelArgs.from_name(name)
        model_args = replace(model_args, **kwargs)
        return cls(model_args, early_exit=early_exit)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(layer_id, config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.layer_id = layer_id
        self.chai_activate = config.chai_activate
        self.prune_layer = config.prune_layer
        self.chai_layer_param = config.chai_layers[layer_id] if layer_id < len(config.chai_layers) else config.chai_layers[-1]

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

# Average tokens/sec: 109.89
# Memory used: 13.88 GB
# Average tokens/sec: 88.67
# Memory used: 27.49 GB
# Average tokens/sec: 109.68
# Memory used: 27.49 GB
class Attention(nn.Module):
    def __init__(self, layer_id: int, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

        self.sdpa = config.sdpa

        self.layer_id = layer_id
        self.chai_activate = config.chai_activate
        self.prune_layer = config.prune_layer
        self.chai_layer_param = config.chai_layers[layer_id] if layer_id < len(config.chai_layers) else config.chai_layers[-1]

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        # NOTE: The first sequence needs to be atleast of size 6
        # if not we throw an error.
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        if self.chai_activate and self.layer_id >= self.prune_layer:
            cluster_assignment_log_per_example = dict()
            if input_pos[0] == 0:
                # first sentence

                k = self.kv_cache.k_cache[:bsz, :, : input_pos[0] + seqlen, :]
                v = self.kv_cache.v_cache[:bsz, :, : input_pos[0] + seqlen, :]
                q = q.view(bsz, self.n_local_heads, seqlen, self.head_dim)
                k = k.view(bsz, self.n_local_heads, seqlen, self.head_dim)
                num_examples, num_org_heads, seq_len, head_dim = q.shape
                q_four = q[:, :, :5, :]
                k_four = k[:, :, :5, :]
                scores_four = F.softmax(
                    (
                        torch.matmul(q_four, k_four.transpose(2, 3))
                        / math.sqrt(self.head_dim)
                    ).float(),
                    dim=-1,
                )
                scores_four_numpy = scores_four.cpu().numpy()
                scores_new_xk_xq = torch.zeros(
                    [num_examples, num_org_heads, seq_len, seq_len],
                    device=q.device,
                    dtype=q.dtype,
                )
                k_new = torch.zeros(
                    [num_examples, self.chai_layer_param, seq_len, head_dim],
                    dtype=k.dtype,
                    device=k.device,
                )
                q_new = torch.zeros(
                    [num_examples, self.chai_layer_param, seq_len, head_dim],
                    dtype=q.dtype,
                    device=q.device,
                )
                for ex_id in range(num_examples):
                    assert num_examples == 1
                    temp_data = dict()
                    ex_id_score = scores_four_numpy[ex_id, :]
                    # if ex_id_score.shape[1] > 4:
                    # use_small = False
                    num_heads = ex_id_score.shape[0]
                    first_sample_score = ex_id_score.reshape((num_heads, -1))
                    dist_arr = cdist(
                        first_sample_score, first_sample_score, metric="cosine"
                    )
                    cluster = AgglomerativeClustering(
                        n_clusters=self.chai_layer_param,
                        metric="precomputed",
                        linkage="average",
                    )
                    try:
                        cluster = cluster.fit(dist_arr)
                    except:
                        import ipdb

                        ipdb.set_trace()
                    cluster_assignment = cluster.labels_
                    self.grouping = cluster_assignment
                    for cluster_idx in range(self.chai_layer_param):
                        grouped_heads = np.where(cluster_assignment == cluster_idx)[
                            0
                        ].tolist()
                        temp_data[cluster_idx] = grouped_heads
                    cluster_assignment_log_per_example[ex_id] = temp_data
                    # else:
                    # cluster_assignment_log_per_example[ex_id] = temp_data
                    # xk_new = xk
                    # xq_new = xq
                    selected = [np.where(self.grouping == cluster_idx)[0].tolist()[0] for cluster_idx in range(self.chai_layer_param)]
                    k_new[ex_id, :, :, :] = k[ex_id, selected, :, :]
                    q_new[ex_id, :, :, :] = q[ex_id, selected, :, :]
            else:
                # scores
                k = self.kv_cache.k_cache[:bsz, :, : input_pos[0] + seqlen, :]
                v = self.kv_cache.v_cache[:bsz, :, : input_pos[0] + seqlen, :]
                q = q.view(bsz, self.n_local_heads, 1, self.head_dim)
                k = k.view(bsz, self.n_local_heads, input_pos[0] + seqlen, self.head_dim)
                num_examples, num_org_heads, seq_len, head_dim = k.shape
                scores_new_xk_xq = torch.zeros(
                    [num_examples, num_org_heads, 1, seq_len],
                    device=q.device,
                    dtype=q.dtype,
                )
                cluster_assignment = self.grouping
                for ex_id in range(num_examples):
                    selected = [np.where(self.grouping == cluster_idx)[0].tolist()[0] for cluster_idx in range(self.chai_layer_param)]
                    temp_data = dict()
                    for cluster_idx in range(self.chai_layer_param):
                        grouped_heads = np.where(cluster_assignment == cluster_idx)[
                            0
                        ].tolist()
                        temp_data[cluster_idx] = grouped_heads
                    cluster_assignment_log_per_example[ex_id] = temp_data
                k_new = k[:, selected, :, :]
                q_new = q[:, selected, :, :]

            scores_new_temp = torch.matmul(q_new, k_new.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )
            # if use_small:
            # putting them back together
            cluster_idx_index = torch.zeros(num_org_heads, dtype=torch.int)
            for ex_id in range(num_examples):
                for cluster_idx in range(self.chai_layer_param):
                    cluster_idx_index[cluster_assignment_log_per_example[ex_id][cluster_idx]] = cluster_idx
                scores_new_xk_xq[ex_id, :, :, :] = scores_new_temp[ex_id, cluster_idx_index, :, :]

            # else:
            # scores_new_xk_xq = scores_new_temp
            if mask is not None:
                scores_new_xk_xq = scores_new_xk_xq + mask[:, :, :, :seq_len]
            scores_new_xk_xq = F.softmax(scores_new_xk_xq.float(), dim=-1).type_as(q)
            scores = scores_new_xk_xq
            # v = v.transpose(1, 2)
            output = torch.matmul(scores, v)  # (bs, n_local_heads, slen, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            return self.wo(output)

        else:
            k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
            if self.sdpa == "naive":
                y = naive_scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

            y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

            y = self.wo(y)
            return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

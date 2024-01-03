# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional,Tuple,List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

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
        assert len(config) == 1, name
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.float16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        kw_cache_shape = (max_batch_size, max_seq_length, 2, n_heads, n_heads)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('kw_cache', torch.zeros(kw_cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        #kw_out = self.kw_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        #kw_out[:, input_pos] = kw_val

        return k_out, v_out 

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class DynamicWeightProjection(nn.Module):

    def __init__(self, num_heads=32, num_groups=1, residual=True, squeeze_ratio=16, query_input_dim=4096, dynamic_squeeze_ratio=16, dynamic_w_hidden_dim=128,dtype=torch.float16):
        super().__init__()
        #self.layer_idx = layer_idx
        self.num_heads = num_heads 
        self.num_groups = num_groups 
        self.query_input_dim = query_input_dim 
        self.dynamic_w_init = True 
        self.dynamic_d_init = True 
        self.dynamic_squeeze_ratio = dynamic_squeeze_ratio# mqy
        self.dynamic_w_hidden_dim = dynamic_w_hidden_dim # mqy
        self.merge_projection = True
        self.dw_hidden_activation = nn.GELU()
        self.num_heads_per_group = self.num_heads // self.num_groups
        self.dw_activation = nn.Tanh()
        self.dw1_norm = RMSnormNoscale(dim=-1)

        self.pre_proj = CrossHeadProjectionV2('pre', num_heads=self.num_heads, squeeze_ratio=None) # mqy
        self.post_proj = CrossHeadProjectionV2('post', num_heads=self.num_heads, squeeze_ratio=None) # mqy

        if self.dynamic_w_init is not None:
            dynamic_hidden_dim = self.num_heads_per_group // self.dynamic_squeeze_ratio 
            self.dynamic_hidden_dim = dynamic_hidden_dim 
            if self.dynamic_w_hidden_dim:
                self.dw1 = nn.parameter.Parameter(torch.zeros(self.query_input_dim, self.num_groups, 4, self.dynamic_w_hidden_dim, dtype=dtype)) #(4096, 1, 4, 128)
                G, K, M = self.num_groups, self.dynamic_w_hidden_dim, self.num_heads_per_group
                I = dynamic_hidden_dim * 2 # (1 if self.merge_dynamic_w_hidden else 2)
                self.qkw = nn.parameter.Parameter(torch.zeros([G, 4, K, I, M], dtype=dtype)) # (1, 4, 128, 4, 32)
        if self.dynamic_d_init is not None:
            self.dd = nn.parameter.Parameter(torch.zeros(self.query_input_dim, self.num_groups, self.num_heads_per_group * 4, dtype=dtype)) #  (4096, 1, 128)

        #self.merge_weights()

    def merge_weights(self):
        self.dw_m = nn.parameter.Parameter(torch.cat([self.dw1.reshape(self.query_input_dim, -1), self.dd.squeeze(1)], dim=-1)).to(self.dw1.device) # E,(4*K + K)  K=2*N*I
        self.qkw_m = self.qkw.permute(0,1,2,3,4).reshape(4,self.dynamic_w_hidden_dim,-1).to(self.dw1.device) #(4,K,I*M)
        #print('qkw_m', self.qkw_m.shape, self.qkw_m.device)
        self.sw = nn.parameter.Parameter(torch.stack([self.pre_proj.w, self.post_proj.w]).squeeze(1)).to(self.dw1.device) # (2,N,N)
        #print('sw', self.sw.shape, self.sw.device)
        
    def forward(self,query_vec,key_vec,KW:Optional[torch.Tensor]=None, use_cache:Optional[bool]=False):  
        pre_dw_args:Optional[Tuple[Tensor,Tensor,Tensor,Tensor,Tensor,Tensor]] = None
        post_dw_args:Optional[Tuple[Tensor,Tensor,Tensor,Tensor,Tensor,Tensor]] = None
        pre_w:Optional[Tensor] = None
        post_w:Optional[Tensor] = None
        if KW is not None and len(KW.shape)>1:
            B,T,D = query_vec.shape
            N,I = self.num_heads_per_group, self.dynamic_hidden_dim
            shape = (B,T,2,2,-1,N) 
            dw_hidden, dd = (query_vec @ self.dw_m).split([2*2*N*(2*I), 2*2*N*1], -1) # BT(4K), BT4N         # K=2*N*I
            dw_hidden = self.dw_hidden_activation(dw_hidden)
            dw_hidden = dw_hidden.reshape(dw_hidden.shape[:2]+(4,-1)) #B T (4 K) -> B T 4 K
            dw = torch.einsum('B T C K, C K D -> B T C D', dw_hidden, self.qkw_m) # BT4K,4K(MI)->BT4(MI)
            w1, w2 = dw.view(shape).split(I,-2) # BT22(2*I)N -> 2*[BT22IN]:BT(pre/post)(q/k)IN
            w1 = self.dw1_norm(w1) # BT22IN
            qkw = torch.einsum(f'BTKJIN,BTKJIM->BTKJNM', w1, w2)  # j=k=2, BT2{2}NM
            qkdd = self.dw_activation(dd).squeeze(-1).view(shape[:-2]+(N,)) # BT4N->BT22N
            qkw = qkw + torch.diag_embed(qkdd) # BT2{2}NN,->BT2{2}NN
            qw, kw = qkw.unbind(3)  # BT2{2}NN-> [BT2NM]*2
            KW = torch.cat([KW, kw], dim=1)  # BS{2}NM,B1{2}NM->B(S+1){2}NM
            w = self.sw + qw + KW + torch.eye(N,dtype=qw.dtype, device=qw.device)  #(2,N,M)+ (B,1,2,N,M) + (B,S,2,N,M) + (N,M)
            pre_w, post_w = w.unbind(2) #BSNM
        else:
            pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd = None, None, None, None, None, None
            post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd = None, None, None, None, None, None
            dw_hidden = torch.einsum('BTD,DGCK->BTGCK', query_vec, self.dw1)  # C=4 [pre,post]*[query,key]
            dw_hidden = self.dw_hidden_activation(dw_hidden) #BTGCK
            w1, w2 = torch.split(torch.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, self.qkw), self.qkw.shape[-2]//2, dim=-2) #BTGC(2I)M -> [BTGCIM] * 2
            if hasattr(self, 'dw1_norm'): w1 = self.dw1_norm(w1) # BTGCIM
            pre_qw1, pre_kw1, post_qw1, post_kw1 = unbind(w1, 4, dim=3) # BT4GIM->[BTGIM]*4
            pre_qw2, pre_kw2, post_qw2, post_kw2 = unbind(w2, 4, dim=3)
            dd = torch.einsum('BTD,DGM->BTGM', query_vec, self.dd)
            dd = self.dw_activation(dd)
            pre_qdd, pre_kdd, post_qdd, post_kdd = torch.split(dd, dd.shape[-1] // 4, dim=-1)
            pre_dw_args = (pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd)
            post_dw_args = (post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd)
            if use_cache is not None and use_cache:
                pre_kw = torch.einsum('BSGIM, BSGIN->BSMN', pre_kw1, pre_kw2) + torch.diag_embed(pre_kdd.squeeze(2))  # merge kw and kdd
                post_kw = torch.einsum('BSGIM, BSGIN->BSMN', post_kw1, post_kw2) + torch.diag_embed(post_kdd.squeeze(2))
                KW = torch.stack((pre_kw, post_kw), dim=-3)
        return pre_dw_args, post_dw_args, pre_w, post_w, KW

class RMSnormNoscale(nn.Module):
    
    def __init__(self, epsilon=1e-6, dim=-1):
        super().__init__()
        self.dim = dim 
        self.epsilon = epsilon

    def forward(self, inputs):
        var = inputs.pow(2).mean(dim=self.dim, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        return normed_inputs 


class CrossHeadProjectionV2(nn.Module):

    def __init__(self, mode, num_heads=16, num_groups=1, squeeze_ratio=None, dtype=torch.float16):
        super().__init__()
        #self.layer_idx = layer_idx
        self.mode = mode
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_heads_per_group = self.num_heads // self.num_groups
        self.squeeze_ratio = squeeze_ratio
        self.use_static_w = True
        if self.squeeze_ratio is None:
            self.w = nn.parameter.Parameter(data=torch.zeros(self.num_groups, self.num_heads_per_group, self.num_heads_per_group, dtype=dtype))
         
    def forward(self, inputs, 
            dws:Optional[Tuple[Tensor,Tensor, Tensor,Tensor, Tensor,Tensor]]=None,
            query_vec=None, key_vec=None, 
            proj_w:Optional[Tensor]=None,
            ): # v2 raw tag
        if proj_w is not None:
            ret = torch.einsum('BNTS,BSNM->BMTS', inputs, proj_w)
        else:
            assert dws is not None
            qw1, qw2, kw1, kw2, qdd, kdd = dws
            shape = inputs.shape
            inputs = inputs.unsqueeze(1)
            inputs_label = 'BGMTS'
            w = self.w + torch.eye(self.num_heads_per_group, device=self.w.device, dtype=self.w.dtype)
            ret = torch.einsum('BGMTS,GMN->BGNTS', inputs, w)
            if True or qw1 is not None:
                hidden_sym = 'I'; hidden_label = inputs_label.replace('M', 'I') # BGITS
                for sym, (w1, w2) in zip(['T', 'S'], [(qw1, qw2), (kw1, kw2)]): # tag
                    dw_label = f'B{sym}G{hidden_sym}M'  # w1: BTGIM, dw_label:BTGIM
                    dynamic_hidden_dim = w1.shape[dw_label.index(hidden_sym)]
                    eqn1 = f'{inputs_label},{dw_label}->{hidden_label}' # 'BGMTS,BTGMI->BGITS'
                    eqn2 = f'{hidden_label},{dw_label}->{inputs_label}' # 'BGITS,BTGMI->BGMTS'
                    for i in range(dynamic_hidden_dim):
                        hidden = torch.einsum(eqn1.replace(hidden_sym, ''), inputs, w1[..., i, :]) # BGMTS,BTG(I)M->BGTS
                        out = torch.einsum(eqn2.replace(hidden_sym, ''), hidden, w2[..., i, :]) #  'BG(I)TS,BTG(I)M->BGMTS'
                        ret = ret + out
            if True or qdd is not None:
                for sym, dd in zip(['T', 'S'], [qdd, kdd]):
                    dd_label = f'B{sym}GM'
                    dout = torch.einsum(f'{inputs_label},{dd_label}->{inputs_label}', inputs, dd) # BGMTS,B(T/S)GM->BGMTS
                    ret = ret + dout
            ret = ret.squeeze(1) 
        return ret  # BGMTS->BMTS

class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
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
        self.dyn_w_proj = DynamicWeightProjection(num_heads=self.n_head, query_input_dim=config.dim, dynamic_squeeze_ratio=self.n_head//2, dynamic_w_hidden_dim=self.n_head*4)
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim) # BSND
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v)) # BNSD

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        if seqlen == 1:
            logits = q @ k.transpose(-2, -1)
            B,T,D = x.shape

            #N,I = self.dyn_w_proj.num_heads_per_group, self.dyn_w_proj.dynamic_hidden_dim
            #shape = (B,T,2,2,-1,N) 
            #dw_hidden, dd = (x @ self.dyn_w_proj.dw_m).split([2*2*N*(2*I), 2*2*N*1], -1) # BT(4K), BT4N         # K=2*N*I
            #dw_hidden = self.dyn_w_proj.dw_hidden_activation(dw_hidden)
            #dw_hidden = dw_hidden.reshape(dw_hidden.shape[:2]+(4,-1)) #B T (4 K) -> B T 4 K
            #dw = torch.einsum('B T C K, C K D -> B T C D', dw_hidden, self.dyn_w_proj.qkw_m) # BT4K,4K(MI)->BT4(MI)
            #w1, w2 = dw.view(shape).split(I,-2) # BT22(2*I)N -> 2*[BT22IN]:BT(pre/post)(q/k)IN
            #w1 = self.dyn_w_proj.dw1_norm(w1) # BT22IN
            #qkw = torch.einsum(f'BTKJIN,BTKJIM->BTKJNM', w1, w2)  # j=k=2, BT2{2}NM
            #qkdd = self.dyn_w_proj.dw_activation(dd).squeeze(-1).view(shape[:-2]+(N,)) # BT4N->BT22N
            #qkw = qkw + torch.diag_embed(qkdd) # BT2{2}NN,->BT2{2}NN
            #qw, kw_new = qkw.unbind(3)  # BT2{2}NN-> [BT2NM]*2
            #self.kv_cache.kw_cache[:,input_pos] = kw_new # update kw cache
            ##print('w sum', self.dyn_w_proj.sw.shape, qw.shape, self.kv_cache.kw_cache.shape)
            #w = self.dyn_w_proj.sw + qw + self.kv_cache.kw_cache#+ torch.eye(N,dtype=qw.dtype, device=qw.device)  #(2,N,M)+ (B,1,2,N,M) + (B,S,2,N,M) + (N,M)
            #pre_w, post_w = w.unbind(2)
            #logits = torch.einsum('BNTS,BSNM->BMTS', logits, pre_w)
            #min_value = -65504.0
            #logits = torch.where(mask, logits, min_value)
            #probs = logits.softmax(-1)
            #probs = torch.einsum('BNTS,BSNM->BMTS', probs, post_w)
            #y = probs @ v

            # optimized in jit
            N,I = 32,2
            dw_hidden, dd = (x @ self.dyn_w_proj.dw_m).split([2*2*N*(2*I), 2*2*N*1], -1)
            dw_hidden = dw_hidden.view((B,T,4,-1,1))
            dw = (F.gelu(dw_hidden) * self.dyn_w_proj.qkw_m).sum(-2)
            shape = (B,T,2,2,-1,N)
            w1, w2 = dw.view(shape).split(I,-2)
            w1 = self.dyn_w_proj.dw1_norm(w1) # BT22IN
            qkdd = F.tanh(dd.view((B,T,2,2,N))) # BT2{2}N1->BT2{2}N
            qkw = torch.einsum('BTKJIN,BTKJIM->BTKJNM', w1, w2) + torch.diag_embed(qkdd) # j=k=2, BT2{2}NM
            qw, kw_new = qkw.unbind(2)
            self.kv_cache.kw_cache[:,input_pos] = kw_new # update kw cache
            w = self.dyn_w_proj.sw + qw + self.kv_cache.kw_cache#+ torch.eye(N,dtype=qw.dtype, device=qw.device)  #(2,N,M)+ (B,1,2,N,M) + (B,S,2,N,M) + (N,M)
            w = w.permute(0,2,3,4,1)  # BS2NM->B2NMS
            wl, w = w.unbind(1)
            logits = (logits * wl).sum(1).unsqueeze(2)
            min_value = -65504.0
            logits = torch.where(mask, logits, min_value)
            probs = logits.softmax(-1)
            probs = (probs * w).sum(1).unsqueeze(2)
            y = probs @ v
        else:
            #logits = self.dyn_w_proj.pre_proj(logits, dws=pre_proj_dw_args, query_vec=hidden_states, key_vec=hidden_states, proj_w=pre_w)  # XD BN1S
            #probs = logits.softmax(-1)
            #probs = self.dyn_w_proj.post_proj(probs, dws=post_proj_dw_args, query_vec=hidden_states, key_vec=hidden_states, proj_w=post_w) # BN1S
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
    seq_len: int, n_elem: int, base: int = 10000
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.float16)


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

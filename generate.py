# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

# --- Disable Torch compile/JIT entirely for mock testing (Windows-safe) ---
import torch
try:
    torch._dynamo.reset()
except Exception:
    pass

torch._dynamo.config.suppress_errors = True

# Disable compilation on all torch versions
if hasattr(torch, "_compile"):
    torch._compile = lambda *a, **kw: a[0] if a else None

import torch._inductor.config
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

# Optional speculative decoding import (your custom file)
from flex import speculative_generate

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet supported")

# Torch performance tweaks (safe on Windows)
# --- Disable all PyTorch compilation backends safely (works on all versions) ---
import os, torch

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

try:
    torch._dynamo.reset()
except Exception:
    pass

from torch.nn.attention.flex_attention import BlockMask, create_block_mask

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
create_block_mask = create_block_mask

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from tokenizer import get_tokenizer


# ------------------- Utility Functions -------------------

def multinomial_sample_one_no_sync(probs_sort):
    """Sample one token without CUDA sync."""
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def roundup(val, multiplier):
    return ((val - 1) // multiplier + 1) * multiplier

def causal_mask(b, h, q, kv):
    return q >= kv


# ------------------- Decoding Core -------------------

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    mask = create_block_mask(causal_mask, 1, 1, input_pos.shape[0], model.max_seq_length, device=x.device)
    logits = model(mask, x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, block_mask: BlockMask, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input_pos.shape[-1] == 1
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, model.max_seq_length)
    logits = model(mask, x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    block_mask = create_block_mask(causal_mask, 1, 1, model.max_seq_length, model.max_seq_length, device=cur_token.device)
    new_tokens, new_probs = [], []
    for _ in range(num_new_tokens):
        next_token, next_prob = decode_one_token(
            model, cur_token, input_pos, block_mask, **sampling_kwargs
        )
        input_pos += 1
        new_tokens.append(next_token.clone())
        callback(new_tokens[-1])
        new_probs.append(next_prob.clone())
        cur_token = next_token.clone()
    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)


# ------------------- Speculative Decode (Default) -------------------

def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)

    draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)
    draft_tokens = torch.cat(draft_tokens)

    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)

    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k] / p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0:
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        model_forward(draft_model, draft_tokens[-1].view(1, -1), orig_input_pos + speculate_k)
        return torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])


# ------------------- Generation -------------------

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    *,
    interactive: bool,
    draft_model: Optional[Transformer] = None,
    speculate_k: Optional[int] = 8,
    callback=lambda x: x,
    use_flex: bool = False,
    **sampling_kwargs
) -> torch.Tensor:
    is_speculative = draft_model is not None
    T = prompt.size(-1)
    T_new = T + max_new_tokens
    max_seq_length = 350 if interactive else min(T_new, model.config.block_size)
    device, dtype = prompt.device, prompt.dtype
    max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length

    model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
    if is_speculative and draft_model is not model:
        draft_model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    empty = torch.empty(batch_size, T_new, dtype=dtype, device=device)
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs).clone()

    if is_speculative:
        prefill(draft_model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs)

    seq[:, T] = next_token.squeeze()
    input_pos = torch.tensor([T], device=device, dtype=torch.int64)
    accept_counts = [0] * (speculate_k + 1)

    if is_speculative:
        input_pos = input_pos.item()
        while input_pos < T_new - 1:
            cur_token = next_token.view(())
            if use_flex:
                next_tokens_seq, _ = speculative_generate(
                    target_model=model,
                    draft_model=draft_model,
                    input_ids=seq[:, :input_pos + 1],
                    max_new_tokens=speculate_k,
                    eos_token_id=None,
                    temperature=sampling_kwargs.get("temperature", 0.0),
                    top_p=sampling_kwargs.get("top_k", None),
                )
                next_tokens = next_tokens_seq[0, input_pos + 1:]
            else:
                next_tokens = speculative_decode(model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs)

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(T_new - input_pos - 1, len(next_tokens))
            seq[input_pos + 1: input_pos + num_added + 1] = next_tokens[:num_added]
            for token in next_tokens[:num_added]:
                callback(token)
            input_pos += num_added
            next_token = next_tokens[-1]
    else:
        generated_tokens, _ = decode_n_tokens(model, next_token.view(batch_size, -1), input_pos, max_new_tokens - 1, callback=callback, **sampling_kwargs)
        seq[:, T + 1:] = torch.cat(generated_tokens, dim=-1)

    return seq, {"accept_counts": accept_counts}


# ------------------- Model Helpers -------------------

def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        bos_token = getattr(tokenizer, "bos_token_id", None) or getattr(tokenizer, "bos_id", None)
        if bos_token is not None:
            tokens = [bos_token] + tokens
    return torch.tensor(tokens, dtype=torch.int64, device=device)


# ✅ Dummy loader for local FlexDecoding test (mock model)
def _load_model(checkpoint_path, device, precision, use_tp):
    print("⚠️ Using DummyModel (mock) for FlexDecoding pipeline test.")
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self, vocab_size=4096):
            super().__init__()
            self.vocab_size = vocab_size
            self.param = nn.Parameter(torch.zeros(1))

        def setup_caches(self, max_batch_size=None, max_seq_length=None):
            return

        @property
        def max_seq_length(self):
            return 256

        @property
        def config(self):
            class C: block_size = 256
            return C()

        def forward(self, mask, x=None, input_pos=None):
            device = mask.device if hasattr(mask, "device") else torch.device("cpu")
            bsz, seq_len = (1, 1) if x is None else x.shape[:2]
            vocab_idx = torch.arange(self.vocab_size, device=device).float().unsqueeze(0).unsqueeze(0)
            logits = vocab_idx * 0.001 + torch.arange(seq_len, device=device).float().unsqueeze(0).unsqueeze(-1) * 0.01
            return logits.expand(bsz, seq_len, self.vocab_size)

    return DummyModel(vocab_size=4096).eval()


# ------------------- CLI -------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run GPT-Fast text generation (Mock Flex test).')

    def int_or_str(x):
        try:
            return int(x)
        except:
            return x

    parser.add_argument('--prompt', type=int_or_str, default="Hello, my name is")
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--checkpoint_path', type=Path, required=True)
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None)
    parser.add_argument('--speculate_k', type=int, default=5)
    parser.add_argument('--device', type=str, default=default_device)
    parser.add_argument('--flex', action='store_true', help='Use FlexDecoding (speculative_generate)')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    tokenizer_path = args.checkpoint_path.parent / "tokenizer.model"
    try:
        tokenizer = get_tokenizer(tokenizer_path, args.checkpoint_path)
    except Exception:
        print("⚠️ SentencePiece tokenizer not found — using Hugging Face tokenizer instead.")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path.parent)

    model = _load_model(args.checkpoint_path, args.device, torch.bfloat16, use_tp=False)
    draft_model = _load_model(args.draft_checkpoint_path, args.device, torch.bfloat16, use_tp=False) if args.draft_checkpoint_path else None

    torch.manual_seed(args.seed)
    encoded = encode_tokens(tokenizer, args.prompt, bos=True, device=args.device)
    y, stats = generate(
        model,
        encoded,
        args.max_new_tokens,
        batch_size=args.batch_size,
        interactive=args.interactive,
        draft_model=draft_model,
        speculate_k=args.speculate_k,
        use_flex=args.flex,
        temperature=args.temperature,
        top_k=args.top_k
    )
    print("\nOutput:\n", tokenizer.decode(y[0].tolist()))

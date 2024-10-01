# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import re
import shutil
import sys
import os
from pathlib import Path
from typing import Optional
from safetensors.torch import load_file as load_safetensors_file
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf"),
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    model_map_json_safetensors = checkpoint_dir / 'model.safetensors.index.json'
    model_map_json_pytorch = checkpoint_dir / "pytorch_model.bin.index.json"
    model_map_json = None

    try:
        assert model_map_json_safetensors.is_file()
        model_map_json = model_map_json_safetensors
        print(f"Found safetensors index at {model_map_json_safetensors}")
    except AssertionError:
        print(f"{model_map_json_safetensors} not found")
        if model_map_json is None:
            try:
                assert model_map_json_pytorch.is_file()
                model_map_json = model_map_json_pytorch
                print(f"Found pytorch index at {model_map_json_pytorch}")
            except AssertionError:
                print(f"{model_map_json_pytorch} not found")

    # If the solo safetensors file exists, we should load it directly
    model_solo_safetensors = checkpoint_dir / 'model.safetensors'
    if model_solo_safetensors.is_file():
        print(f"Found whole safetensors file at {model_solo_safetensors}")
        merged_result = load_safetensors_file(str(model_solo_safetensors), device="cpu")
    else:
        if model_map_json is None:
            raise Exception("No model map found!")

        with open(model_map_json) as json_map:
            bin_index = json.load(json_map)

        # Refactored merging logic into a separate function
        merged_result = merge_weights(checkpoint_dir, bin_index)

    # Refactored key mapping logic into a separate function
    final_result = map_keys(merged_result, config)

    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")

    if 'llama-3-' in model_name.lower() or 'llama-3.1-' or 'llama-3.2-' in model_name.lower():
        if 'llama-3.1-405b' in model_name.lower():
            original_dir = checkpoint_dir / "original" / "mp16"
        else:
            original_dir = checkpoint_dir / "original"
        tokenizer_model = original_dir / "tokenizer.model"
        tokenizer_model_tiktoken = checkpoint_dir / "tokenizer.model"
        print(f"Copying {tokenizer_model} to {tokenizer_model_tiktoken}")
        shutil.copy(tokenizer_model, tokenizer_model_tiktoken)

def merge_weights(checkpoint_dir, bin_index):
    bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    merged_result = {}
    for file in sorted(bin_files):
        if "safetensors" in str(file):
            state_dict = load_safetensors_file(str(file), device="cpu")
            merged_result.update(state_dict)
        else:
            state_dict = torch.load(str(file), map_location="cpu", mmap=True, weights_only=True)
            merged_result.update(state_dict)
    return merged_result

def map_keys(merged_result, config):
    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        'model.layers.{}.self_attn.rotary_emb.inv_freq': None,
        'model.layers.{}.mlp.gate_proj.weight': 'layers.{}.feed_forward.w1.weight',
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    final_result = {}
    for key, value in merged_result.items():
        if "layers" in key:
            abstract_key = re.sub(r'(\d+)', '{}', key)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map.get(abstract_key)
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map.get(key)

        if new_key is not None:
            final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            q = permute(q, config.n_head, config)
            k = permute(k, config.n_local_heads, config)
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]
    return final_result

def permute(w, n_head, config):
    dim = config.dim
    return (
        w.view(n_head, 2, config.head_dim // 2, dim)
        .transpose(1, 2)
        .reshape(config.head_dim * n_head, dim)
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path("checkpoints/meta-llama/llama-2-7b-chat-hf"))
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )

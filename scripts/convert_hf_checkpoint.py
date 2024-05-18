# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

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
    if "stories" in checkpoint_dir.name:
        # No need to convert tinyllamas
        return
    if model_name is None:
        model_name = checkpoint_dir.name

    # Llama 3 8B doesn't need conversion; instead, the original/consolidated.NN.pth files
    # need to be copied into model.pth.
    # Llama 3 70B can't be easily merged into one model.pth file, though, since names of the
    # weights is state dict are the same in each consolidated.NN.pth file. Thus, it is not
    # currently supported.
    # Along this, we need to copy the original/tokenizer.model file to tokenizer.model.tiktoken
    is_llama3 = "Llama-3" in model_name
    if is_llama3:
        # Check if we have multiple original/consolidated.NN.pth files and report error
        # if we do for Llama 3.
        original_dir = checkpoint_dir / "original"
        pattern = re.compile(r"^consolidated\.\d{2}\.pth$")
        bin_files = [bin for bin in original_dir.iterdir() if pattern.match(bin.name)]
        if len(bin_files) > 1:
            raise ValueError(
                f"Multiple consolidated.NN.pth files found in {original_dir}. "
                "Merging them into one model.pth file is not supported for Llama 3.")


    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    if not is_llama3:
        model_map_json = checkpoint_dir / "pytorch_model.bin.index.json"

        assert model_map_json.is_file()

        with open(model_map_json) as json_map:
            bin_index = json.load(json_map)

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
        bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    else:
        # There is no separate pytorch_model.bin.index.json file for llama3.
        # Instead, we will just use all original/consolidated.NN.pth files.
        # so, we use model.safetensors.index.json
        weight_map = None
        original_dir = checkpoint_dir / "original"
        pattern = re.compile(r"^consolidated\.\d{2}\.pth$")
        bin_files = {bin for bin in original_dir.iterdir() if pattern.match(bin.name)}
        

    def permute(w, n_head):
        dim = config.dim
        return (
            w.view(n_head, 2, config.head_dim // 2, dim)
            .transpose(1, 2)
            .reshape(config.head_dim * n_head, dim)
        )

    merged_result = {}
    for file in sorted(bin_files):
        state_dict = torch.load(str(file), map_location="cpu", mmap=True, weights_only=True)
        merged_result.update(state_dict)
    final_result = {}
    if weight_map is not None:
        for key, value in merged_result.items():
            if "layers" in key:
                abstract_key = re.sub(r'(\d+)', '{}', key)
                layer_num = re.search(r'\d+', key).group(0)
                new_key = weight_map[abstract_key]
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = weight_map[key]

            final_result[new_key] = value

        for key in tuple(final_result.keys()):
            if "wq" in key:
                q = final_result[key]
                k = final_result[key.replace("wq", "wk")]
                v = final_result[key.replace("wq", "wv")]
                q = permute(q, config.n_head)
                k = permute(k, config.n_local_heads)
                final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
                del final_result[key]
                del final_result[key.replace("wq", "wk")]
                del final_result[key.replace("wq", "wv")]
    else:
        final_result = merged_result
    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")
    if is_llama3:
        original_dir = checkpoint_dir / "original"
        tokenizer_model = original_dir / "tokenizer.model"
        tokenizer_model_tiktoken = checkpoint_dir / "tokenizer.model"
        print(f"Copying {tokenizer_model} to {tokenizer_model_tiktoken}")
        shutil.copy(tokenizer_model, tokenizer_model_tiktoken)

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

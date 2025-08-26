# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs

@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/micorosoft/Phi-3-mini-4k-instruct"),
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    model_map_json = checkpoint_dir / "model.safetensors.index.json"

    assert model_map_json.is_file()

    with open(model_map_json) as json_map:
        bin_index = json.load(json_map)

    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.qkv_proj.weight": "layers.{}.attention.wqkv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": "",
        "model.layers.{}.mlp.gate_up_proj.weight": "layers.{}.feed_forward.w1.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight"
    }
    bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}

    def permute(w, n_head):
        dim = config.dim
        return (
            w.view(n_head, 2, config.head_dim // 2, dim)
            .transpose(1, 2)
            .reshape(config.head_dim * n_head, dim)
        )

    merged_result = {}
    for file in sorted(bin_files):
        state_dict = load_file(str(file))
        merged_result.update(state_dict)
    final_result = {}
    for key, value in merged_result.items():
        print(key)
        if "layers" in key:
            abstract_key = re.sub(r'(\d+)', '{}', key, count=1)
            layer_num = re.search(r'\d+', key).group(0)

            new_key = weight_map[abstract_key]

            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "wqkv" in key:
            qkv = final_result[key]
            q, k, v = qkv.chunk(3, dim=0)
            q = permute(q, config.n_head)
            k = permute(k, config.n_head)
            final_result[key] = torch.cat([q, k, v])

    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path("checkpoints/microsoft/Phi-3-mini-4k-instruct"))
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )

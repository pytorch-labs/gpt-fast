# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Transformer, ConditionalFeedForward

##### Quantization Primitives ######

def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scales and zero_points based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scales is the same dtype as the original tensor
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scales/zp
    # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scales, zero_points


##### Weight-only int8 per-channel quantized code ######

def replace_linear_weight_only_bit8_per_channel(module, target_dtype):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name != "gate":
            setattr(module, name, WeightOnlyBit8Linear(child.in_features, child.out_features, target_dtype=target_dtype))
        elif isinstance(child, ConditionalFeedForward):
            num_experts, intermediate_size, dim = child.w1.shape
            setattr(module, name, ConditionalFeedForwardBit8(num_experts, intermediate_size, dim, target_dtype=target_dtype))
        else:
            replace_linear_weight_only_bit8_per_channel(child, target_dtype)

class WeightOnlyBit8QuantHandler:
    def __init__(self, mod, target_dtype):
        self.mod = mod
        self.target_dtype = target_dtype

    @torch.no_grad()
    def create_quantized_state_dict(self):
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear) and not fqn.endswith(".gate"):
                int8_weight, scales, _ = dynamically_quantize_per_channel(mod.weight.float(), -128, 127, self.target_dtype)
                cur_state_dict[f"{fqn}.weight"] = int8_weight
                cur_state_dict[f"{fqn}.scales"] = scales.to(mod.weight.dtype)
            elif isinstance(mod, ConditionalFeedForward):
                for weight_idx in range(0, 3):
                    weight_name = f"w{weight_idx + 1}"
                    scales_name = f"scales{weight_idx + 1}"
                    weight = getattr(mod, weight_name)
                    num_experts, intermediate_size, dim = weight.shape

                    bit8_weight_list = []
                    scales_list = []
                    for expert_idx in range(num_experts):
                        bit8_weight, scales, _ = dynamically_quantize_per_channel(weight[expert_idx].float(), -128, 127, self.target_dtype)
                        bit8_weight_list.append(bit8_weight.reshape(1, intermediate_size, dim))
                        scales_list.append(scales.reshape(1, intermediate_size))

                    cur_state_dict[f"{fqn}.{weight_name}"] = torch.cat(bit8_weight_list, dim=0)
                    cur_state_dict[f"{fqn}.{scales_name}"] = torch.cat(scales_list, dim=0)

        return cur_state_dict

    def convert_for_runtime(self):
        replace_linear_weight_only_bit8_per_channel(self.mod, self.target_dtype)
        return self.mod


class WeightOnlyBit8Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, target_dtype=None) -> None:
        assert target_dtype is not None
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=target_dtype))
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales


class ConditionalFeedForwardBit8(nn.Module):
    def __init__(self, num_experts, intermediate_size, dim, target_dtype):
        super().__init__()

        self.target_dtype = target_dtype

        self.register_buffer("w1", torch.empty(num_experts, intermediate_size, dim, dtype=target_dtype))
        self.register_buffer("w2", torch.empty(num_experts, dim, intermediate_size, dtype=target_dtype))
        self.register_buffer("w3", torch.empty(num_experts, intermediate_size, dim, dtype=target_dtype))

        self.register_buffer("scales1", torch.empty(num_experts, intermediate_size, dtype=torch.bfloat16))
        self.register_buffer("scales2", torch.empty(num_experts, dim, dtype=torch.bfloat16))
        self.register_buffer("scales3", torch.empty(num_experts, intermediate_size, dtype=torch.bfloat16))

    def forward(self, x, expert_indices):
        w1_weights = self.w1.to(x.dtype)[expert_indices] # [T, A, D, D]
        w3_weights = self.w3.to(x.dtype)[expert_indices] # [T, A, D, D]
        w2_weights = self.w2.to(x.dtype)[expert_indices]
        x1 = F.silu(torch.einsum('ti,taoi -> tao', x, w1_weights) * self.scales1[expert_indices].to(x.dtype))
        x3 = torch.einsum('ti, taoi -> tao', x, w3_weights) * self.scales3[expert_indices].to(x.dtype)
        expert_outs = torch.einsum('tao, taio -> tai', (x1 * x3), w2_weights) * self.scales2[expert_indices].to(x.dtype)  # [T, A, D, D]
        return expert_outs


def quantize(
    checkpoint_path: Path = Path("checkpoints/mistralai/Mixtral-8x7B-v0.1/model.pth"),
    mode: str = 'int8',
    label: str = '',
) -> None:
    assert checkpoint_path.is_file(), checkpoint_path

    device = 'cpu'
    precision = torch.bfloat16

    print("Loading model ...")
    t0 = time.time()

    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(dtype=precision, device=device)

    if mode == 'int8':
        print("Quantizing model weights for int8 weight-only symmetric per-channel quantization")
        quant_handler = WeightOnlyBit8QuantHandler(model, target_dtype=torch.int8)
        quantized_state_dict = quant_handler.create_quantized_state_dict()

        dir_name = checkpoint_path.parent
        base_name = checkpoint_path.name
        new_base_name = base_name.replace('.pth', f'{label}int8.pth')

    else:
        raise ValueError(f"Invalid quantization mode {mode} needs to be one of [int8,]")

    quantize_path = dir_name / new_base_name
    print(f"Writing quantized weights to {quantize_path}")
    quantize_path.unlink(missing_ok=True) # remove existing file if one already there
    torch.save(quantized_state_dict, quantize_path)
    print(f"Quantization complete took {time.time() - t0:.02f} seconds")
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Quantize a model.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"), help='Path to the model checkpoint to be quantized.')
    parser.add_argument('--mode', '-q', type=str, default='int8', choices=['int8', 'int4', 'int4-gptq'], help='type of quantization to perform')
    parser.add_argument('--label', type=str, default='_', help='label to add to output filename')

    args = parser.parse_args()
    quantize(args.checkpoint_path, args.mode, args.label)

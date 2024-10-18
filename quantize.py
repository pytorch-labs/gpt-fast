# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import get_tokenizer

try:
    from GPTQ import GenericGPTQRunner, InputRecorder
    from eval import get_task_dict, evaluate, lm_eval
except:
    pass

from model import Transformer

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

def get_group_qparams(w, n_bit=4, groupsize=128):
    # needed for GPTQ with padding
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    zeros = min_val + scales * (2 ** (n_bit - 1))
    return scales.to(torch.bfloat16).reshape(w.shape[0], -1), zeros.to(
        torch.bfloat16
    ).reshape(w.shape[0], -1)


def pack_scales_and_zeros(scales, zeros):
    assert scales.shape == zeros.shape
    assert scales.dtype == torch.bfloat16
    assert zeros.dtype == torch.bfloat16
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )


def unpack_scales_and_zeros(scales_and_zeros):
    assert len(scales_and_zeros.shape) == 3 and scales_and_zeros.shape[2] == 2
    assert scales_and_zeros.dtype == torch.float
    return torch.split(scales_and_zeros.transpose(0, 1), 1, 2)


def group_quantize_tensor_from_qparams(w, scales, zeros, n_bit=4, groupsize=128):
    assert groupsize > 1
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    min_val = zeros - scales * (2 ** (n_bit - 1))
    max_int = 2**n_bit - 1
    min_int = 0
    w_int32 = (
        to_quant.sub(min_val)
        .div(scales)
        .round()
        .clamp_(min_int, max_int)
        .to(torch.int32)
        .reshape_as(w)
    )

    return w_int32


def group_quantize_tensor(w, n_bit=4, groupsize=128):
    scales, zeros = get_group_qparams(w, n_bit, groupsize)
    w_int32 = group_quantize_tensor_from_qparams(w, scales, zeros, n_bit, groupsize)
    scales_and_zeros = pack_scales_and_zeros(scales, zeros)
    return w_int32, scales_and_zeros


def group_dequantize_tensor_from_qparams(
    w_int32, scales, zeros, n_bit=4, groupsize=128
):
    assert groupsize > 1
    # needed for GPTQ single column dequantize
    if groupsize > w_int32.shape[-1] and scales.shape[-1] == 1:
        groupsize = w_int32.shape[-1]
    assert w_int32.shape[-1] % groupsize == 0
    assert w_int32.dim() == 2

    w_int32_grouped = w_int32.reshape(-1, groupsize)
    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)

    w_dq = (
        w_int32_grouped.sub(2 ** (n_bit - 1)).mul(scales).add(zeros).reshape_as(w_int32)
    )
    return w_dq


def group_dequantize_tensor(w_int32, scales_and_zeros, n_bit=4, groupsize=128):
    scales, zeros = unpack_scales_and_zeros(scales_and_zeros)
    return group_dequantize_tensor_from_qparams(
        w_int32, scales, zeros, n_bit, groupsize
    )

class QuantHandler:
    def __init__(self, mod):
        self.mod = mod

    def create_quantized_state_dict(self) -> "StateDict":
        pass

    def convert_for_runtime(self) -> "nn.Module":
        pass

class GPTQQuantHandler(QuantHandler):
    """
    This class implements a GPTQ QuantHandler that can be used to apply GPTQ to a model in concert with the GenericGPTQRunner class.
    Unlike the base QuantHandler class, the user does not need to implement the create_quantized_state_dict, instead they have to reimplement
    __init__ such that it defines the functions for the quantization mode. User is expected to reimplement convert_for_runtime.

    The following functions (which must be defined in __init__) are used to define the quantization mode for both GPTQ and
    create_quantized_state_dict. Here is a description of each function.

    get_qparams_func:
        A function that calculates the quantization qparams for an input tensor.
        Args:
            weight: A 2d weight tensor with non-integer dtype.
        Returns:
            qparams: it can have any format but will need to be handled by the other defined functions below.

    quantize_func:
        A function that applies quantization to an input tensor. It should be noted
        that this function needs to be able to handle quantizing the entire weight tensor, a single group,
        or a single column.
        Args:
            weight: A 2d weight tensor with non-integer dtype.
            qparams: the output from get_qparams_func
        Returns:
            quantized_weight: A 2d quantized weight tensor (generally with an integer dtype)


    dequantize_func:
        A function that dequantizes an input quantized weight tensor. It should be noted
        that this function needs to be able to handle dequantizing the entire weight tensor, a single group,
        or a single column.
        Args:
            quantized_weight: A 2d quantized weight tensor (generally with an integer dtype)
            qparams: the output from get_qparams_func
        Returns:
            weight: A 2d weight tensor with non-integer dtype.

    combine_qparams_list_func:
        A function that combines several qparams into one qparam.
        Args:
            qparams_list: a list of qparams objects, each obtained by calling get_qparams_func
            on a single group from a weight tensor
        Returns:
            qparams: an object of the same format as the qparams above.

    skip_layer_func:
        A function that determines which linear layers should be skipped during GPTQ
        Args:
            weight: A 2d weight tensor with non-integer dtype.
        Returns:
            skip: boolean indicating whether layer should be skipped

    make_names_and_values_dict_func:
        A function that prepares the qparams and quantized_weight and creates a dictionary indicating how they
        should be inserted into the state_dict. Generally any packing of the weight and qparams should be done here.
        Args:
            quantized_weight: A 2d quantized weight tensor (generally with an integer dtype)
            qparams: the output from get_qparams_func
        Returns:
            names_and_values_dict: a dictionary mapping the name of the parameters of the quantized module to the
            corresponding quantized weights and qparams.
    """
    def __init__(self):
        assert self.mod is not None
        assert self.get_qparams_func is not None
        assert self.quantize_func is not None
        assert self.dequantize_func is not None
        assert self.combine_qparams_list_func is not None
        assert self.make_names_and_values_dict_func is not None

    @staticmethod
    def get_inputs(model, tokenizer, calibration_tasks, calibration_limit, calibration_seq_length, pad_calibration_inputs) -> "MultiInput":
        input_recorder = InputRecorder(
            model,
            tokenizer,
            calibration_seq_length,
            pad_calibration_inputs,
        )

        try:
            lm_eval.tasks.initialize_tasks()
        except:
            pass
        task_dict = get_task_dict(calibration_tasks)
        print("Obtaining GPTQ calibration inputs on: ", calibration_tasks)

        evaluate(
            input_recorder,
            task_dict,
            limit=calibration_limit,
        )
        inputs = input_recorder.get_recorded_inputs()
        assert inputs is not None, (
            f"No inputs were collected, use a task other than {calibration_tasks}, "+
            f"use option pad_calibration_inputs, or decrease calibration_sequence_length (currently "+
            f"{calibration_seq_length})"
        )
        print(f"Obtained {len(inputs[0].values)} calibration samples")
        return inputs

    @torch.no_grad()
    def create_quantized_state_dict(
        self,
        tokenizer,
        blocksize,
        percdamp,
        groupsize,
        calibration_tasks,
        calibration_limit,
        calibration_seq_length,
        pad_calibration_inputs,
    ) -> "StateDict":
        inputs = GPTQQuantHandler.get_inputs(self.mod, tokenizer, calibration_tasks, calibration_limit, calibration_seq_length, pad_calibration_inputs)
        print("Tracing model for GPTQ")
        GPTQ_runner = GenericGPTQRunner(
            self.mod,
            inputs,
            blocksize,
            percdamp,
            groupsize,
        ).configure_quantization_mode(
            self.get_qparams_func,
            self.quantize_func,
            self.dequantize_func,
            self.combine_qparams_list_func,
            self.make_names_and_values_dict_func,
            self.skip_layer_func
        )

        print("Applying GPTQ to weights")
        GPTQ_runner.run()
        return GPTQ_runner.get_quantized_state_dict()

    def convert_for_runtime(self) -> "nn.Module":
        pass

##### Weight-only int8 per-channel quantized code ######

def replace_linear_weight_only_int8_per_channel(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, WeightOnlyInt8Linear(child.in_features, child.out_features))
        else:
            replace_linear_weight_only_int8_per_channel(child)

class WeightOnlyInt8QuantHandler:
    def __init__(self, mod):
        self.mod = mod

    @torch.no_grad()
    def create_quantized_state_dict(self):
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear):
                int8_weight, scales, _ = dynamically_quantize_per_channel(mod.weight.float(), -128, 127, torch.int8)
                cur_state_dict[f"{fqn}.weight"] = int8_weight
                cur_state_dict[f"{fqn}.scales"] = scales.to(mod.weight.dtype)

        return cur_state_dict

    def convert_for_runtime(self):
        replace_linear_weight_only_int8_per_channel(self.mod)
        return self.mod


class WeightOnlyInt8Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales

##### weight only int4 per channel groupwise quantized code ######

def prepare_int4_weight_and_scales_and_zeros(weight_bf16, groupsize, inner_k_tiles):
    weight_int32, scales_and_zeros = group_quantize_tensor(
        weight_bf16, n_bit=4, groupsize=groupsize
    )
    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(weight_int32, inner_k_tiles)
    return weight_int4pack, scales_and_zeros


def linear_forward_int4(x, weight_int4pack, scales_and_zeros, out_features, groupsize):
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    c = torch.ops.aten._weight_int4pack_mm(x, weight_int4pack, groupsize, scales_and_zeros)
    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c


def _check_linear_int4_k(k, groupsize = 1, inner_k_tiles = 1):
    return k % groupsize == 0 and k % (inner_k_tiles * 16) == 0

def replace_linear_int4(module, groupsize, inner_k_tiles, padding):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if _check_linear_int4_k(child.in_features, groupsize, inner_k_tiles):
                setattr(module, name, WeightOnlyInt4Linear(
                    child.in_features, child.out_features, bias=False,
                    groupsize=groupsize, inner_k_tiles=inner_k_tiles, padding=False,
                ))
            elif padding:
                setattr(module, name, WeightOnlyInt4Linear(
                    child.in_features, child.out_features, bias=False,
                    groupsize=groupsize, inner_k_tiles=inner_k_tiles, padding=True,
                ))
        else:
            replace_linear_int4(child, groupsize, inner_k_tiles, padding)


class WeightOnlyInt4QuantHandler:
    def __init__(self, mod, groupsize=128, inner_k_tiles=8, padding=True):
        self.mod = mod
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.padding = padding
        assert groupsize in [32, 64, 128, 256]
        assert inner_k_tiles in [2, 4, 8]

    @torch.no_grad()
    def create_quantized_state_dict(self, use_cuda = True):
        if use_cuda:
            device="cuda"
        else:
            device="cpu"

        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear):
                assert not mod.bias
                out_features = mod.out_features
                in_features = mod.in_features
                assert out_features % 8 == 0, "require out_features % 8 == 0"
                print(f"linear: {fqn}, in={in_features}, out={out_features}")

                weight = mod.weight.data
                if not _check_linear_int4_k(in_features, self.groupsize, self.inner_k_tiles):
                    if self.padding:
                        from model import find_multiple
                        import torch.nn.functional as F
                        print(f"warning: {fqn} is padded to satisfy in_features % 1024 == 0")
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(weight, pad=(0, padded_in_features - in_features))
                    else:
                        print(f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, " +
                            "and that groupsize and inner_k_tiles*16 evenly divide into it")
                        continue
                weight_int4pack, scales_and_zeros = prepare_int4_weight_and_scales_and_zeros(
                    weight.to(torch.bfloat16).to(device=device), self.groupsize, self.inner_k_tiles
                )
                cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to('cpu')
                cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to('cpu')

        return cur_state_dict

    def convert_for_runtime(self):
        replace_linear_int4(self.mod, self.groupsize, self.inner_k_tiles, self.padding)
        return self.mod

class WeightOnlyInt4GPTQQuantHandler(GPTQQuantHandler):
    def __init__(self, mod, groupsize=128, inner_k_tiles=8, padding=True):
        from model import find_multiple
        self.mod = mod
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.padding = padding
        self.get_qparams_func = lambda w: get_group_qparams(w, 4, groupsize)
        self.quantize_func = lambda w, qparams: \
            group_quantize_tensor_from_qparams(w, qparams[0], qparams[1], 4, groupsize)
        self.dequantize_func = lambda q, qparams: \
            group_dequantize_tensor_from_qparams(q, qparams[0], qparams[1], 4, groupsize).float()
        self.combine_qparams_list_func = lambda qparams_list: \
            [torch.cat(x, dim=1) for x in zip(*qparams_list)]
        # skip unless padding=True or its correctly sized
        self.skip_layer_func = lambda linear_weight: not (
            _check_linear_int4_k(linear_weight.shape[-1], groupsize, inner_k_tiles) or padding
        )
        # we need to do the padding here, both for q and the qparams if necessary
        def make_names_and_values_dict_func(q, qparams):
            k = q.shape[1]
            new_k = find_multiple(k, 1024)
            # how much we need to pad the weight
            delta_k = new_k - q.shape[1]
            final_q = torch.ops.aten._convert_weight_to_int4pack(F.pad(q, pad=(0, delta_k)), inner_k_tiles)
            scales_and_zeros = pack_scales_and_zeros(*qparams)
            # how many new groups we need for padded weight
            delta_groups = new_k // groupsize - scales_and_zeros.shape[0]
            final_s_and_z = F.pad(scales_and_zeros, pad=(0,0,0,0,0, delta_groups), value=1)
            return {"weight": final_q, "scales_and_zeros": final_s_and_z}
        self.make_names_and_values_dict_func = make_names_and_values_dict_func
        super().__init__()


    def convert_for_runtime(self):
        replace_linear_int4(self.mod, self.groupsize, self.inner_k_tiles, self.padding)
        return self.mod

class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
            self, in_features: int, out_features: int,
            bias=True, device=None, dtype=None, groupsize: int = 128, inner_k_tiles: int = 8, padding: bool = True,
    ) -> None:
        super().__init__()
        self.padding = padding
        if padding:
            from model import find_multiple
            self.origin_in_features = in_features
            in_features = find_multiple(in_features, 1024)

        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles

        assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert in_features % (inner_k_tiles * 16) == 0, "require in_features % (innerKTiles * 16) == 0"
        self.register_buffer(
            "weight",
            torch.empty((out_features // 8, in_features // (inner_k_tiles * 16), 32, inner_k_tiles // 2), dtype=torch.int32)
        )
        self.register_buffer(
            "scales_and_zeros",
            torch.empty((in_features // groupsize, out_features, 2), dtype=torch.bfloat16)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.bfloat16)
        if self.padding:
            import torch.nn.functional as F
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(
            input,
            self.weight, self.scales_and_zeros, self.out_features, self.groupsize
        )


def quantize(
    checkpoint_path: Path = Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"),
    mode: str = 'int8',
    # following arguments only available when setting int4 quantization.
    groupsize: int = 128,
    # following arguments only used for GPTQ
    calibration_tasks: list = ["hellaswag"],
    calibration_limit: int = 1000,
    calibration_seq_length: int = 100,
    pad_calibration_inputs: bool = False,
    percdamp: float = .01,
    blocksize: int = 128,
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

    dir_name = checkpoint_path.parent
    base_name = checkpoint_path.name

    if 'torchao-int4' in mode:
        import torchao
        from torchao.quantization import (quantize_, int4_weight_only)            
        use_hqq = 'hqq' in mode
        print(f"Quantizing model weights for int4 weight-only symmetric per-channel quantization {'with hqq' if use_hqq else ''}")
        quantize_(model, int4_weight_only(group_size=groupsize, use_hqq=use_hqq), device='cuda')
        quantized_state_dict = model.state_dict()
        new_base_name = base_name.replace('.pth', f'{label}{mode}.pth')
    elif 'torchao-int8' in mode:
        import torchao
        from torchao.quantization import (quantize_, int8_weight_only)      
        print("Quantizing model weights for int8 weight-only symmetric per-channel quantization")
        quantize_(model, int8_weight_only())
        quantized_state_dict = model.state_dict()
        new_base_name = base_name.replace('.pth', f'{label}{mode}.pth')
    elif mode == 'int8':
        print("Quantizing model weights for int8 weight-only symmetric per-channel quantization")
        quant_handler = WeightOnlyInt8QuantHandler(model)
        quantized_state_dict = quant_handler.create_quantized_state_dict()
        new_base_name = base_name.replace('.pth', f'{label}int8.pth')
    elif mode == 'int4':
        print("Quantizing model weights for int4 weight-only affine per-channel groupwise quantization")
        quant_handler = WeightOnlyInt4QuantHandler(model, groupsize)
        quantized_state_dict = quant_handler.create_quantized_state_dict()
        new_base_name = base_name.replace('.pth', f"{label}int4.g{groupsize}.pth")

    elif mode == 'int4-gptq':
        print("Quantizing model weights for int4 weight-only affine per-channel groupwise quantization using GPTQ...")
        quant_handler = WeightOnlyInt4GPTQQuantHandler(model, groupsize)

        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), str(tokenizer_path)
        tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)

        quantized_state_dict = quant_handler.create_quantized_state_dict(
            tokenizer,
            blocksize,
            percdamp,
            groupsize,
            calibration_tasks,
            calibration_limit,
            calibration_seq_length,
            pad_calibration_inputs
        )
        new_base_name = base_name.replace('.pth', f"{label}int4-gptq.g{groupsize}.pth")
    else:
        raise ValueError(f"Invalid quantization mode {mode} needs to be one of [int8, int4, int4-gpptq, torchao-int4, torchao-int8, torchao-int4-hqq]")

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
    parser.add_argument('--mode', '-q', type=str, default='int8', choices=['int8', 'int4', 'int4-gptq', 'torchao-int4', 'torchao-int8', 'torchao-int4-hqq'], help='type of quantization to perform')
    parser.add_argument('--groupsize', type=int, default=32, help='Group size for int4 quantization.')
    parser.add_argument('--calibration_tasks', type=str, nargs='+', default=['wikitext'], help='tasks to do gptq calibration on, if doing gptq')
    parser.add_argument('--calibration_limit', type=int, default=1000, help='number of samples to use for gptq calibration')
    parser.add_argument('--calibration_seq_length', type=int, default=100, help='length of sequences to use for gptq calibration')
    parser.add_argument('--pad_calibration_inputs', type=bool, default=False, help='pads sequences shorter than calibration_seq_length to that length, yielding more calibration inputs but running much slower')
    parser.add_argument('--percdamp', type=float, default=.01, help='gptq percentage dampening')
    parser.add_argument('--blocksize', type=int, default=128, help='blocksize for gptq')
    parser.add_argument('--label', type=str, default='_', help='label to add to output filename')

    args = parser.parse_args()
    quantize(args.checkpoint_path, args.mode, args.groupsize, args.calibration_tasks, args.calibration_limit, args.calibration_seq_length, args.pad_calibration_inputs, args.percdamp, args.blocksize, args.label)

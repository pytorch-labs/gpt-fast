# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_flatten, tree_unflatten

aten = torch.ops.aten

from eval import (
    setup_cache_padded_seq_input_pos_max_seq_length_for_prefill,
    GPTFastEvalWrapper
)


class InputRecorder(GPTFastEvalWrapper):
    """
    This is a fake evaluation wrapper that just records the inputs
    so that they can be used in calibration.

    If pad_calibration_inputs is enabled, the input recorder will take
    each input and pad/truncate it down to the calibration_seq_length.
    It will also edit the model embeddings to be zero for the 0 token used
    in padding and avoid any inputs with the 0 token.

    If not, it will only truncate inputs to the desired length.
    """

    def __init__(
        self,
        model,
        tokenizer,
        calibration_seq_length,
        pad_calibration_inputs=False,
    ):
        super().__init__(model, tokenizer, calibration_seq_length)
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device("cpu")
        self.vocab_size = model.config.vocab_size
        self.calibration_seq_length = calibration_seq_length
        self.pad_calibration_inputs = pad_calibration_inputs
        self.inputs = None

        if self.pad_calibration_inputs:
            # This is needed for the pad_calibration_inputs option
            # to work properly, the 0 token's embeddings are set to 0 so that
            # the padded inputs will not affect the model numerics. This token isn't used
            # commonly in the eval tasks for the meta-llama tokenizer and we skip any inputs
            # where it appears
            try:
                if isinstance(self._model.transformer.wte, nn.Embedding):
                    self.mod.transformer.wte.weight.data[0, :] *= 0
            except:
                print(
                    "Did not find embeddings in model.transformer.wte, disabling padding"
                )
                self.pad_calibration_inputs = False


    def add_input(self, args):
        if self.inputs is None:
            self.inputs = [MultiInput([arg]) for arg in args]
        else:
            self.inputs = [
                multi.add_input(arg) for (multi, arg) in zip(self.inputs, args)
            ]

    def get_recorded_inputs(self):
        return self.inputs

    def _model_call(self, inps):
        inps = inps.squeeze(0)
        T = len(inps)
        if (
            # can't use inputs that are too short when padding disabled
            (T < self.calibration_seq_length and not self.pad_calibration_inputs)
            or
            # can't use inputs that actually use token we use for padding
            (self.pad_calibration_inputs and 0 in inps)
        ):
            # give random output
            return torch.randn(
                (1, T, self.vocab_size), dtype=torch.bfloat16, device=self._device
            )

        # pad or truncate to the right size
        if T >= self.calibration_seq_length:
            inps = inps[: self.calibration_seq_length]
        else:
            inps = F.pad(inps, (0, self.calibration_seq_length - T))

        max_new_tokens = 1
        (
            seq,
            input_pos,
            max_seq_length,
        ) = setup_cache_padded_seq_input_pos_max_seq_length_for_prefill(
            self._model, inps, max_new_tokens, self.max_length
        )
        x = seq.index_select(0, input_pos).view(1, -1)
        self.add_input((x, input_pos))

        # output `something` with correct shape to keep eval going
        return torch.randn(
            (1, T, self.vocab_size), dtype=torch.bfloat16, device=self._device
        )



class MultiInput:
    def __init__(self, inputs):
        self.values = list(inputs)

    def add_input(self, input):
        self.values.append(input)
        return self

    def __getitem__(self, slice):
        return MultiInput(self.values[slice])

    def cuda(self):
        self.values = [val.cuda() if isinstance(val, torch.Tensor) else val for val in self.values]


class GenericGPTQRunner(fx.Interpreter):
    """
    This is a generic GPTQ runner that takes an existing model and applies GPTQ.
    It uses torch._dynamo.export to obtain a graph of the model and then hooks
    into function calls and when it detects a linear, it applies GPTQ to the weight
    given the calibration of inputs passed in at initialization. It puts the results
    into the state_dict so that the quantized model weights/qparams can be loaded
    directly into the model.

    This class is expected to work in concert with a GPTQSimpleQuantizer
    class to define the specific type of quantization being done.
    """

    def __init__(
        self, model, inputs: MultiInput, blocksize=128, percdamp=0.01, groupsize=128
    ):
        self.id_to_name = {
            id(value): name for name, value in dict(model.named_parameters()).items()
        }

        # trace model for one input
        one_input = [multi.values[0].cpu() for multi in inputs]
        exported_model = torch._dynamo.export(
            model.cpu(), aten_graph=True, pre_dispatch=True, tracing_mode="fake"
        )(*one_input)
        super().__init__(exported_model.graph_module)
        self.new_state_dict = model.state_dict()
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.groupsize = groupsize
        self.inputs = inputs
        self.gptq_done = False
        self.debug = False

    def configure_quantization_mode(
        self,
        get_qparams_func,
        quantize_func,
        dequantize_func,
        combine_qparams_list_func,
        make_names_and_values_dict_func,
        skip_layer_func,
    ):
        # these functions need to already be curried with all inputs other than weight, qparams
        self.get_qparams_func = (
            get_qparams_func  # accepts [2d weight tensor], outputs qparams.
        )

        self.quantize_func = quantize_func  # accepts [2d weight tensor], [qparams], outputs a 2d quantized tensor of desired dtype

        self.dequantize_func = dequantize_func
        # accepts [quantized] tensor and [qparams], outputs a 2d dequantized tensor of type float,
        # assumes this output .to(w_orig_dtype) is ~eventual desired dequant behavior

        self.combine_qparams_list_func = combine_qparams_list_func
        # accepts [`list` of qparams] from quantizing one group at a time,
        # outputs a qparams object that could be passed into quant/dequantize_func

        self.skip_layer_func = skip_layer_func  # accepts [weight tensor], outputs a bool on whether or not to apply gptq to this layer

        self.make_names_and_values_dict_func = make_names_and_values_dict_func  # accepts [2d quantized tensor], [qparams], returns a dict of names, values to put in state_dict
        # note any final packing for storage should happen here
        return self

    def run(self):
        assert (
            self.get_qparams_func is not None
        ), "need to configure quantization mode before running"
        self.gptq_done = True
        super().run(*self.inputs)

    def get_quantized_state_dict(self):
        assert (
            self.gptq_done
        ), "need to run GPTQRunner before you can get_quantized_state_dict"
        quantized_state_dict = self.new_state_dict
        # Don't want to store/load the kv_cache so remove it from the state_dict
        del_list = []
        for param_fqn in quantized_state_dict:
            if "kv_cache" in param_fqn:
                del_list.append(param_fqn)
        for param_fqn in del_list:
            quantized_state_dict.pop(param_fqn)
        return quantized_state_dict

    def call_function(self, target, args, kwargs, skip_quant=False):
        def tensors_to_cuda(args):
            new_args = []
            for x in args:
                new_args.append(x.cuda() if isinstance(x, torch.Tensor) else x)
            return new_args

        # flatten args and kwargs together
        flat_args, spec = tree_flatten((args, kwargs))
        # move all single tensors to cuda, will move MultiInputs to cuda one at a time
        flat_args = tensors_to_cuda(flat_args)

        has_multi_input = MultiInput in [type(x) for x in flat_args]
        if has_multi_input:
            # Just some trickery to convert
            # [MultiInput[a, a, a], MultiInput(b, b, b)] => [a, b], [a, b], [a, b]
            multi_input_count = max(
                [len(x.values) if isinstance(x, MultiInput) else 1 for x in flat_args]
            )
            transposed_args = list(
                zip(
                    *[x.values if isinstance(x, MultiInput) else [x] * multi_input_count for x in flat_args]
                )
            )
        else:
            transposed_args = [flat_args]
        outputs = []

        # check whether we apply GPTQ to this module
        quantize_linear = (
            (target == aten.linear.default) # if its a linear
            and id(args[1]) in self.id_to_name # and if we know the layer name
            and not skip_quant  # and if we weren't told to skip quantization
            # and if the skip_layer_func doesn't say we should skip
            and not (self.skip_layer_func is not None and self.skip_layer_func(args[1]))
        )  # then we will quantize this linear layer/weight

        if quantize_linear:  # instantiate variables for GPTQ
            H = 0
            total_batches = 0

        for inp in transposed_args:
            inp = tensors_to_cuda(inp)
            cur_args, cur_kwargs = tree_unflatten(inp, spec)

            if (
                quantize_linear
            ):  # calculate H instead of output (will run the linear eventually with updated weight)
                x = cur_args[0].float()
                shape = x.shape
                n = 1 if len(shape) == 2 else shape[0]
                H *= total_batches / (total_batches + n)
                total_batches += n
                x = ((2 / total_batches) ** (1 / 2)) * x.reshape(
                    -1, shape[-1]
                ).t().float()
                H += x.matmul(x.t())
            else:
                # get output if its not a linear
                out = super().call_function(target, cur_args, cur_kwargs)

                if isinstance(out, torch.Tensor):
                    outputs.append(out.cpu())
                else:
                    outputs.append(out)

        if quantize_linear:
            mod_fqn = ".".join(self.id_to_name[id(args[1])].split(".")[:-1])
            W = args[1].to(H.device)
            Q, DQ, qparams = self.faster_quant(H, W.detach())
            print(mod_fqn)
            names_and_values_dict = self.make_names_and_values_dict_func(Q, qparams)

            # delete old weight
            if mod_fqn + ".weight" in self.new_state_dict:
                self.new_state_dict.pop(mod_fqn + ".weight")
            if len(args) > 2:
                self.new_state_dict[mod_fqn + ".bias"] = args[2]
            for name, value in names_and_values_dict.items():
                self.new_state_dict[mod_fqn + "." + name] = value

            # run linear with new weight to get corrected output
            new_out = self.call_function(
                target, (args[0], DQ, *args[2:]), kwargs, skip_quant=True
            )

            if self.debug:
                old_out = self.call_function(
                    target, (args[0][:2], args[1], *args[2:]), kwargs, skip_quant=True
                )

                def SQNR(x, y):
                    return 20 * torch.log10(torch.norm(x) / torch.norm(x - y))

                DQ_after = self.dequantize_func(Q, qparams).to(W.dtype)
                print(
                    "SQNR for QDQ (this should be inf)", SQNR(DQ, DQ_after)
                )  # matches

                print(
                    "SQNR for weight (can be low)", SQNR(W, DQ.cuda())
                )  # fine to not match
                print(
                    "SQNR for output with GPTQ (hopefully 35+)",
                    torch.cat(
                        [
                            SQNR(old.cpu(), new.cpu()).unsqueeze(0)
                            for (old, new) in zip(old_out.values, new_out.values[:2])
                        ]
                    ).mean(),
                )

                qparams2 = self.get_qparams_func(W)
                Q2 = self.quantize_func(W, qparams2)
                DQ2 = self.dequantize_func(Q2, qparams2).to(W.dtype)
                old_q_out = self.call_function(
                    target, (args[0][:2], DQ2, *args[2:]), kwargs, skip_quant=True
                )

                print("SQNR for output without GPTQ (should be less than above)",
                    torch.cat([
                            SQNR(old.cpu(), old_q.cpu()).unsqueeze(0)
                            for (old, old_q) in zip(old_out.values, old_q_out.values)
                    ]).mean(),
                )
            return new_out

        return MultiInput(outputs) if has_multi_input else outputs[0]

    def faster_quant(self, H, W):
        percdamp = self.percdamp
        blocksize = self.blocksize
        groupsize = self.groupsize
        orig_dtype = W.dtype
        W = W.detach().float()
        rows, columns = W.shape[0], W.shape[1]
        device = W.device

        if groupsize == -1:
            cur_qparams = self.get_qparams_func(W)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        DQ = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        all_qparams = []
        for i1 in range(0, columns, blocksize):
            i2 = min(i1 + blocksize, columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            DQ1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1 and (i1 + i) % groupsize == 0:  # start of new group
                    cur_qparams = self.get_qparams_func(
                        W[:, (i1 + i) : (i1 + i + groupsize)]
                    )
                    all_qparams.append(cur_qparams)

                q = self.quantize_func(w.unsqueeze(1), cur_qparams).flatten()
                dq = self.dequantize_func(q.unsqueeze(1), cur_qparams).flatten()

                DQ1[:, i] = dq
                Losses1[:, i] = (w - dq) ** 2 / d**2

                err1 = (w - dq) / d
                W1[:, i:] -= (
                    err1.to(Hinv1.dtype).unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                )
                Err1[:, i] = err1

            DQ[:, i1:i2] = DQ1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.to(Hinv.dtype).matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if all_qparams == []:
            all_qparams.append(cur_qparams)

        # convert a list of qparams objects into a single one. enerally by
        # concatenating a bunch of n,1 scale/zeros tensors into a n,num_groups tensor
        all_qparams = self.combine_qparams_list_func(all_qparams)
        Q = self.quantize_func(DQ, all_qparams)
        return Q, DQ.to(orig_dtype), all_qparams

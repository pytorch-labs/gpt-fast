# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config
from torch.nn.utils.rnn import pad_sequence

torch._dynamo.config.automatic_dynamic_shapes = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.epilogue_fusion = False
torch._dynamo.config.cache_size_limit = 100000

from tokenizer import get_tokenizer

from model import Transformer
from generate import generate

try:
    import lm_eval
    lm_eval_available = True
except:
    lm_eval_available = False

from generate import _load_model, encode_tokens, model_forward

if lm_eval_available:
    try: # lm_eval version 0.4
        from lm_eval.models.huggingface import HFLM as eval_wrapper
        from lm_eval.tasks import get_task_dict
        from lm_eval.evaluator import evaluate
    except: #lm_eval version 0.3
        from lm_eval import base
        from lm_eval import tasks
        from lm_eval import evaluator
        eval_wrapper=base.BaseLM
        get_task_dict=tasks.get_task_dict
        evaluate=evaluator.evaluate


def setup_cache_padded_seq_input_pos_max_seq_length_for_prefill(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    max_seq_length: Optional[int] = None,
):
    """
    Sets up model cache and does some bookkeeping calculations for prompt, input_pos and max_seq_length
    that are needed for prefill or model_forward

    Args:
        model (LLaMA): The model whose cache gets set up
        prompt (torch.Tensor): Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens (int): The desired maximum number of new tokens that can be generated.
        max_seq_length (Optional[int], optional): The maximum sequence length allowed.

    Returns:
        seq (torch.Tensor): prompt but padded with zeros to size max_seq_length
        input_pos (torch.Tensor): tensor of integers in increasing order
        max_seq_length (int): The maximum sequence length allowed, updated based on other numbers
    """
    T = prompt.size(0)
    T_new = T + max_new_tokens
    if max_seq_length is None:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    return seq, input_pos, max_seq_length

class GPTFastEvalWrapper(eval_wrapper):
    """
    A wrapper class for GPTFast, providing integration with the lm-evaluation-harness library.
    """
    def __init__(
        self,
        model: Transformer,
        tokenizer,
        max_seq_length: Optional[int]=None,
        batch_size: int = 1,
    ):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device('cuda')
        self._max_seq_length = 2048 if max_seq_length is None else max_seq_length
        self._batch_size = batch_size

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id()

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 50

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs):
        encoded = encode_tokens(self._tokenizer,
            string, bos=True, device=self._device)
        # encoded is a pytorch tensor, but some internal logic in the
        # eval harness expects it to be a list instead
        # TODO: verify this for multi-batch as well
        encoded = encoded.tolist()
        return encoded

    def tok_batch_encode(
        self, text: List[str], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized_text = [self.tok_encode(x) for x in text]

        # pad left
        x = pad_sequence(
            [
                torch.tensor(x[::-1]) for x in tokenized_text
            ],  # first flip each sequence and pad
            batch_first=True,
            padding_value=self._tokenizer.pad_id(),
        ).flip(
            dims=[1]
        )  # flip back to correct order

        return x, torch.ones_like(x)  # return 'mask' b/c it's expected by the harness

    def tok_decode(self, tokens):
        decoded = self._tokenizer.decode(tokens)
        return decoded

    def _model_call(self, inps):
        # TODO: make batches work
        inps = inps.squeeze(0)

        max_new_tokens = 1
        seq, input_pos, max_seq_length = \
            setup_cache_padded_seq_input_pos_max_seq_length_for_prefill(
                self._model,
                inps,
                max_new_tokens,
                self.max_length,
            )
        x = seq.index_select(0, input_pos).view(1, -1)
        logits = model_forward(self._model, x, input_pos)
        return logits

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        curr_batch_size = context.size(0)
        assert curr_batch_size == 1, "Currently generation only supports batch size of 1. Provided prompt has batch size {curr_batch_size}."

        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        # TODO: handle top_p and top_k

        # Setup caches for a given batch size
        # Technically this is not necessary, but it's a good way to ensure that
        # the caches won't error on a different batch size. In addition, caches
        # are not needed for a regular model call, so we just setup here
        # TODO: call setup_cache_padded_seq_input_pos_max_seq_length_for_prefill() instead
        with context.device:
            self._model.setup_caches(max_batch_size=curr_batch_size, max_seq_length=max_length)

        # TODO: currently, the generate() function assumes 1D tensor with batch size 1. Need to update it to accept 2D tensor
        context = context.flatten(0)

        toks, accept_counts = generate(
            self._model,
            context,
            max_new_tokens=self.max_gen_toks,
            interactive=False,
            draft_model=None,
            temperature=generation_kwargs["temperature"],
            # top_k=None,  # do_sample is not supported currently
            # stop_tokens=self._tokenizer.stop_tokens,
        )

        # TODO: output from generate() is 1D tensor with batch size 1. Need to update to return 2D tensor.
        toks = toks.unsqueeze(0)

        return torch.tensor(toks, dtype=torch.int32)


@torch.no_grad()
def eval(
    model: Transformer,
    tokenizer,
    tasks: list = ["hellaswag"],
    limit: Optional[int] = None,
    max_seq_length: Optional[int] = None,
) -> dict:
    """
    Evaluates a language model on a specified task using the lm-evaluation-harness library.

    Args:
        model (Transformer): The pre-trained language model to evaluate.
        tokenizer: The tokenizer to use for encoding/decoding text.
        tasks (list): The names of the evaluation tasks to perform.
        limit (Optional[int]): The maximum number of samples to evaluate (None for all available).
        max_seq_length (Optional[int]): The maximum sequence length allowed for input text.

    Returns:
        eval_results (dict): A dictionary of evaluation results for the specified task(s).
    """
    model_eval_wrapper = GPTFastEvalWrapper(
        model,
        tokenizer,
        max_seq_length,
    )

    try:
        lm_eval.tasks.initialize_tasks()
    except:
        pass

    if 'hendrycks_test' in tasks:
        tasks.remove('hendrycks_test')
        tasks += [x for x in lm_eval.tasks.hendrycks_test.create_all_tasks().keys()]
    task_dict = get_task_dict(tasks)

    eval_results = evaluate(
        model_eval_wrapper,
        task_dict,
        limit=limit,
    )
    return eval_results


def main(
    checkpoint_path: Path = Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/lit_model.pth"),
    compile: bool = False,
    tasks: list = ["hellaswag"],
    limit: Optional[int] = None,
    max_seq_length: Optional[int] = None,
) -> None:
    """Evaluates model on a task from the `lm-evaluation-harness` library.

    Args:
        checkpoint_path (Path): The path to the model checkpoint file to load.
        compile (bool): Whether or not to compile the model for optimization.
        tasks (list): The names of the evaluation tasks to perform.
        limit (Optional[int]): The maximum number of samples to evaluate (None for all available).
        max_seq_length (Optional[int]): The maximum sequence length allowed for input text.

    """

    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    device = 'cuda'
    precision = torch.bfloat16

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, False)

    torch.cuda.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds.")

    model.eval()

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)

    torch.manual_seed(1234)

    if compile:
        global model_forward
        model_forward = torch.compile(model_forward,  mode="reduce-overhead", dynamic=True, fullgraph=True)
        torch._inductor.config.coordinate_descent_tuning = True

    t1 = time.time()
    result = eval(
        model,
        tokenizer,
        tasks,
        limit,
        max_seq_length,
    )
    print(f"Time to run eval: {time.time() - t1:.02f} seconds.")
    print(f"For model {checkpoint_path}")
    for task, res in result["results"].items():
        print(f"{task}: {res}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/lit_model.pth"), help='Model checkpoint path.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--tasks', nargs='+', type=str, default=["hellaswag"], help='list of lm-eluther tasks to evaluate usage: --tasks task1 task2')
    parser.add_argument('--limit', type=int, default=None, help='number of samples to evalulate')
    parser.add_argument('--max_seq_length', type=int, default=None, help='maximum length sequence to evaluate')

    args = parser.parse_args()
    main(
        Path(args.checkpoint_path), args.compile, args.tasks, args.limit, args.max_seq_length,
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
from collections.abc import Iterable

import torch
import torch._dynamo.config
import torch._inductor.config

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from tokenizer import get_tokenizer

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = 1.0, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    probs = torch.nn.functional.softmax(logits, dim=-1)

    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = 1.0):
    probs = logits_to_probs(logits[0, -1], temperature, top_k, top_p)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_one_token_early(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model.forward_early(x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: False, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if model.early_num_layers < model.num_layers:
                next_token, next_prob = decode_one_token_early(
                    model, cur_token, input_pos, **sampling_kwargs
                )
            else:
                next_token, next_prob = decode_one_token(
                    model, cur_token, input_pos, **sampling_kwargs
                )
            input_pos += 1
            new_tokens.append(next_token.clone())
            if callback(next_token):
                break
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)

def forward_remainder(model, x, input_pos):
    return model.forward_remainder(x, input_pos)

def forward_early(model, x, input_pos):
    return model.forward_early(x, input_pos)

def self_speculative_decode(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    draft_tokens, draft_probs = decode_n_tokens(model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)

    draft_tokens = torch.cat(draft_tokens)
    # parallel inference on target model using draft tokens
    target_logits = forward_remainder(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        forward_early(
            model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
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

def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)

    draft_tokens = torch.cat(draft_tokens)
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
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

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    is_self_speculative: bool = False,
    callback = lambda x: False,
    max_seq_len: Optional[int] = -1,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    is_speculative = draft_model is not None
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)
    T_new = T + max_new_tokens

    if max_seq_len == -1:
        if interactive:
            max_seq_length = 350
        else:
            max_seq_length = min(T_new, model.config.block_size)
    else:
        max_seq_length = max_seq_len

    max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    print("\nSetting max_seq_length to ", max_seq_length)

    # truncate to avoid error
    if T_new > max_seq_length:
        T_new = max_seq_length

    device, dtype = prompt.device, prompt.dtype

    accept_counts = [0] * (speculate_k + 1)

    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
        if is_speculative and draft_model is not model:
            draft_model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    if T > max_seq_length:
        print(f"WARNING: size of prompt {prompt.size()} is greater than max_seq_length {max_seq_length}. Not generating tokens for this sample.")
        return prompt, {'accept_counts': 0}

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs).clone()
    if is_speculative:
        prefill(draft_model, prompt.view(1, -1), input_pos, **sampling_kwargs)
    seq[T] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    if is_speculative:
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        while input_pos < T_new - 1:
            cur_token = next_token.view(())

            if is_self_speculative:
                next_tokens = self_speculative_decode(
                    model, cur_token, input_pos, speculate_k, **sampling_kwargs
                )
            else:
                next_tokens = speculative_decode(
                    model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
                )

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(T_new - input_pos - 1, len(next_tokens))
            generation_done = False
            for i in next_tokens[: num_added,]:
                generation_done = callback(i)
            seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
            input_pos = input_pos + num_added
            next_token = next_tokens[-1]

            if generation_done:
                break
        seq = seq[:input_pos]
    else:
        generated_tokens, _ = decode_n_tokens(model, next_token.view(1, -1), input_pos, max_new_tokens - 1, callback=callback, **sampling_kwargs)
        seq[T + 1: T + 1 + len(generated_tokens)] = torch.cat(generated_tokens)
        seq = seq[:T + 1 + len(generated_tokens)]

    generate_stats = {
        'accept_counts': accept_counts
    }
    return seq, generate_stats

def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def _load_model(checkpoint_path, device, precision, use_tp, early_exit: int = -1, model_name: str = None):
    use_cuda = 'cuda' in device
    with torch.device('meta'):
        if model_name is None:
            model_name = checkpoint_path.parent.name
        model = Transformer.from_name(model_name, early_exit=early_exit)

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 weight-only quantization!")
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True, strict=False)

    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()

def _get_model_size(model):
    model_size = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
    return model_size

# TODO: add more details from where it got copied
def stop_at_stop_words(decoded_string, stop_words):
    """
    Produces the prefix of decoded_string that ends at the first occurrence of
    a stop_word.
    WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
    itself.
    """
    min_stop_index = len(decoded_string)
    for stop_word in stop_words:
        stop_index = decoded_string.find(stop_word)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]

B_INST, E_INST = "[INST]", "[/INST]"

def main(
    prompts: Union[str, List[str]] = "Hello, my name is",
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    top_k: int = 200,
    top_p: float = 1.0,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
    compile: bool = True,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    draft_checkpoint_path: Optional[Path] = None,
    draft_early_exit: Optional[int] = None,
    speculate_k: int = 5,
    self_speculative: bool = False,
    early_exit: int = -1,
    device=default_device,
    log_results: Optional[Path] = None,
    log_generations: Optional[Path] = None,
    model_name: Optional[str] = None,
    stop_words: Optional[List[str]] = None,
    max_seq_len: Optional[int] = -1,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    if not isinstance(prompts, Iterable):
        prompts = [prompts]

    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    global print
    from tp import maybe_init_dist
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    print(f"Using device={device}")
    precision = torch.bfloat16
    is_speculative = draft_checkpoint_path is not None
    is_chat = "chat" in str(checkpoint_path)

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp, early_exit=early_exit if self_speculative else -1, model_name=model_name)

    if is_speculative:
        draft_model = _load_model(draft_checkpoint_path, device, precision, use_tp)
        if draft_early_exit is not None and draft_early_exit > -1:
            draft_model.layers = draft_model.layers[0:draft_early_exit]
            draft_model.num_layers = draft_early_exit
    elif self_speculative:
        draft_model = model
        is_speculative = True
    else:
        draft_model = None

    device_sync(device=device) # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path.parent.name if model_name is None else model_name)

    if stop_words:
        stop_words_ids = [tokenizer.encode(stop_word) for stop_word in stop_words]
        for i in range(len(stop_words_ids)):
            stop_word_ids = stop_words_ids[i]
            # Remove control sequences from stop_ids that are not detected if stop_word exists in the middle of a sequence
            for id in stop_word_ids:
                if tokenizer.encode(tokenizer.decode(id)) == []:
                    stop_word_ids.remove(id)
            stop_words_ids[i] = stop_word_ids
        # stop_words_ids.append([tokenizer.eos_id()])
        stop_words_ids_length = torch.tensor([len(stop_word_ids) for stop_word_ids in stop_words_ids], device=device)
        max_stop_words_ids_length = max(stop_words_ids_length)
        stop_words_ids = [torch.tensor(stop_word_ids, device=device) for stop_word_ids in stop_words_ids]
        # stop_words_to_compare = torch.nn.utils.rnn.pad_sequence(stop_words_ids, batch_first=True)
        stop_words_to_compare = torch.nn.utils.rnn.pad_sequence([ids.flip(dims=[0]) for ids in stop_words_ids], batch_first=True).flip(dims=[1])
    eos_id = torch.tensor([tokenizer.eos_id()], device=device)

    if max_seq_len == -1:
        prompt_lengths = [encode_tokens(tokenizer, prompt, bos=True, device=device).size(0) for prompt in prompts]
        max_prompt_length = max(prompt_lengths)
        max_seq_len = max_prompt_length + max_new_tokens

    torch.manual_seed(1234)
    model_size = _get_model_size(model)
    if compile:
        if is_speculative and use_tp: # and ("cuda" in device):
            torch._inductor.config.triton.cudagraph_trees = False # Bug with cudagraph trees in this case

        if is_speculative:
            global model_forward, logits_to_prob
            model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)

            global decode_one_token_early
            decode_one_token_early = torch.compile(decode_one_token_early, mode="reduce-overhead", fullgraph=True)
            global forward_remainder
            forward_remainder = torch.compile(forward_remainder, fullgraph=True, dynamic=True)
            global forward_early
            forward_early = torch.compile(forward_early, mode="reduce-overhead", fullgraph=True)

        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        # Uncomment to squeeze more perf out of prefill
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)


    aggregate_metrics = {
        'tokens_per_sec': [],
        'accept_counts': [],
        'time_for_inference': []
    }
    start = -1 if compile else 0
    prompt = prompts[0]
    max_seq_len_check = -1

    if log_generations:
        generations = []

    for i in range(start, num_samples):
        device_sync(device=device) # MKG
        if i >= 0:
            if interactive:
                prompt = input("What is your prompt? ")
                if is_chat:
                    prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            else:
                if i < len(prompts):
                    prompt = prompts[i]
        max_seq_len_check = max(max_seq_len_check, len(prompt))
        encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
        prompt_length = encoded.size(0)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode('.')[0]
            done_generating = False
            def callback(x: torch.Tensor):
                nonlocal done_generating
                if done_generating:
                    return done_generating
                # TODO: optimize to handle calling x.item()
                buffer.append(tokenizer.decode([period_id] + x.item())[1:])
                if torch.equal(x, eos_id):
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print(''.join(buffer), end='', flush=True)
                    buffer.clear()
                # print(, end='', flush=True)
                return done_generating
        else:
            done_generating = False
            stop_ids_buffer = torch.empty(0, device=device, dtype=torch.int32)
            check_stop_words_period = 2
            def callback(x: torch.Tensor):
                nonlocal done_generating, stop_ids_buffer, check_stop_words_period
                if done_generating:
                    return True
                if stop_words:
                    stop_ids_buffer = torch.cat([stop_ids_buffer, x])
                    if stop_ids_buffer.numel() >= max_stop_words_ids_length * check_stop_words_period:
                        ## Check stop words by ids
                        # buffer_to_check = stop_ids_buffer.repeat(len(stop_words_ids), 1)
                        # stop_words_match = (buffer_to_check == stop_words_to_compare).sum(dim=1)
                        # if torch.any(stop_words_match >= stop_words_ids_length):
                        #     done_generating = True
                        #     return True

                        # Check stop words by string
                        decoded = tokenizer.decode(stop_ids_buffer.tolist())
                        for stop_word in stop_words:
                            if stop_word in decoded:
                                done_generating = True
                                return True

                        stop_ids_buffer = stop_ids_buffer[(check_stop_words_period - 1) * max_stop_words_ids_length:]
                return False
        t0 = time.perf_counter()
        import contextlib
        if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            y, metrics = generate(
                model,
                encoded,
                max_new_tokens,
                draft_model=draft_model,
                speculate_k=speculate_k,
                interactive=interactive,
                callback=callback,
                max_seq_len=max_seq_len,
                temperature=temperature,
                is_self_speculative=self_speculative,
                top_k=top_k,
                top_p=top_p,
            )
            aggregate_metrics['accept_counts'].append(metrics['accept_counts'])
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        if hasattr(prof, "export_chrome_trace"):
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")
        device_sync(device=device) # MKG
        t = time.perf_counter() - t0

        if not interactive:
            y_dec = y.tolist()
            if tokenizer.eos_id() in y_dec:
                y_dec = y_dec[:y_dec.index(tokenizer.eos_id()) + 1]
            decoded = tokenizer.decode(y_dec)
            if stop_words:
                decoded = prompt + stop_at_stop_words(decoded.removeprefix(prompt), stop_words)
            print(decoded)
            if log_generations:
                generations.append([decoded])
        else:
            print()
        tokens_generated = len(y) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(tokens_sec)
        aggregate_metrics['time_for_inference'].append(t)
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
        print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
    print("==========")
    print("max_seq_len_check: ", max_seq_len_check)
    if is_speculative:
        print(aggregate_metrics)
        counts_aggregated = [sum(i) for i in zip(*aggregate_metrics['accept_counts'])]
        acceptance_probs = [i/sum(counts_aggregated) for i in counts_aggregated]
        print(f"Acceptance probs: {acceptance_probs}")
        print(f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}")

    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    print(f"Average timer for inference: {torch.mean(torch.tensor(aggregate_metrics['time_for_inference'])).item():.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    aggregate_metrics["memory_used"] = torch.cuda.max_memory_reserved()

    if log_results:
        # Create parent directory if needed
        log_results.parents[0].mkdir(parents=True, exist_ok=True)
        # Save config and results to file
        with open(log_results, "w") as f:
            json.dump(aggregate_metrics, f)

    if log_generations:
        # Create parent directory if needed
        log_generations.parents[0].mkdir(parents=True, exist_ok=True)
        # Save config and results to file
        with open(log_generations, "w") as f:
            json.dump(generations, f)

    return aggregate_metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--prompt', type=str, default="Hello, my name is", help='Input prompt.')
    parser.add_argument('--interactive', action='store_true', help='Whether to launch in interactive mode')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k for sampling.')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--model_name', type=str, default=None, help='Model name to help find the architecture of the model.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--speculate_k', type=int, default=5, help='Speculative execution depth.')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Draft checkpoint path.')
    parser.add_argument('--draft_early_exit', type=int, default=None, help='Early exit layer of draft model.')
    parser.add_argument('--self_speculative', action='store_true', help='Whether to use self speculative decoding')
    parser.add_argument('--early_exit', type=int, default=-1, help='The layer to exit early')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')
    parser.add_argument('--log_results', type=Path, default=None, help='Path to log results')
    parser.add_argument('--log_generations', type=Path, default=None, help='Path to log generations')
    parser.add_argument('--stop_words', type=str, nargs='+', default=None, help='Words to stop generating when encountered')

    args = parser.parse_args()
    main(
        args.prompt, args.interactive, args.num_samples, args.max_new_tokens, args.top_k, args.top_p,
        args.temperature, args.checkpoint_path, args.compile, args.compile_prefill, args.profile, args.draft_checkpoint_path, args.draft_early_exit,
        args.speculate_k, args.self_speculative, args.early_exit, args.device, args.log_results, args.log_generations, args.model_name, args.stop_words,
    )

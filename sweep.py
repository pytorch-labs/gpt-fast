import torch
from typing import List, Dict
from pathlib import Path
import pandas as pd
import subprocess
import shlex
import json
import jsonpickle
import tabulate

from data import get_data

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--log_dir', type=Path, required=True, help="Directory to log output")
    parser.add_argument('--prompt', type=str, default="Hello, my name is", help='Input prompt.')
    parser.add_argument('--dataset', type=str, default=None, help='Name of dataset')
    parser.add_argument('--data_path', type=Path, default=None, help='Dataset path (not really used)')
    parser.add_argument('--random_shuffle', action='store_true', help='Whether to randomly shuffle prompts of dataset')
    parser.add_argument('--seed', type=int, default=42, help='Seed for dataset reshuffling')
    parser.add_argument('--n_shot', type=int, default=0, help='Number of example shots in prompt')
    parser.add_argument('--interactive', action='store_true', help='Whether to launch in interactive mode')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k for sampling.')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--model_name', type=str, default=None, help='Model name to help find the architecture of the model.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--enable_flash', action='store_true', help='Whether to enable flash attention')
    parser.add_argument('--enable_mem_efficient', action='store_true', help='Whether to enable memory efficient attention')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--sdpa', type=str, help='Implementation type for scaled dot product attention')
    parser.add_argument('--speculate_k_start', type=int, default=5, help='Speculative execution depth start.')
    parser.add_argument('--speculate_k_end', type=int, default=5, help='Speculative execution depth end.')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Draft checkpoint path.')
    parser.add_argument('--early_exit_start', type=int, default=-1, help='Early exit layer of draft model.')
    parser.add_argument('--early_exit_end', type=int, default=-1, help='Early exit layer of draft model.')
    parser.add_argument('--self_speculative', action='store_true', help='Whether to use self speculative decoding')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')
    parser.add_argument('--sample_warmup', type=int, default=0, help='Number of samples at the beginning to skip when calculating average tokens per second')

    args = parser.parse_args()

    if args.dataset:
        evaluation_set = get_data(args.random_shuffle, args.num_samples, args.dataset, args.data_path, args.n_shot, args.seed)
        prompts = [example.input for example in evaluation_set]
    else:
        prompts = args.prompt

    is_speculative = args.draft_checkpoint_path is not None or args.self_speculative

    args.log_dir.mkdir(parents=True, exist_ok=False)
    with open(args.log_dir / "args.json", "w") as f:
        f.write(jsonpickle.encode(args))

    results: List[Dict] = []
    for speculate_k in range(args.speculate_k_start, args.speculate_k_end+1):
        for early_exit in range(args.early_exit_start, args.early_exit_end+1):
            log_results: Path = args.log_dir / f"{speculate_k}_{early_exit}.json"
            if args.dataset:
                subprocess.check_call(
                    shlex.split(
                        f"python benchmark.py --dataset={args.dataset} --n_shot={args.n_shot} {'--interactive' if args.interactive else ''} --num_samples={args.num_samples} --max_new_tokens={args.max_new_tokens} {'--top_k='+str(args.top_k) if args.top_k else ''} --top_p={args.top_p} --temperature={args.temperature} --checkpoint_path={args.checkpoint_path} {'--compile' if args.compile else ''} {'--compile_prefill' if args.compile_prefill else ''} {'--profile' if args.profile else ''} {'--sdpa='+str(args.sdpa) if args.sdpa else ''} --draft_checkpoint_path={args.draft_checkpoint_path} --draft_early_exit={draft_early_exit} --speculate_k={speculate_k} {'--self_speculative' if args.self_speculative else ''} --early_exit={args.early_exit} --device={args.device} --log_file={log_file}"
                    )
                )
            else:
                subprocess.check_call(
                    shlex.split(
                        f"python generate.py --prompt=\"{args.prompt}\" {'--interactive' if args.interactive else ''} --num_samples={args.num_samples} --max_new_tokens={args.max_new_tokens} {'--top_k='+str(args.top_k) if args.top_k else ''} --top_p={args.top_p} --temperature={args.temperature} --checkpoint_path={args.checkpoint_path} {'--compile' if args.compile else ''} {'--compile_prefill' if args.compile_prefill else ''} {'--profile' if args.profile else ''} {'--sdpa='+str(args.sdpa) if args.sdpa else ''} --draft_checkpoint_path={args.draft_checkpoint_path} --draft_early_exit={draft_early_exit} --speculate_k={speculate_k} {'--self_speculative' if args.self_speculative else ''} --early_exit={args.early_exit} --device={args.device} --log_file={log_file}"
                    )
                )

            performance_metrics = json.load(log_results.open())
            average_metrics = performance_metrics["average_metrics"]
            log_results.unlink()

            if is_speculative:
                counts_aggregated = average_metrics['Counts Aggregated']
                acceptance_probs = average_metrics['Acceptance probs']
                mean_accepted = average_metrics['Mean Accepted']

            average_tokens_per_second = average_metrics['Average tokens/sec']
            memory_used = average_metrics['Memory used (GB)']

            results.append({
                "speculate_k": speculate_k,
                "early_exit": early_exit,
                "counts_aggregated": counts_aggregated,
                "acceptance_probs": acceptance_probs,
                "mean_accepted": mean_accepted,
                "average_tokens_per_second": average_tokens_per_second,
                "memory_used": memory_used,
            })
            df = pd.DataFrame(results) 
            # Update table every iteration
            df.to_csv(args.log_dir / "log.csv", index=False)

    # Print summary table
    print("\n")
    header = results[0].keys()
    rows =  [x.values() for x in results]
    print(tabulate.tabulate(rows, header))

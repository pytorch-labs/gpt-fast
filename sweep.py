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
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--speculate_k_start', type=int, default=5, help='Speculative execution depth start.')
    parser.add_argument('--speculate_k_end', type=int, default=5, help='Speculative execution depth end.')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Draft checkpoint path.')
    parser.add_argument('--draft_early_exit_start', type=int, default=None, help='Early exit layer of draft model.')
    parser.add_argument('--draft_early_exit_end', type=int, default=None, help='Early exit layer of draft model.')
    parser.add_argument('--self_speculative', action='store_true', help='Whether to use self speculative decoding')
    parser.add_argument('--early_exit', type=int, default=-1, help='The layer to exit early')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')

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
        for draft_early_exit in range(args.draft_early_exit_start, args.draft_early_exit_end+1):
            log_file: Path = args.log_dir / f"{speculate_k}_{draft_early_exit}.json"
            subprocess.check_call(
                shlex.split(
                    f"python generate.py --prompt=\"{prompts}\" {'--interactive' if args.interactive else ''} --num_samples={args.num_samples} --max_new_tokens={args.max_new_tokens} --top_k={args.top_k} --temperature={args.temperature} --checkpoint_path={args.checkpoint_path} {'--compile' if args.compile else ''} {'--compile_prefill' if args.compile_prefill else ''} {'--profile' if args.profile else ''} --draft_checkpoint_path={args.draft_checkpoint_path} --draft_early_exit={draft_early_exit} --speculate_k={speculate_k} {'--self_speculative' if args.self_speculative else ''} --early_exit={args.early_exit} --device={args.device} --log_file={log_file}"
                )
            )

            aggregate_metrics = json.load(log_file.open())

            if is_speculative:
                counts_aggregated = [sum(i) for i in zip(*aggregate_metrics['accept_counts'])]
                acceptance_probs = [i/sum(counts_aggregated) for i in counts_aggregated]
                mean_accepted = {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}

            average_tokens_per_second = torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item()
            memory_used = torch.cuda.max_memory_reserved() / 1e9

            results.append({
                "speculate_k": speculate_k,
                "draft_early_exit": draft_early_exit,
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

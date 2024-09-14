import torch

from data import get_data, get_stop_words
from generate import main
from pathlib import Path

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--dataset', type=str, default="human_eval", help='Name of dataset')
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
    parser.add_argument('--sdpa', type=str, help='Implementation type for scaled dot product attention')
    parser.add_argument('--enable_flash', action='store_true', help='Whether to enable flash attention')
    parser.add_argument('--enable_mem_efficient', action='store_true', help='Whether to enable memory efficient attention')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--speculate_k', type=int, default=5, help='Speculative execution depth.')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Draft checkpoint path.')
    parser.add_argument('--draft_early_exit', type=int, default=None, help='Early exit layer of draft model.')
    parser.add_argument('--self_speculative', action='store_true', help='Whether to use self speculative decoding')
    parser.add_argument('--early_exit', type=int, default=-1, help='The layer to exit early')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')
    parser.add_argument('--log_results', type=Path, default=None, help='Path to log results')
    parser.add_argument('--log_generations', type=Path, default=None, help='Path to log generations')
    parser.add_argument('--max_seq_len', type=int, default=-1, help='Maximum sequence length')

    args = parser.parse_args()

    evaluation_set = get_data(args.random_shuffle, args.num_samples, args.dataset, args.data_path, args.n_shot, args.seed)
    num_samples = len(evaluation_set) if args.num_samples is None else args.num_samples
    prompts = [example.input for example in evaluation_set]
    stop_words = get_stop_words(args.dataset)
    main(
        prompts, args.interactive, num_samples, args.max_new_tokens, args.top_k, args.top_p,
        args.temperature, args.checkpoint_path, args.compile, args.compile_prefill, args.sdpa, args.enable_flash, args.enable_mem_efficient,
        args.profile, args.draft_checkpoint_path, args.draft_early_exit,
        args.speculate_k, args.self_speculative, args.early_exit, args.device, args.log_results, args.log_generations, args.model_name, stop_words, args.max_seq_len
    )

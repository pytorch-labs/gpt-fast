# 56.80
export MODEL_REPO=deepseek-ai/deepseek-coder-6.7b-instruct
export DRAFT_MODEL_REPO=deepseek-ai/deepseek-coder-1.3b-instruct
time python generate.py --compile  --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model_int8.pth  --checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth --speculate_k 6 --prompt "def quicksort(arr):" --max_new_tokens 200 --num_samples 50


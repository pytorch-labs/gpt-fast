# 49.26
export MODEL_REPO=meta-llama/Llama-2-70b-chat-hf
export DRAFT_MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
time python generate.py --compile  --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model_int4.g32.pth  --checkpoint_path checkpoints/$MODEL_REPO/model_int4.g32.pth --speculate_k 4 --prompt "def quicksort(arr):" --max_new_tokens 100 --num_samples 50 --temperature 0

export MODEL_REPO=meta-llama/Llama-2-70b-chat-hf
export DRAFT_MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
time torchrun --standalone --nproc_per_node=8 generate.py --compile  --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model_int8.pth  --checkpoint_path checkpoints/$MODEL_REPO/model.pth --speculate_k 5 --prompt "def quicksort(arr):" --max_new_tokens 200 --num_samples 50 --temperature 0

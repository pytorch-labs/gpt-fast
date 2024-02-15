# 56.80
export MODEL_REPO=codellama/CodeLlama-34b-Python-hf
export DRAFT_MODEL_REPO=codellama/CodeLlama-7b-Python-hf
time python generate.py --compile  --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model_int4.g32.pth  --checkpoint_path checkpoints/$MODEL_REPO/model.pth --speculate_k 6 --prompt "def quicksort(arr):" --max_new_tokens 200 --num_samples 50

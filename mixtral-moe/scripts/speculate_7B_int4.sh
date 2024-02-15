export MODEL_REPO=codellama/CodeLlama-7b-Python-hf
export DRAFT_MODEL_REPO=PY007/TinyLlama-1.1B-intermediate-step-480k-1T
time python generate.py --compile  --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model_int4.g32.pth  --checkpoint_path checkpoints/$MODEL_REPO/model_int4.g32.pth --speculate_k 5 --prompt "Hi my name is" --max_new_tokens 200 --num_samples 50  --temperature 0 --compile_prefill

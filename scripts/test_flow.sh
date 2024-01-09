#export MODEL_REPO=PY007/TinyLlama-1.1B-intermediate-step-480k-1T
export MODEL_REPO=codellama/CodeLlama-7b-Python-hf
#export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
#rm -r checkpoints/$MODEL_REPO
#python scripts/download.py --repo_id $MODEL_REPO
#python scripts/new.py --checkpoint_dir checkpoints/$MODEL_REPO
#python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$MODEL_REPO
#python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --max_new_tokens 100
#python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth --max_new_tokens 100

export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
rm -r checkpoints/$MODEL_REPO
python scripts/download.py --repo_id $MODEL_REPO
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$MODEL_REPO
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth --max_new_tokens 100

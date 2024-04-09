export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf

# python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --compile # working
# echo "base"
# export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
# python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4-gptq --calibration_tasks wikitext --calibration_limit 5
# python eval.py --checkpoint_path checkpoints/$MODEL_REPO/model_int4-gptq.g32.cuda.pth --tasks wikitext --limit 5


python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4-hqq
# python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model_int4-hqq.g32.cuda.pth --compile
python eval.py --checkpoint_path checkpoints/$MODEL_REPO/model_int4-hqq.g32.cuda.pth --tasks wikitext

python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4
# python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model_int4.g32.cuda.pth --compile
python eval.py --checkpoint_path checkpoints/$MODEL_REPO/model_int4.g32.cuda.pth --tasks wikitext

# python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4
# python eval.py --checkpoint_path checkpoints/$MODEL_REPO/model_int4.g32.cuda.pth --tasks wikitext --limit 5
# broken

# export MODEL_REPO=meta-llama/Llama-2-70b-chat-hf
# python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4
# python eval.py --checkpoint_path checkpoints/$MODEL_REPO/model_int4.g32.cuda.pth --tasks wikitext --limit 5
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model_int4.g32.cuda.pth

# python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4-gptq --calibration_tasks wikitext --calibration_limit 5

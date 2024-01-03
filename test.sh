export MODEL_REPO=/home/kf/checkpoints/Llama-2-7b-hf
#export DRAFT_MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
time python generate.py --compile --checkpoint_path $MODEL_REPO/model.pth --prompt "def quicksort(arr):" --max_new_tokens 128 --num_samples 3 --temperature 0.5 #--profile ./llama_talking_compile
#TORCH_COMPILE_DEBUG=1 time python generate.py --compile --checkpoint_path $MODEL_REPO/model.pth --prompt "def quicksort(arr):" --max_new_tokens 128 --num_samples 3 --temperature 0.5 --profile ./llama_orig_compile
#TORCH_COMPILE_DEBUG=1 time python generate.py --checkpoint_path $MODEL_REPO/model.pth --prompt "def quicksort(arr):" --max_new_tokens 128 --num_samples 3 --temperature 0.5 --profile ./llama_talking_no_compile

#TORCH_COMPILE_DEBUG=1 
#time python generate.py  --checkpoint_path $MODEL_REPO/model.pth --prompt "def quicksort(arr):" --max_new_tokens 128 --num_samples 3 --temperature 0.5 #--profile /home/kf/qingye_llama/gptfast_compile

#time python generate.py --checkpoint_path $MODEL_REPO/model.pth --prompt "def quicksort(arr):" --max_new_tokens 4 --num_samples 1 --temperature 0.5 --profile /home/kf/qingye_llama/gptfast_nocompile
#time python generate.py --checkpoint_path $MODEL_REPO/model.pth --prompt "def quicksort(arr):" --max_new_tokens 100 --num_samples 1 --temperature 0.5

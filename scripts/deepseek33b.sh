# 56.80
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0,1
export ENABLE_INTRA_NODE_COMM=1
export OMP_NUM_THREADS=16
export MODEL_REPO=deepseek-ai/deepseek-coder-33b-instruct
export DRAFT_MODEL_REPO=deepseek-ai/deepseek-coder-1.3b-instruct
time torchrun  --standalone --nproc_per_node=2 generate.py --compile  --compile_prefill --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model_int8.pth  --checkpoint_path checkpoints/$MODEL_REPO/model_int4.g32.pth --speculate_k 6 --prompt "def quicksort(arr):" --max_new_tokens 16000 --num_samples 10

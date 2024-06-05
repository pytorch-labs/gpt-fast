# LayerSkip Implementation on GPT-Fast

## Steps
- Setup Code and Environment
```
git clone git@github.com:mostafaelhoushi/gpt-fast.git
cd gpt-fast

# you can use the same environment you built for LayerSkip code base
conda activate layer_skip
```

- Setup Model

```
mkdir -p checkpoints/layer_skip/Llama-2-7b-hf/
cp -r <path to Llama 7B>/* checkpoints/layer_skip/Llama-2-7b-hf/*

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/layer_skip/Llama-2-7b-hf/ 
```

- Run
```
python generate.py --compile --checkpoint_path checkpoints/layer_skip/Llama-2-7b-hf/model.pth --top_k 100 --top_p 0.9 --temperature 0.6 --self_speculative --early_exit 5 --speculate_k 3
```
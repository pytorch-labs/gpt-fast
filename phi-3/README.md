# Phi-3-mini-4k-instruct
[Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/) is a high-quality Language model. This repro is a simple and efficient PyTorch native implementation of Phi-3-mini-4k-instruct.

## Downloading Weights

```bash
export MODEL_REPO=microsoft/Phi-3-mini-4k-instruct
python scripts/download.py --repo_id $MODEL_REPO
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$MODEL_REPO
```

## Benchmarks
Benchmarks run on a single 3090. Note that all benchmarks are run at *batch size=1*, making the reported tokens/s numbers equivalent to "tokens/s/user". In addition, they are run with a very small prompt length (just 5 tokens).

| Model    | Technique | Tokens/Second | Memory Bandwidth (GB/s) |
| -------- | ------- | ------ | ------ |
| Phi-3-mini-4k-instruct  | Base    |  106.3  | 791 |
|           | 8-bit   | 160.5   | 598 |


## Generate Text

Model definition in `model.py`, generation code in `generate.py`.

```bash
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is"
```

To squeeze out a little bit more performance, you can also compile the prefill with `--compile_prefill`. This will increase compilation times though.

## Quantization
### Int8 Weight-Only Quantization
To generate this version of the model
```bash
# Spits out model at checkpoints/$MODEL_REPO/model_int8.pth
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int8
```
To run with int8, just pass the int8 checkpoint to generate.py.
```bash
python generate.py --compile --compile_prefill --checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth
```

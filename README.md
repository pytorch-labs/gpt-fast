# gpt-fast
Simple and efficient pytorch-native transformer text generation.

Featuring:
1. Very low latency
2. <1000 lines of python
3. No dependencies other than PyTorch and sentencepiece
4. int8/int4 quantization
5. Speculative decoding
6. Tensor parallelism
7. Supports Nvidia and AMD GPUs

This is *NOT* intended to be a "framework" or "library" - it is intended to show off what kind of performance you can get with native PyTorch :) Please copy-paste and fork as you desire.

For an in-depth walkthrough of what's in this codebase, see this [blog post](https://pytorch.org/blog/accelerating-generative-ai-2/).

## Supported Models

### LLaMA family
Please check the rest of this page about benchmark of LLaMA family models.

### Mixtral 8x7B
We also supported [Mixtral 8x7B](https://mistral.ai/news/mixtral-of-experts/) which is a high-quality sparse mixture of experts (MoE) model, the average token generation rates are:

|                  |   1 GPU |    2 GPU  | 4 GPU  |    8 GPU   |
|------------------|---------|-----------|--------|------------|
|baseline(bfloat16)|    OOM  |    96.67  | 155.35 |  227.82    |
|        int8      |   97.92 |   155.03  | 216.87 |  279.35    |

Note that the benchmarks run on an 8xA100-80GB, power limited to 330W with a hybrid cube mesh topology. Note that all benchmarks are run at *batch size=1*, making the reported tokens/s numbers equivalent to "tokens/s/user". In addition, they are run with a very small prompt length (just 5 tokens).

For more details about Mixtral 8x7B, please check [this page](./mixtral-moe) or this [note](https://thonking.substack.com/p/short-supporting-mixtral-in-gpt-fast).

## Examples
In the spirit of keeping the repo minimal, here are various examples of extensions you can make to gpt-fast as PRs.
- [Google Gemma](https://github.com/pytorch-labs/gpt-fast/pull/115)
- [xAI Grok-1](https://github.com/pytorch-labs/gpt-fast/pull/171)
- [Databricks DBRX](https://github.com/pytorch-labs/gpt-fast/pull/174)

## Community

Projects inspired by gpt-fast in the community:

- [gpt-blazing](https://github.com/armed-gpt/gpt-blazing): applies the same performance optimization strategy to more models (e.g., baichuan2).
- [gptfast](https://github.com/MDK8888/GPTFast): applies a subset of the performance optimizations to all Huggingface models
- [gpt-accelera](https://github.com/Edward-Sun/gpt-accelera): extends `gpt-fast` to SFT/RM/PPO training and batched inference to optimize the throughput

## Installation
[Download PyTorch nightly](https://pytorch.org/get-started/locally/)

Install required packages:

```bash
pip install -r requirements.txt
```

To download llama models, go to https://huggingface.co/meta-llama/Llama-2-7b and go through steps to obtain access.
Then login with `huggingface-cli login`



## Downloading Weights
Models tested/supported
```text
tinyllamas/stories{15,42,100}
openlm-research/open_llama_7b
meta-llama/Llama-2-7b-chat-hf
meta-llama/Llama-2-13b-chat-hf
meta-llama/Llama-2-70b-chat-hf
codellama/CodeLlama-7b-Python-hf
codellama/CodeLlama-34b-Python-hf
mistralai/Mistral-7B-v0.1
mistralai/Mistral-7B-Instruct-v0.1
mistralai/Mistral-7B-Instruct-v0.2
meta-llama/Meta-Llama-3-8B
```

For example, to convert Llama-2-7b-chat-hf
```bash
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
./scripts/prepare.sh $MODEL_REPO
```

## Benchmarks
Benchmarks run on an 8xA100-80GB, power limited to 330W with a hybrid cube mesh topology. Note that all benchmarks are run at *batch size=1*, making the reported tokens/s numbers equivalent to "tokens/s/user". In addition, they are run with a very small prompt length (just 5 tokens).

| Model    | Technique | Tokens/Second | Memory Bandwidth (GB/s) |
| -------- | ------- | ------ | ------ |
| Llama-2-7B  | Base    |  104.9  | 1397.31 |
|           | 8-bit   | 155.58   | 1069.20 |
|           | 4-bit (G=32)   | 196.80   | 862.69 |
| Llama-2-70B | Base    | OOM     ||
|           | 8-bit   | 19.13    | 1322.58 |
|           | 4-bit (G=32)   | 25.25    | 1097.66 |
| Llama-3-8B  | Base    |  94.25  | 1411.95 |
|           | 8-bit   | 139.55   | 1047.23 |

### Speculative Sampling
[Verifier: Llama-70B (int4), Draft: Llama-7B (int4)](./scripts/speculate_70B_int4.sh): 48.4 tok/s

### Tensor Parallelism
| Model    | Number of GPUs | Tokens/Second | Memory Bandwidth (GB/s) |
| -------- | ------- | ------ | ------ |
| Llama-2-7B  | 1    |  104.9  | 1397.31 |
|           | 2   | 168.84   | 1181.99 |
|           | 4   | 254.02   | 955.83 |
|           | 8   | 328.43   | 704.10 |
| Llama-2-70B  | 1    |  OOM  |  |
|           | 2   | 21.32   | 1481.87 |
|           | 4   | 38.01   | 1340.76 |
|           | 8   | 62.50   | 1135.29 |
| Llama-3-8B  | 1    |  94.19  | 1411.76 |
|           | 2   | 150.48   | 1208.80 |
|           | 4   | 219.77   | 991.63 |
|           | 8   | 274.65   | 768.55 |

### Tensor Parallelism + Quantization
| Model    | Technique | Tokens/Second | Memory Bandwidth (GB/s) |
| -------- | ------- | ------ | ------ |
| Llama-2-70B | Base    | 62.50     | 1135.29 |
|           | 8-bit   | 80.44    | 752.04 |
|           | 4-bit (G=32)   | 90.77    | 548.10 |

### AMD
Benchmarks run on one GCD of a MI-250x.

| Model    | Technique | Tokens/Second | Memory Bandwidth (GB/s) |
| -------- | ------- | ------ | ------ |
| Llama-2-7B  | Base    |  76.33  | 1028.70 |
|           | 8-bit   | 101.86   | 700.06 |

## Generate Text

Model definition in `model.py`, generation code in `generate.py`.

```bash
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is"
```

To squeeze out a little bit more performance, you can also compile the prefill with `--compile_prefill`. This will increase compilation times though.

## Quantization
Choose device to use by
```bash
# The current support devices: cuda, cpu
export DEVICE=cuda
```
### Int8 Weight-Only Quantization
To generate this version of the model
```bash
# Spits out model at checkpoints/$MODEL_REPO/model_int8.pth
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int8
```
To run with int8, just pass the int8 checkpoint to generate.py.
```bash
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth --device $DEVICE
```

### Int4 Weight-Only Quantization
To generate int4 version of model
```bash
# Spits out model at checkpoints/$MODEL_REPO/model_int4.g32.$DEVICE.pth
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4 --groupsize 32
```

To run with int4, just pass the int4 checkpoint to generate.py.
```bash
python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model_int4.g32.pth --compile
```

## Speculative Sampling
To generate with speculative sampling (DRAFT_MODEL_REPO should point to a smaller model compared with MODEL_REPO).

In this example, the "smaller" model is just the int8 quantized version of the model.
```
export DRAFT_MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model_int8.pth
```

Note: Running on an A100 80GB, albeit power-limited to 330 watts. Empirically, seems like peak bandwidth is about 1700 GB/s.


## Tensor Parallelism
```bash
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=2 generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth
```

## Experimental
### Evaluation
We use the EleutherAI evaluation harness to evaluate our model accuracy. To evaluate the accuracy, make sure the evaluation harness is installed and pass your model checkpoint and desired tasks to eval.py.

```bash
python eval.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --compile --tasks hellaswag winogrande
```

Note: Generative tasks are currently not supported for gpt-fast

Installation Instructions for the evaluation harness: https://github.com/EleutherAI/lm-evaluation-harness/tree/master#install

### GPTQ
We have a pure pytorch implementation of GPTQ that utilizes torch._dynamo.export to access the model structure. You can generate a GPTQ quantized
version of int4 quantization by using the same command to quantize it but adding 'gptq' to the quantization mode i.e.
```bash
# Spits out model at checkpoints/$MODEL_REPO/model_int4-gptq.g32.pth
python quantize.py --mode int4-gptq --calibration_tasks wikitext --calibration_seq_length 2048
```

You can then eval or generate text with this model in the same way as above.

## License

`gpt-fast` is released under the [BSD 3](https://github.com/pytorch-labs/gpt-fast/main/LICENSE) license.

## Acknowledgements
Thanks to:
* Lightning AI for supporting pytorch and work in flash attention, int8 quantization, and LoRA fine-tuning.
* GGML for driving forward fast, on device inference of LLMs
* Karpathy for spearheading simple, interpretable and fast LLM implementations
* MLC-LLM for pushing 4-bit quantization performance on heterogeneous hardware

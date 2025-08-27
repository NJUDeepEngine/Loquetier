# Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving

Low-Rank Adaptation (LoRA) has emerged as a widely adopted parameter-efficient fine-tuning (PEFT) technique to enhance the performance of large language models (LLMs) for downstream tasks. Loquetier is a virtualized multi-LoRA framework that enables unified LoRA fine-tuning and serving. Loquetier consists of two main components:

- a Virtualized Module that isolates model modifications, enabling flexible instance level migration upon one base model to support multiple PEFT methods within a shared base model architecture
- an optimized computation flow and the Segmented Multi-LoRA Multiplication (SMLM) kernel design that merges fine-tuning and inference paths in forward propagation, enabling efficient batching and minimizing kernel invocation overhead

Extensive experiments across three task settings demonstrate that Loquetier consistently outperforms existing baselines in performance and flexibility, achieving 3.0× the performance of the state-of-the-art co-serving system for the inference-only tasks, and 46.4× higher SLO attainment than PEFT in the unified tasks.

## Getting started

Please follow the guidelines in [kernel_src](kernel_src/README.md).

## Examples

You can explore your own strategies for unified fine-tuning and serving by making modifications based on these examples.

A simple example for llama without LoRA:

```bash
# Fill parameters in `run_llama.py` with your model path before running the example
cd examples
python run_llama.py
```

A simple example for llama with LoRA, using mixed model from Loquetier with pre-prepared inputs:

```bash
# Fill parameters in `run_mixed.py` and the configuration file you want to use with your model paths, dataset paths and modify other settings if you want before running the example
cd examples
python run_mixed.py -c [CONFIG_FILE] -d [DEVICE_INDEX] -o [OUTPUT_RESULT_FILE]
```

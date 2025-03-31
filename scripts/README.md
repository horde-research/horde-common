# Multiple Choice Model Evaluation Script

This repository contains a script to evaluate Hugging Face models on multiple-choice datasets such as MMLU-like and ENT-like datasets. The script processes formatted prompts, computes predictions using the provided model, and saves accuracy results for analysis.

## Usage

To execute the script, use the following command:
```bash
$ python mc-eval-simplified-inference.py --model_id Qwen/Qwen2.5-7B-Instruct --output_path .
```

### Parameters:
- `--model_id`: The Hugging Face model ID to use for predictions (e.g., `Qwen/Qwen2.5-7B-Instruct`).
- `--output_path`: The directory where the results will be saved.

## Requirements

This script is designed to work in the **kaz-llm-eval-lb** environment. However, if you are setting up a new environment, you can install the required libraries using the commands below:

```bash
!pip install -q -U bitsandbytes
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q -U langchain
!pip install -q einops
!pip install -q datasets
!pip install sentencepiece # optional, only for some models using older versions of LLaMA
```

## Alternative way to set up environment:

```bash
conda create -n py312 python=3.12
```

Activate env:

```bash
conda activate py312
```

Install required libraries:

```bash
pip install -r requirements.txt
```

## Script Features

- Evaluates models on two types of datasets: **MMLU-like** and **ENT-like**.
- Uses formatted templates to generate model prompts.
- Aligns logits with template formatting for accurate prediction scoring.
- Outputs two CSV files:
  - `final-<model_name>.csv`: Contains aggregated accuracy results per dataset.
  - `df-<model_name>.csv`: Contains detailed prediction results.

## Known Weakness

The alignment between template formatting (`template_*`) and the logits computation (`get_ans`) is critical and needs refinement. Current implementation works better than the original `kaz-llm-eval-lb` results, but further optimization is possible.

## Placeholders for Future Scripts

Pass

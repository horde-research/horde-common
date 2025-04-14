# Multiple Choice Model Evaluation Script

This repository contains a script to evaluate Hugging Face models on multiple-choice datasets such as MMLU-like and ENT-like datasets. The script processes formatted prompts, computes predictions using the provided model, and saves accuracy results for analysis.

## Usage

To execute the script, use the following command:
```bash
python mc-eval-simplified-inference.py --model_id meta-llama/Llama-3.3-70B-Instruct --dtype float16 --output_path .
```

### Parameters:
- `--model_id`: The Hugging Face model ID to use for predictions (e.g., `Qwen/Qwen2.5-7B-Instruct`).
- `--output_path`: The directory where the results will be saved.
- `--dtype`: Data type for model weights: float16, bfloat16, or float32 (default: float16).
- `--apply_chat_template`: Flag to use chat template formatting for instruct models.


## Set up environment:

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
  - `final-<model_name>.json`: Containts submittable json file for the leaderboard


## Placeholders for Future Scripts

Pass

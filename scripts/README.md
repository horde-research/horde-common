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

## Adding a new model

There is no per-model branching and none should be added. The script reads
`config.architectures` from the checkpoint and instantiates that exact class, falling
back to `AutoModelForCausalLM` / `AutoModelForImageTextToText`. Any model supported by
your installed `transformers` works by passing `--model_id`; new models generally need
only a `transformers` upgrade, not a code change.

This matters for multimodal checkpoints, where the auto classes disagree with each
other. `qwen3_5` maps to `Qwen3_5ForCausalLM` (the text tower alone) under
`AutoModelForCausalLM` but to `Qwen3_5ForConditionalGeneration` under
`AutoModelForImageTextToText` -- and the checkpoint declares the latter. Honouring the
declared class avoids silently mismatched weight prefixes.

Verified on a single A100 80GB, all in bfloat16 and all text-only:

| `--model_id` | Declared class | Weights |
| --- | --- | --- |
| `meta-models/Muse-Glimmer-30B` | `MuseGlimmerForConditionalGeneration` | 59.6 GB |
| `google/gemma-4-31B-it` | `Gemma4ForConditionalGeneration` | 62.6 GB |
| `Qwen/Qwen3.6-27B` | `Qwen3_5ForConditionalGeneration` | 55.6 GB |

```bash
pip install -U "transformers>=5.15.0"   # muse_glimmer support landed in 5.15
python mc-eval-simplified-inference.py \
  --model_id meta-models/Muse-Glimmer-30B \
  --dtype bfloat16 \
  --output_path .
```

Notes:

- **Use `bfloat16`, not `float16`.** All three checkpoints are bf16-native, and
  Muse-Glimmer additionally applies logit softcapping and attention scale factors, so
  float16 risks overflow to NaN. The script emits a `RuntimeWarning` if you force
  float16 on a bf16 checkpoint. bf16 costs the same memory and runs at the same speed
  on A100/H100.
- **Do not pass `--apply_chat_template` for Muse-Glimmer.** Its chat template ends the
  generation prompt at `<|start|>assistant` with no `<|message|>`, so the next-token
  distribution is over routing tokens (` to=user`), not answer letters — the `" A"`
  logit lookup would be meaningless. Raw completion mode (the default) is correct here
  and matches how the other leaderboard numbers were produced.
- Resources per model: 56–63 GB of weights to download, ~70 GB free disk for the HF
  cache, and roughly 2–4 hours for the full 17.8k-example run at batch size 1.
- Sanity check the first ~20 rows: mixed predicted letters mean the run is healthy.
  All-one-letter output, or accuracy at chance (~25% MMLU-like, ~12% ENT), means a
  dtype or weight-loading problem — kill it rather than waiting out the run.

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

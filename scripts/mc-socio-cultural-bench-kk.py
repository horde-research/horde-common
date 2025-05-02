"""
Quick-adapt evaluation script for a single 4-choice dataset 

The final JSON looks like
{
  "model_name": "meta-llama/Llama-3.3-70B-Instruct",
  "org_name":  "Your-Company",
  "overall_accuracy": 0.8123,
  "category_accuracy": {
        "cat1": 0.83,
        "cat2": 0.79,
        ...
  },
  "subcategory_accuracy": {
        "subcat1": 0.81,
        "subcat2": 0.85,
        ...
  }
}
"""
import argparse, json, os, sys
from typing import Dict, Any

import pandas as pd
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
import torch
from huggingface_hub.hf_api import HfFolder
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
from tqdm import tqdm

load_dotenv()
if "HUGGINGFACE_TOKEN" in os.environ:
    HfFolder.save_token(os.environ["HUGGINGFACE_TOKEN"])

_TEMPLATE = """
Сіз қазақ тілінде жауап беретін білімді, пайдалы көмекшісіз. Сұрақты және жауап нұсқаларын мұқият оқып,
ең дұрысын бір ғана әріппен (A, B, C, D) белгілеңіз.

Cұрақ: {prompt}
A) {a}
B) {b}
C) {c}
D) {d}

Жауап:""".strip()

_TEMPLATE_API = """
Сіз қазақ тілінде жауап беретін білімді, пайдалы көмекшісіз. Сұрақты және жауап нұсқаларын мұқият оқып,
ең дұрысын бір ғана әріппен (A, B, C, D) белгілеңіз.  
Жауапты **тек** төмендегі JSON құрылымында қайтарыңыз:

```json
{{"answer": "A"}}
```

Cұрақ: {prompt}
A) {a}
B) {b}
C) {c}
D) {d}

Жауап:
"""

prompt_4 = PromptTemplate(
    template=_TEMPLATE,
    input_variables=["prompt", "a", "b", "c", "d"],
)

def load_mc_dataset() -> Dataset:
    """Returns HF Dataset with .text, .answer, .category, .subcategory, .idx"""
    raw_ds = load_dataset("kz-transformers/kk-socio-cultural-bench-mc", split="train")
    def build_text(ex: Dict[str, Any]) -> Dict[str, str]:
        return {
            "text": prompt_4.format(
                prompt=ex["question"],
                a=ex["A"],
                b=ex["B"],
                c=ex["C"],
                d=ex["D"],
            )
        }
    ds = raw_ds.map(build_text)
    ds = ds.select(range(10))  # for testing
    print(ds["text"][0])
    ds = ds.rename_column("formatted_answer", "answer")
    ds = ds.add_column("idx", [f"{i}" for i in range(len(ds))])
    return ds


def load_model_and_tokenizer(model_id: str, dtype: str = "float16"):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype_map[dtype],
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def predict_option(model, tokenizer, prompt: str, apply_chat_template: bool = False) -> str:
    if apply_chat_template:
        print("Using chat template")
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]

    # Four-choice logits (space-prefixed tokens)
    scores = {
        "A": logits[tokenizer(" A").input_ids[-1]],
        "B": logits[tokenizer(" B").input_ids[-1]],
        "C": logits[tokenizer(" C").input_ids[-1]],
        "D": logits[tokenizer(" D").input_ids[-1]],
    }
    return max(scores, key=scores.get)


def run_eval(ds: Dataset, model, tokenizer, apply_chat_template: bool = False) -> pd.DataFrame:
    rows = []
    for ex in tqdm(ds, total=len(ds)):
        pred = predict_option(model, tokenizer, ex["text"], apply_chat_template)
        rows.append(
            {
                "idx": ex["idx"],
                "category": ex["category"],
                "subcategory": ex["subcategory"],
                "answer": ex["answer"].strip(),  # ensure no whitespace
                "predict": pred,
            }
        )
    df = pd.DataFrame(rows)
    df["correct"] = (df["answer"] == df["predict"]).astype(int)
    return df

def save_final_json(df: pd.DataFrame, model_name: str, org_name: str, out_dir: str):
    overall_acc = df["correct"].mean()

    cat_acc = (
        df.groupby("category")["correct"]
        .mean()
        .round(6)
        .to_dict()
    )

    subcat_acc = (
        df.groupby(["category", "subcategory"])["correct"]
        .mean()
        .reset_index()                                     # drop the MultiIndex
        .assign(key=lambda x: x["category"] + "|" + x["subcategory"])
        .set_index("key")["correct"]
        .round(6)
        .to_dict()
    )

    final_json = {
        "model_name": model_name,           # do NOT sanitise
        "org_name": org_name,
        "overall_accuracy": round(float(overall_acc), 6),
        "category_accuracy": cat_acc,
        "subcategory_accuracy": subcat_acc,
    }

    os.makedirs(out_dir, exist_ok=True)
    file_name = model_name.replace("/", "__") + ".json"
    with open(os.path.join(out_dir, file_name), "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    print(f"\nSaved ⇒ {os.path.join(out_dir, file_name)}")



def main():
    p = argparse.ArgumentParser(description="Evaluate 4-choice MC dataset and emit one JSON report.")
    p.add_argument("--model_id", required=True, help="HF model to evaluate.")
    p.add_argument("--org_name", required=True, help="Your organisation/company name.")
    p.add_argument("--output_path", required=True, help="Folder to write the JSON file.")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--apply_chat_template", action="store_true", help="Use chat template for chat-tuned models.")
    args = p.parse_args()

    ds = load_mc_dataset()
    model, tok = load_model_and_tokenizer(args.model_id, dtype=args.dtype)

    df = run_eval(ds, model, tok, apply_chat_template=args.apply_chat_template)
    save_final_json(df, args.model_id, args.org_name, args.output_path)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
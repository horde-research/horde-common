"""
This script processes multiple-choice datasets (MMLU-like and ENT-like) to evaluate Hugging Face models' performance in predicting correct answers based on formatted prompts. 

### Known Weakness:
The interconnection between the choice formatting in `template_*` and the options list in the `get_ans` function is a critical point that requires attention:
1. The logits computation relies on tokens generated by `tokenizer(' A')`, which are different from `tokenizer('A')` since `' A'` (space included) is treated as a separate token.
2. Based on the current templates, the choice formatting must align with this distinction (`' A' != 'A'`). This discrepancy can impact prediction accuracy.

Although we have not completed all experiments to optimize this behavior, the current approach—combining `' A'`-based logits with the templates—yields better results than the original implementation used in `kaz-llm-eval-lb`.

### Recommendation:
Future iterations should consider revisiting the choice alignment between templates and logits calculations. Refining this alignment could further improve performance and ensure consistency across experiments.
"""

import argparse
import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate

def load_and_prepare_datasets() -> Dict[str, Any]:
    """
    Load and prepare datasets for processing.

    Returns:
        Dict[str, Any]: Dictionary containing formatted datasets.
    """
    mmlu = load_dataset("kz-transformers/mmlu-translated-kk")["validation"]
    const = load_dataset("kz-transformers/kazakh-constitution-mc")["test"]
    dastur = load_dataset("kz-transformers/kazakh-dastur-mc")["test"]
    ent = load_dataset("kz-transformers/kazakh-unified-national-testing-mc")

    template_mmlu = ("""
    Сіз қазақ тілінде жауап беретін білімді, пайдалы көмекшісіз. Қазақстан мәдениеті, ғылым, тарих және басқа да
    салалар бойынша көптаңдаулы сұрақтарға жауап беріңіз. Сұрақты және берілген жауап нұсқаларын мұқият оқып, ең дұрысын
    бір ғана әріппен (A, B, C, т.б.) белгілеңіз.

    Cұрақ: {prompt}\n
    A) {a}\n
    B) {b}\n
    C) {c}\n
    D) {d}\n

    Жауап:""")

    template_ent = ("""
    Сіз қазақ тілінде жауап беретін білімді, пайдалы көмекшісіз. Қазақстан мәдениеті, ғылым, тарих және басқа да
    салалар бойынша көптаңдаулы сұрақтарға жауап беріңіз. Сұрақты және берілген жауап нұсқаларын мұқият оқып, ең дұрысын
    бір ғана әріппен (A, B, C, т.б.) белгілеңіз.

    Cұрақ: {prompt}\n
    A) {a}\n
    B) {b}\n
    C) {c}\n
    D) {d}\n
    E) {e}\n
    F) {f}\n
    G) {g}\n
    H) {h}\n

    Жауап:""")

    prompt_mmlu = PromptTemplate(template=template_mmlu, input_variables=['prompt', 'a', 'b', 'c', 'd'])
    prompt_ent = PromptTemplate(template=template_ent, input_variables=['prompt', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

    def format_text_mmlu(example: Dict[str, Any]) -> Dict[str, str]:
        return {
            "text": prompt_mmlu.format(
                prompt=example['Question'],
                a=example['Option A'],
                b=example['Option B'],
                c=example['Option C'],
                d=example['Option D']
            )
        }

    def format_text_ent(example: Dict[str, Any]) -> Dict[str, str]:
        required_keys = ['question', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        missing_keys = [key for key in required_keys if key not in example]
        if missing_keys:
            raise ValueError(f"Missing keys in example: {missing_keys}")
        return {
            "text": prompt_ent.format(
                prompt=example.get('question', ''),
                a=example.get('A', ''),
                b=example.get('B', ''),
                c=example.get('C', ''),
                d=example.get('D', ''),
                e=example.get('E', ''),
                f=example.get('F', ''),
                g=example.get('G', ''),
                h=example.get('H', '')
            )
        }

    mmlu = mmlu.map(format_text_mmlu).to_pandas()
    const = const.map(format_text_mmlu).to_pandas()
    dastur = dastur.map(format_text_mmlu).to_pandas()
    ent = pd.concat([ent[i].map(format_text_ent).to_pandas() for i in ent])

    ent['idx'] = ent.index.astype(str) + "-" + ent['subject']
    mmlu['idx'] = mmlu.index.astype(str) + "-mmlu"
    const['idx'] = const.index.astype(str) + "-const"
    dastur['idx'] = dastur.index.astype(str) + "-dastur"

    ent = ent.rename({'correct_answer': 'answer'}, axis=1)
    mmlu = mmlu.rename({'Correct Answer': 'answer'}, axis=1)
    const = const.rename({'Correct Answer': 'answer'}, axis=1)
    dastur = dastur.rename({'Correct Answer': 'answer'}, axis=1)

    columns = ["answer", "text", "idx"]
    mmlu_like = pd.concat([mmlu[columns], const[columns], dastur[columns]])

    return {
        "ent_ds": Dataset.from_pandas(ent[columns]),
        "mmlu_like_ds": Dataset.from_pandas(mmlu_like)
    }

def load_model_and_tokenizer(model_id: str) -> Dict[str, Any]:
    """
    Load the Hugging Face model and tokenizer.

    Args:
        model_id (str): Hugging Face model ID.

    Returns:
        Dict[str, Any]: Dictionary containing the model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to('cuda')
    model.eval()

    return {"model": model, "tokenizer": tokenizer}

def get_ans(model: Any, tokenizer: Any, text: str, mode: str = "mmlu") -> tuple:
    """
    Generate an answer for the given text.

    Args:
        model (Any): Hugging Face model.
        tokenizer (Any): Tokenizer for the model.
        text (str): Input text.
        mode (str): Dataset mode, either "mmlu" or "ent".

    Returns:
        tuple: Predicted answer.
    """
    inputs = tokenizer(text, return_tensors='pt')
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]

    options_list = [
        (logits[tokenizer(' A').input_ids[-1]], 'A'),
        (logits[tokenizer(' B').input_ids[-1]], 'B'),
        (logits[tokenizer(' C').input_ids[-1]], 'C'),
        (logits[tokenizer(' D').input_ids[-1]], 'D')
    ]

    if mode == "ent":
        options_list += [
            (logits[tokenizer(' E').input_ids[-1]], 'E'),
            (logits[tokenizer(' F').input_ids[-1]], 'F'),
            (logits[tokenizer(' G').input_ids[-1]], 'G'),
            (logits[tokenizer(' H').input_ids[-1]], 'H')
        ]

    return max(options_list, key=lambda x: x[0])

def process_dataset(dataset: Dataset, model: Any, tokenizer: Any, mode: str) -> pd.DataFrame:
    """
    Process a dataset to compute predictions and accuracy.

    Args:
        dataset (Dataset): Input dataset.
        model (Any): Hugging Face model.
        tokenizer (Any): Tokenizer for the model.
        mode (str): Dataset mode, either "mmlu" or "ent".

    Returns:
        pd.DataFrame: DataFrame with predictions and accuracy.
    """
    results = []
    for data in tqdm(dataset, total=len(dataset)):
        ans_list = get_ans(model, tokenizer, data['text'], mode=mode)
        predict = ans_list[1]
        answer = data['answer']
        acc = int(predict == answer)
        idx = data['idx']
        results.append({'idx': idx, 'answer': answer, 'predict': predict, 'acc': acc})
    return pd.DataFrame(results)

def save_results(answers_mmlu: pd.DataFrame, answers_ent: pd.DataFrame, model_id: str, output_path: str) -> None:
    """
    Save results to CSV files.

    Args:
        answers_mmlu (pd.DataFrame): Results for MMLU-like datasets.
        answers_ent (pd.DataFrame): Results for ENT-like datasets.
        model_id (str): Hugging Face model ID.
        output_path (str): Directory to save output files.
    """
    df = pd.concat((answers_mmlu, answers_ent))
    df['dataset'] = df.idx.apply(lambda x: x.split('-')[-1])

    final = df.groupby('dataset').acc.agg(['sum', 'count'])
    final['acc'] = final['sum'] / final['count']

    name = "_".join(model_id.split("/"))
    final.to_csv(os.path.join(output_path, f"final-{name}.csv"))
    df.to_csv(os.path.join(output_path, f"df-{name}.csv"))

def main(model_id: str, output_path: str) -> None:
    """
    Main function to load data, model, and process predictions.

    Args:
        model_id (str): Hugging Face model ID.
        output_path (str): Directory to save output CSV files.
    """
    datasets = load_and_prepare_datasets()
    model_data = load_model_and_tokenizer(model_id)

    answers_mmlu = process_dataset(datasets["mmlu_like_ds"], model_data["model"], model_data["tokenizer"], mode="mmlu")
    answers_ent = process_dataset(datasets["ent_ds"], model_data["model"], model_data["tokenizer"], mode="ent")

    save_results(answers_mmlu, answers_ent, model_id, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute predictions for MMLU-like and ENT-like datasets.")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID to use for predictions.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save output CSV files.")
    args = parser.parse_args()

    main(model_id=args.model_id, output_path=args.output_path)

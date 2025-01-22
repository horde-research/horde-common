import json
import requests
import logging
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset, DatasetDict, load_from_disk
from datetime import datetime


logging.basicConfig(
    filename='translation_process.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)
lingvanex = "put your key here"
url = "https://api-b2b.backenster.com/b1/api/v3/translate"


headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": lingvanex
}


def lingvanex_request(text: str) -> str:
    payload = {
        "from": "en_GB",
        "to": "kk_KZ",
        "data": text,
        "platform": "api"
    }
    response = requests.post(url, json=payload, headers=headers)
    try:
        return json.loads(response.text)['result']
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Failed to parse response for text: '{text}' | Error: {e}")
        return "Translation Error"


def translate_dataset(ds):
    translated_data = {
        'test': [],
        'validation': [],
        'dev': []
    }
    for split in ['test', 'validation', 'dev']:
        logging.info(f"Starting translation for the {split} set...")
        for idx, obj in enumerate(tqdm(ds[split], desc=f"Processing {split} set")):
            subject = obj['subject']
            try:
                translated_question = lingvanex_request(obj['question'])
                translated_choices = [lingvanex_request(choice) for choice in obj['choices']]
                
                translated_data[split].append({
                    'question': translated_question,
                    'subject': subject,
                    'choices': translated_choices,
                    'answer': obj['answer'],
                    'id': idx
                })
            except Exception as e:
                logging.error(f"Error in subject '{subject}' (question ID: {idx}): {e}")

    translated_dataset_dict = DatasetDict({
        split: ds[split].from_list(translated_data[split])
        for split in ['test', 'validation', 'dev']
    })

    return translated_dataset_dict


def save_to_excel(translated_data, ds):
    rows = []
    for split, data in translated_data.items():
        for entry in data:
            row = {
                "Split": split,
                "Subject": entry["subject"],
                "Question ID": entry["id"],
                "Original Question": ds[split][entry['id']]["question"],
                "Translated Question": entry["question"]
            }
            for i, (orig_choice, trans_choice) in enumerate(zip(ds[split][entry['id']]["choices"], entry["choices"])):
                row[f"Original Choice {i+1}"] = orig_choice
                row[f"Translated Choice {i+1}"] = trans_choice

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel("mmlu_translation_comparison.xlsx", index=False)
    logging.info("Translation and comparison process complete. The Excel file 'mmlu_translation_comparison.xlsx' has been created.")


def main():
    logging.info("Loading dataset...")
    ds = load_dataset("cais/mmlu", "all")
    logging.info("Dataset loaded successfully.")

    translated_dataset_dict = translate_dataset(ds)

    translated_dataset_dict.save_to_disk("translated_mmlu_dataset")
    logging.info("Translated dataset has been saved to disk as 'translated_mmlu_dataset'.")

    save_to_excel(translated_dataset_dict, ds)


if __name__ == "__main__":
    main()

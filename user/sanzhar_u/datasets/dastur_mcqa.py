import logging
import pickle
from tqdm import tqdm
import pandas as pd
from openai import OpenAI


client = OpenAI(api_key=openai_key)

kazakh_tradition_prompt = """
You are an expert in Kazakh culture and language. Below is the name and definition of a Kazakh tradition. Your task is to generate multiple-choice questions in Kazakh based on this input. The number of questions should be determined by the length and richness of the input text, ensuring all questions are directly related to the given tradition.

**Tradition Name:** {tradition_name}  
**Tradition Definition:** {tradition_definition}

**Instructions for generating the questions:**

1. Generate multiple-choice questions that:
   - Test understanding of the definition.
   - Cover important details about the tradition mentioned in the input text (e.g., its purpose, when it is practiced, who participates, or its significance).
   - Include questions of different types, such as:
     - "What is {tradition_name}?" (e.g., "{tradition_name} деген не?")
     - Questions exploring the main purpose of the tradition.
     - Questions about how, when, or by whom the tradition is practiced.
     - Any other relevant details from the text that can form the basis of a question.

2. Each question must include:
   - The question text.
   - Four answer options (labeled as A, B, C, D), one of which is correct.
   - Clearly indicate the correct answer in the JSON output.

3. Adapt the number of questions based on the input size:
   - Short inputs: 3-5 questions.
   - Longer inputs: Up to 10 questions.

4. Use the following structured JSON format to present your output:

**JSON Output Format:**
{{
  "questions": [
    {{
      "question": "Question text in Kazakh",
      "options": {{
        "A": "Option A text",
        "B": "Option B text",
        "C": "Option C text",
        "D": "Option D text"
      }},
      "correct_answer": "Correct option label (e.g., 'B')"
    }},
    {{
      "question": "Next question text in Kazakh",
      "options": {{
        "A": "Option A text",
        "B": "Option B text",
        "C": "Option C text",
        "D": "Option D text"
      }},
      "correct_answer": "Correct option label (e.g., 'C')"
    }}
    // Add more questions as needed
  ]
}}

**Example Input:**
{{
  "tradition_name": "Бесікке салу",
  "tradition_definition": "Бесікке салу – қазақ халқының нәрестені бесікке салып, оның өмірінің жаңа кезеңіне қадам басу салтанатты рәсімі."
}}

**Example Output:**
{{
  "questions": [
    {{
      "question": "Бесікке салу деген не?",
      "options": {{
        "A": "Нәрестені жаялыққа орау",
        "B": "Нәрестені бесікке жатқызу",
        "C": "Нәрестені құндақтау",
        "D": "Нәрестенің атын қою"
      }},
      "correct_answer": "B"
    }},
    {{
      "question": "Бесікке салу рәсімінің негізгі мақсаты қандай?",
      "options": {{
        "A": "Нәрестені қорғау",
        "B": "Нәрестеге жаңа өмір кезеңін бастауға жол ашу",
        "C": "Отбасы мүшелерін жинау",
        "D": "Той жасау"
      }},
      "correct_answer": "B"
    }},
    {{
      "question": "Бесікке салу рәсімінде қандай зат қолданылады?",
      "options": {{
        "A": "Жаялық",
        "B": "Бесік",
        "C": "Ою-өрнек",
        "D": "Құндақ"
      }},
      "correct_answer": "B"
    }}
  ]
}}
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("generation_log.log"),
        logging.StreamHandler()
    ]
)


def generation_questions(text: str, model: str = "gpt-4") -> str:
    messages = [
        {"role": "system", "content": text}
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2000,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating questions: {e}")
        return None


def main(input_file: str = 'ranker/dastur.csv', output_file: str = 'dastur_failed_indices.pkl'):
    df = pd.read_csv(input_file)
    x = pd.read_pickle('dastur.pkl')
    df = df[df.index.isin(x['failed_indices'])]
    logging.info(f"Loaded input file: {input_file}, total rows: {df.shape[0]}")
    
    responses = []
    failed_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            prompt_filled = kazakh_tradition_prompt.format(
                tradition_name=row['Title'],
                tradition_definition=row['Text']
            )
            
            response = generation_questions(prompt_filled)
            
            if response is not None:
                responses.append((idx, response))
                logging.info(f"Successfully generated questions for index {idx}")
            else:
                failed_indices.append(idx)
                responses.append((idx, None))
                logging.warning(f"Failed to generate questions for index {idx}")
            
            if idx % 10 == 0:
                with open(output_file, "wb") as f:
                    pickle.dump({"responses": responses, "failed_indices": failed_indices}, f)
                logging.info(f"Progress saved at index {idx}")
            
        except Exception as e:
            logging.error(f"Error processing row {idx}: {e}")
            failed_indices.append(idx)
    
    with open(output_file, "wb") as f:
        pickle.dump({"responses": responses, "failed_indices": failed_indices}, f)
    logging.info(f"Final results saved to {output_file}")


if __name__ == "__main__":
    
    logging.info("Script started.")
    try:
        main()
    except Exception as e:
        logging.critical(f"Script failed with error: {e}")
    logging.info("Script finished.")

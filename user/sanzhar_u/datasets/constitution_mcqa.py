import logging
import pickle
from tqdm import tqdm
import pandas as pd
from openai import OpenAI


client = OpenAI(api_key=openai_key)

kazakh_constitution_prompt = """
You are an expert in Kazakh law and the Constitution of Kazakhstan. Below is the name of a constitutional article and its description. Your task is to generate multiple-choice questions in Kazakh based on this input. Ensure that all questions are directly related to the content of the given article.

**Article Name:** {article_name}  
**Article Description:** {article_description}

**Instructions for generating the questions:**

1. Generate multiple-choice questions that:
   - Test understanding of the article and its key points.
   - Cover important aspects such as its purpose, implications, or responsibilities outlined in the article.
   - Include questions of different types, such as:
     - "What is the main idea of {article_name}?" (e.g., "{article_name} негізгі мақсаты қандай?")
     - Questions exploring the specific roles, responsibilities, or rights described in the article.
     - Questions about the significance of the article in the context of Kazakh law.
     - Any other relevant details that can form the basis of a question.

2. Each question must include:
   - The question text.
   - Four answer options (labeled as A, B, C, D), one of which is correct.
   - Clearly indicate the correct answer in the JSON output.

3. Adapt the number of questions based on the input size:
   - Short inputs: 3-5 questions.
   - Longer inputs: Up to 10 questions.

4. Use the following structured JSON format to present your output:

JSON Output Format:
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
            max_tokens=3000,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating questions: {e}")
        return None


def main(input_file: str = 'ranker/adilet_constitution.csv', output_file: str = 'constitution_outputs.pkl'):
    df = pd.read_csv(input_file)
    logging.info(f"Loaded input file: {input_file}, total rows: {df.shape[0]}")
    
    responses = []
    failed_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            prompt_filled = kazakh_constitution_prompt.format(
                article_name=row['section'],
                article_description=row['text']
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

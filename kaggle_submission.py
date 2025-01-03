import re  # Added import for regex
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from rag import RAG, process_cheatsheet

load_dotenv()
client = OpenAI()
rag = RAG(client, cheatsheet_path="cheatsheet.md")  # RAG instance

# Read the CSV files
test_data = pd.read_csv("generative-ai-for-reliability-engineering/test.csv")


# System prompt for consistent responses
SYSTEM_PROMPT = """You will be given a multiple choice question about reliability engineering. Choose the correct answer. At the end of your response, start a new line and use the following format to output your answer: [Answer] [The letters you choose]. For example, if you think the answer [a] is correct, output [Answer] [a]. If you think there are multiple correct answer, using a comma to separate them, e.g., [Answer] [a], [b]. You will be given some context followed by [Context]. Use it to reason step-by-step. if you need to calculate anything, generate a python script to do the calculation. Limit your output to 400 words maximum."""


def get_ai_response(question):
    try:
        # Retrieve relevant chunk from RAG
        context = rag.get_relevant_chunks(question, top_k=1)[0]
        rag_prompt = f"{SYSTEM_PROMPT}\n[Context]: {context}\n"

        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # Use appropriate model
            messages=[
                {"role": "system", "content": rag_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        response = completion.choices[0].message.content
        # Extract only the letter(s) using regex
        generated_answer = response.split("[Answer]")[-1].strip().split("\n")[0]

        letters = re.findall(r"\[([a-zA-Z])\]", generated_answer)

        extracted_answer = ",".join(letters)

        print(extracted_answer)

        return extracted_answer
    except Exception as e:
        print(f"Error processing question: {e}")
        return None


# Process questions and collect answers
predictions = []
for idx, question in enumerate(test_data["question"]):
    print(f"Processing question {idx + 1}/{len(test_data)}")

    question_predictions = []
    for _ in range(5):
        # Get AI response
        answer = get_ai_response(question)
        question_predictions.append(answer)

        # Add delay to respect API rate limits
        time.sleep(1)

    predictions.append(question_predictions)

# Create submission dataframe
submission = pd.DataFrame(
    {
        "question_id": [
            i + 1 for i in range(len(predictions))
        ],  # Changed to start from 1
        "prediction_1": [pred[0] for pred in predictions],
        "prediction_2": [pred[1] for pred in predictions],
        "prediction_3": [pred[2] for pred in predictions],
        "prediction_4": [pred[3] for pred in predictions],
        "prediction_5": [pred[4] for pred in predictions],
    }
)

# Save results
submission.to_csv("submission_results.csv", index=False)
print("Processing complete. Results saved to submission_results.csv")

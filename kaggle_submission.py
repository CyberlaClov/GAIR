import re  # Added import for regex
import time

import pandas as pd
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Read the CSV files
test_data = pd.read_csv("generative-ai-for-reliability-engineering/test.csv")


# System prompt for consistent responses
SYSTEM_PROMPT = """You will be given a multiple choice question about reliability engineering. Choose the correct answer. At the end of your response, start a new line and use the following format to output your answer: [Answer] [The letters you choose]. For example, if you think the answer [a] is correct, output [Answer] [a]. If you think there are multiple correct answer, using a comma to separate them, e.g., [Answer] [a], [b]. Reason step by step, if you need to calculate anything, generate a python script to do the calculation. Limit your output to 400 words maximum."""


def get_ai_response(question):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # Use appropriate model
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        response = completion.choices[0].message.content.strip()
        # Extract only the letter(s) using regex
        letters = re.findall(r"\[([a-zA-Z])\]", response)

        extracted_answer = ",".join(letters)

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

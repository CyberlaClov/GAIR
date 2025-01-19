import re  # Added import for regex
import time

import pandas as pd
from openai import OpenAI
import json
import sys
from rag_jerem import RAG, process_cheatsheet
import subprocess

from agent_jerem import llm_runner, execute_python_script, kaggle_agent

client = OpenAI()
rag = RAG(client, cheatsheet_path="full_reliability_documentation.md")  # RAG instance

# Read the CSV files
test_data = pd.read_csv("generative-ai-for-reliability-engineering/test.csv")


# System prompt for consistent responses
SYSTEM_PROMPT = """You will be given a multiple choice question about reliability engineering.
Choose the correct answer.
You will be given some context followed by "Context".
Use it to reason step-by-step.

When performing calculations:
1. Think step-by-step and derive an equation.
2. Generate a python script using primarily the 'reliability' library.
If needed, import specific distributions using:
from reliability.Distributions import [distribution].
3. Include error handling in your script using try-except blocks.
4. Print results with print(QoI).
5. Use execute_python_script to run the code.

If the calculation fails:
- DO NOT return None or invalid results
- Fall back to theoretical reasoning using the reliability documentation context
- Use approximations if necessary
- Always provide a reasoned answer choice

For reliability library usage:
- Reference the provided documentation
- Prefer reliability library over scipy when documentation shows clear advantages
- Double-check distribution parameter requirements

At the end, use format:
[Answer] [letter choice]
For multiple answers: [Answer] [a], [b]

Maximum response: 400 words."""



def get_ai_response(question):
    try:
        # Get context from RAG
        context = rag.get_relevant_chunks(question, top_k=3)[0]
        
        # Format base prompt with context and system instructions
        rag_prompt = f"{SYSTEM_PROMPT}\n[Context]: {context}\n"

        # First try with direct GPT-4 completion
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": rag_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        response = completion.choices[0].message.content

        # If no clear answer, use kaggle_agent for code execution
        if "[Answer]" not in response:
            full_prompt = f"""
            Question: {question}
            Context: {context}
            Choose the best answer and format as [Answer] [letter].
            """
            response = kaggle_agent(
                prompt=full_prompt,
                client=client,
                mdl_name="gpt-4o-mini",
                temperature=0
            )

        # Extract letters only between [Answer] tag
        answer_match = re.search(r'\[Answer\]\s*\[(.*?)\]', response)
        if not answer_match:
            raise ValueError("No valid answer format found")
        
        # Split and clean the letters
        letters = [letter.strip().lower() for letter in answer_match.group(1).split(',')]
        unique_letters = list(dict.fromkeys(letters))  # Remove duplicates while preserving order
        
        # Strictly enforce maximum 2 answers
        if len(unique_letters) > 2:
            unique_letters = unique_letters[:2]
        elif len(unique_letters) == 0:
            raise ValueError("No valid letters found in answer")
            
        extracted_answer = ",".join(unique_letters)
        print(extracted_answer)
        return extracted_answer
    except Exception as e:
        print(f"Error processing question: {e}")
        return get_ai_response(question)  # Retry on failure


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
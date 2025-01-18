import json
import os
import re  # Added import for regex
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm  # Add this import

from agents import execute_python_script, tools
from rag import RAG


def get_ai_response(question, sys_prompt, temperature=1, tools=None):
    try:
        # Retrieve relevant chunks from RAG
        contexts = rag.get_relevant_chunks(question)

        # Combine all contexts into one, with separators
        combined_context = "\n---\n".join(contexts)

        # print(combined_context)

        rag_prompt = f"{sys_prompt}\n[Context]: {combined_context}\n"

        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # Use appropriate model
            messages=[
                {"role": "system", "content": rag_prompt},
                {"role": "user", "content": question},
            ],
            temperature=temperature,
            tools=tools,
        )
        script_executed = False

        if completion.choices[0].finish_reason == "tool_calls":
            tool_call = completion.choices[0].message.tool_calls[0]  # tool
            arguments = json.loads(tool_call.function.arguments)

            try:
                script = arguments.get("script")
                # print(script)
                result = execute_python_script(script, venv_path=".virtualvenv")
                if result is not None:
                    script_executed = True
                    result = result.strip("'").rstrip("\n")

                else:
                    result = "Please ignore the result of the script and generate your response independently with a theoritical reasonning. Don't forget : at the end of your response, start a new line and use the following format to output your answer: [Answer] [The letters you choose]"
            except Exception as e:
                result = "Script execution failed. Ignore the script result and generate your response independently, following the same output format as instructed."

            # print(result)

            function_call_result_message = {
                "role": "tool",
                "content": f"script_result: {result}",
                "tool_call_id": completion.choices[0].message.tool_calls[0].id,
            }

            completion_payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": rag_prompt},
                    {"role": "user", "content": question},
                    completion.choices[0].message,
                    function_call_result_message,
                ],
            }

            # OpenAI chat completions API call

            try:
                completion_final = client.chat.completions.create(
                    model=completion_payload["model"],
                    messages=completion_payload["messages"],
                    temperature=temperature,
                )
                if completion_final.choices[0].message.content is None:
                    completion_final = completion
            except:
                completion_final = completion
                script_executed = False
        else:
            completion_final = completion

        response_txt = completion_final.choices[0].message.content

        # Extract only the letter(s) using regex
        generated_answer = response_txt.split("[Answer]")[-1].strip().split("\n")[0]

        letters = re.findall(r"\[([a-zA-Z])\]", generated_answer)

        extracted_answer = ",".join(letters)

        return extracted_answer
    except Exception as e:
        print(f"Error processing question: {e}")
        return None


if __name__ == "__main__":
    # Process questions and collect answers

    # Read the CSV files
    test_data = pd.read_csv("generative-ai-for-reliability-engineering/test.csv")

    # System prompt for consistent responses
    SYSTEM_PROMPT = """You will be given a multiple choice question about reliability engineering. Choose the correct answer. You will be given some context followed by [Context]. Use it to reason step-by-step. If you need any calculation, follow the steps below:

    ○ First think step-by-step and derive an equation for the calculation while explaining your reasonning.
    ○ Generate a python script for the calculation. Use primarily 'reliability' or 'scipy' python libraries. When using 'reliability', be careful on the importation of desired classes and functions. When you need a particular Distributions import it using from reliability.Distributions import [distrbution you need]. Always print what is relevant to answer the question at the end of the script. For example, if the result you want is called QoI, add this line at the end of your script: "print(QoI)". Be careful when data is provided in the question, you should use it in your script,
    ○ Next, Use the tool "execute_python_script" to execute the python script. This tool runs the code in a sandbox and catch the std.out. You will receive the std.out from the script your generated that is why you need to use print in your script.
    ○ Choose the correct answer from the given choices based on the results of your script. If no exact match exists, choose the choice that is the closest to the calculated result. If the calculation fails:
    - DO NOT return None or invalid results
    - Fall back to theoretical reasoning using the reliability documentation context
    - Use approximations if necessary
    - Always provide a reasoned answer choice
    - Never generate a script without a print statement at the end providing the needed calculations.

    At the end of your response, start a new line and use the following format to output your answer: [Answer] [The letters you choose]. For example, if you think the answer [a] is correct, output [Answer] [a]. If you think there are multiple correct answer, using a comma to separate them, e.g., [Answer] [a], [b]. Limit your output to 400 words maximum."""

    mdl_name = "gpt-4o-mini"  # Model name
    prompt_name = "full_RAG_corrected_agent"  # Prompt name
    temperature = 0  # Set temperature for AI response
    cheatsheet_path = "full_reliability_documentation.md"

    load_dotenv()
    client = OpenAI()
    rag = RAG(client, cheatsheet_path, chunks_to_load=10)  # RAG instance
    predictions = []

    for idx, question in tqdm(
        enumerate(test_data["question"]),
        total=len(test_data),
        desc="Processing questions",
    ):
        question_predictions = []
        for attempt in tqdm(
            range(5), desc=f"Generating predictions for Q{idx+1}", leave=False
        ):
            # Get AI response
            answer = get_ai_response(
                question, sys_prompt=SYSTEM_PROMPT, temperature=temperature, tools=tools
            )
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

    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime("%Y%m%d%H%M%S")

    log_folder = f"outputs/{mdl_name}_prompt_{prompt_name}_temperature_{temperature}_{datetime_string}/"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    # Save results
    submission.to_csv(
        f"{log_folder}/predition_{mdl_name}_{prompt_name}.csv",
        index=False,
    )

    print(
        f"Prediction results saved to {log_folder}/predition_{mdl_name}_{prompt_name}_temperature_{temperature}.csv"
    )
    # question = test_data["question"][3]
    # print(question)
    # answer = get_ai_response(
    #     question, sys_prompt=SYSTEM_PROMPT, temperature=temperature, tools=tools
    # )

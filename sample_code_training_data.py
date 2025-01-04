import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm


def extract_failed_answers(base_dict, prompt_chain=False):
    result_path = f"{base_dict}/df_results.csv"
    # Read JSON objects line by line and create a DataFrame
    df = pd.read_csv(result_path)

    # Convert the list of dictionaries to a DataFrame
    df = df[["mean"]]
    df.rename(columns={"mean": "mean correctness rate"}, inplace=True)

    # Create a list of all .json files except "df_results.json"
    json_files = [
        f
        for f in os.listdir(base_dict)
        if f.endswith(".json") and f != "df_results.json"
    ]

    for idx, result_log_name in enumerate(json_files):
        with open(f"{base_dict}/{result_log_name}", "r") as file:
            result_log = json.load(file)
        prefix = f"exp_{idx+1}"

        for j in range(len(result_log) - 1):
            if idx == 0:
                df.loc[j, "question_id"] = result_log[j]["question_id"]
                df.loc[j, "question"] = result_log[j]["question"]
            if not prompt_chain:
                df.loc[j, f"{prefix}_llm_answer"] = result_log[j]["llm_answer"]
            else:
                df.loc[j, f"{prefix}_llm_answer"] = result_log[j]["llm_answer"]
                df.loc[j, f"{prefix}_output_analyzer"] = result_log[j][
                    "output_analyzer"
                ]
            df.loc[j, f"{prefix}_correct_answer"] = result_log[j]["correct_answer"]
            df.loc[j, f"{prefix}_is_correct"] = result_log[j]["is_correct"]
            df.loc[j, f"{prefix}_response"] = result_log[j]["response"]

    df = df.sort_values(by="mean correctness rate", ascending=True)
    df.to_csv(f"{base_dict}/wrong_answers.csv", index=False)
    print("Wrong answers saved to wrong_answers.csv")


def run_single_benchmark(
    dataset_path,
    mdl_name,
    sys_prompt,
    temperature=1,
    log_path="outputs/",
    print_each_run=False,
):
    """
    # Description
    This function runs a single benchmark on all the questions in the dataset.
    It uses a specified model to answer questions and logs the process and results.

    # Parameters
    - dataset_path: Path to the dataset file, which contains the questions for the benchmark.
    - mdl_name: Name of the model to be used for answering the questions.
    - sys_prompt: System prompt to be used when sending messages to the OpenAI API.
    - temperature: Sampling temperature for the model's output, controlling the randomness of the output. Default is 1.
    - log_path: Directory path to save the log files. Default is 'putputs/'.

    # Returns
    - predicted_result: A list containing the model's predicted answers.
    """
    # Read the dataset
    dataset = pd.read_csv(dataset_path)

    # Run the benchmark
    client = OpenAI()
    n_correct = 0  # Counter for the correct answers.
    # Initialize the log
    log_entries = []  # Initialize the log
    predicted_result = []  # Initialize the prediction
    results_per_question = []
    for idx, row in dataset.iterrows():
        question, answer = row["question"], row["answer"]
        # Clean up the prompt and answer
        clean_question = question.strip()
        correct_answer = answer.strip().split("\n")[0]

        # Query the OpenAI API using ChatCompletion
        input_msg = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": clean_question},
        ]
        completion = client.chat.completions.create(
            model=mdl_name, messages=input_msg, temperature=temperature
        )

        # Extract GPT-4's answer and explanation
        response = completion.choices[0].message.content
        generated_answer = response.split("[Answer]")[-1].strip().split("\n")[0]

        # Regular expression to extract the letter
        letters = re.findall(r"\[([a-zA-Z])\]", generated_answer)
        extracted_answer = ", ".join(letters)

        is_correct = extracted_answer == correct_answer
        results_per_question.append(is_correct)

        if is_correct:
            n_correct += 1

        # Log the result
        log_entry = {
            "question_id": f"Question_{idx}",
            "question": clean_question,
            "llm_answer": generated_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "temperature": temperature,
            "response": response,
        }
        log_entries.append(log_entry)

        # Save the result
        predicted_result.append(extracted_answer)

        if print_each_run:
            # Output result
            print(f"Question {idx}: {clean_question}\n LLM response: {response}\n")

    # Calculate the correct percentage
    accuracy = n_correct / len(dataset)
    print(f"The accuracy is {accuracy:.2%} ({n_correct}/{len(dataset)})")
    # Log the result
    log_entry = {
        "question_id": "Summary",
        "question": None,
        "llm_answer": "",
        "correct_answer": "",
        "is_correct": f"The accuracy is {accuracy:.2%} ({n_correct}/{len(dataset)})",
        "temperature": "",
        "response": "",
    }
    log_entries.append(log_entry)

    # Save the log to a file
    # Generate log name.
    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime("%Y%m%d%H%M%S")
    log_name = f"{mdl_name}_{datetime_string}.json"
    log_file_path = f"{log_path}/{log_name}"
    # Save log.
    with open(log_file_path, "w") as log_file:
        json.dump(log_entries, log_file, indent=4)

    return results_per_question


if __name__ == "__main__":
    #     # Datapath.
    #     dataset_path = "generative-ai-for-reliability-engineering/train.csv"

    #     # Define the parameters of the benchmark
    #     mdl_name = "gpt-4o-mini"  # Model name.
    #     # mdl_name = 'gpt-4o' # Model name.

    #     prompt_name = "zero_shot"  # A string summarizing your prompt.
    #     temperature = 1  # Temperature of the model. Default is 1.

    #     # Generate log foler
    #     current_datetime = datetime.now()
    #     datetime_string = current_datetime.strftime("%Y%m%d%H%M%S")
    #     log_folder = f"outputs/{mdl_name}_prompt_{prompt_name}_training_data_temperature_{temperature}_{datetime_string}/"
    #     # Check if the folder exists
    #     if not os.path.exists(log_folder):
    #         # Create the folder
    #         os.makedirs(log_folder)

    #     sys_prompt = """You will be given a multiple choice question about reliability engineering. Choose the correct answer. At the end of your response, start a new line and use the following format to output your answer: [Answer] [The letters you choose]. For example, if you think the answer [a] is correct, output [Answer] [a]. If you think there are multiple correct answer, using a comma to separate them, e.g., [Answer] [a], [b]. Limit your output to 400 words maximum.
    # """

    #     # Run the benchmark
    #     df_results = pd.DataFrame()  # Dataframe to store the results.

    #     n_experiments = 5
    #     for i in tqdm(range(n_experiments)):
    #         print(f"Experiment {i+1}/{n_experiments}")
    #         result = run_single_benchmark(
    #             dataset_path,
    #             mdl_name,
    #             sys_prompt,
    #             temperature,
    #             log_folder,
    #             print_each_run=True,
    #         )
    #         df_results[f"Experiment_{i+1}"] = result

    # # Calculate pass rate for each test (row-wise mean)
    # df_results = df_results.astype(int)

    # # Calculate the mean and standard deviation for each row
    # df_results["mean"] = df_results.mean(axis=1)
    # df_results["std"] = df_results.std(axis=1)

    # # Calculate the mean of each column excluding the 'mean' and 'std' columns
    # column_means = df_results.iloc[:, :-2].mean()
    # # Add a new row for overall accuracy (mean of columns excluding the last two)
    # df_results.loc["overall_accuracy"] = column_means

    # df_results.loc["overall_accuracy", "mean"] = (
    #     df_results.loc["overall_accuracy"].iloc[:-2].mean()
    # )
    # df_results.loc["overall_accuracy", "std"] = (
    #     df_results.loc["overall_accuracy"].iloc[:-2].std()
    # )

    # df_results.to_csv(f"{log_folder}/df_results.csv", index=False)

    # print(f"Prediction results saved to {log_folder}/df_results.csv")

    # Extract failed answers
    log_folder = "outputs/gpt-4o-mini_prompt_zero_shot_training_data_temperature_1_20241210154024"
    extract_failed_answers(log_folder, prompt_chain=False)

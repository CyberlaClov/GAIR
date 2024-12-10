from openai import OpenAI
import re, json, os
import pandas as pd
import numpy as np
from datetime import datetime


def run_single_benchmark(dataset_path, mdl_name, sys_prompt, temperature=1, log_path='outputs/'):
    '''
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
    '''
    # Read the dataset
    dataset = pd.read_csv(dataset_path)

    # Run the benchmark
    client = OpenAI()
    
    log_entries = [] # Initialize the log
    predicted_result = [] # Initialize the prediction
    for idx, row in dataset.iterrows():
        question = row['question']
        # Clean up the prompt and answer
        clean_question = question.strip()
        
        # Query the OpenAI API using ChatCompletion
        input_msg = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": clean_question}
            ]
        completion = client.chat.completions.create(
            model=mdl_name,
            messages=input_msg,
            temperature=temperature
            )

        # Extract GPT-4's answer and explanation
        response = completion.choices[0].message.content
        generated_answer = response.split("[Answer]")[-1].strip().split('\n')[0]

        # Regular expression to extract the letter
        letters  = re.findall(r'\[([a-zA-Z])\]', generated_answer)
        extracted_answer = ', '.join(letters)

        # Log the result
        log_entry = {
            "question_id": f"Question_{idx}",
            'question': clean_question,
            "llm_answer": generated_answer,
            'temperature': temperature,
            "response": response
        }
        log_entries.append(log_entry)

        # Save the result
        predicted_result.append(extracted_answer)

        print(f'Question {idx}: {clean_question}\n LLM response: {response}\n')


    # Save the log to a file
    # Generate log name.
    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime("%Y%m%d%H%M%S")
    log_name = f'{mdl_name}_{datetime_string}.json'
    log_file_path = f'{log_path}/{log_name}'
    # Save log.
    with open(log_file_path, 'w') as log_file:
        json.dump(log_entries, log_file, indent=4)

    return predicted_result



if __name__ == '__main__':
    # Datapath.
    dataset_path = 'dataset/test.csv'

    # Define the parameters of the benchmark
    mdl_name = 'gpt-4o-mini' # Model name.
    # mdl_name = 'gpt-4o' # Model name.

    prompt_name = 'zero_shot' # A string summarizing your prompt.    
    temperature = 1 # Temperature of the model. Default is 1.
    
    # Generate log foler
    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime("%Y%m%d%H%M%S")
    log_folder = f'outputs/{mdl_name}_prompt_{prompt_name}_temperature_{temperature}_{datetime_string}/'
    # Check if the folder exists
    if not os.path.exists(log_folder):
        # Create the folder
        os.makedirs(log_folder)

    sys_prompt = '''You will be given a multiple choice question about reliability engineering. Choose the correct answer. At the end of your response, start a new line and use the following format to output your answer: [Answer] [The letters you choose]. For example, if you think the answer [a] is correct, output [Answer] [a]. If you think there are multiple correct answer, using a comma to separate them, e.g., [Answer] [a], [b]. Limit your output to 400 words maximum. 
'''

    # Run the benchmark
    df_results = pd.DataFrame() # Dataframe to store the results.
    for i in range(1, 6):    
        predicted_result = run_single_benchmark(dataset_path, mdl_name, sys_prompt, temperature, log_folder)    
        if i==1:
            df_results["question_id"] = np.arange(len(predicted_result))+1      
        df_results[f"prediction_{i}"] = predicted_result
    
    df_results.to_csv(f"{log_folder}/predition_{mdl_name}_{prompt_name}.csv", index=False)

    print(f"Prediction results saved to {log_folder}/predition_{mdl_name}_{prompt_name}.csv")


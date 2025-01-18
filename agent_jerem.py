
from openai import OpenAI
import re, json, os
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import subprocess

def llm_runner(sys_prompt, question, client, mdl_name, tools=None, temperature=1):
    
    input_msg = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question}
        ]
    
    if tools is not None:        
        completion = client.chat.completions.create(
            model=mdl_name,
            messages=input_msg,
            temperature=temperature,
            tools=tools,
            seed=133442
            )
    else:
        completion = client.chat.completions.create(
            model=mdl_name,
            messages=input_msg,
            temperature=temperature,
            seed=133442
            )

    return completion

def execute_python_script(script: str) -> str:
    # Garde la même fonction execute_python_script
    time_out = 5
    python_executable = sys.executable

    with open("sandbox_script.py", "w") as file:
        file.write(script)

    try:
        result = subprocess.run(
            [python_executable, "sandbox_script.py"],
            capture_output=True,
            text=True,
            timeout=time_out,
        )
        print("Output:", result.stdout)
        os.remove("sandbox_script.py")
        if result.stdout == "":
            return None
        else:
            return result.stdout.rstrip('\n')
    except Exception as e:
        os.remove("sandbox_script.py")
        print(f"An error occurred: {e}")
        return None

def kaggle_agent(prompt, client, mdl_name="gpt-4o-mini", temperature=1):
    sys_prompt = '''You are a helpful AI assistant that can help with data analysis and Python coding.
    If calculations or data manipulation is needed, you can generate and execute Python code.
    When you need to execute code, create a Python script and use the execute_python_script tool.
    Make sure to explain your reasoning and results clearly.'''
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_python_script",
                "description": "Executes a python script in a sandboxed environment. Return the stdout printed by the script.",
                "parameters": {
                    "type": "object",
                    "required": ["script"],
                    "properties": {
                        "script": {
                            "type": "string",
                            "description": "The python script to execute."
                        }
                    }
                }
            }
        }
    ]

    completion = llm_runner(sys_prompt, prompt, client, mdl_name, tools, temperature)
    
    if completion.choices[0].finish_reason == "tool_calls":
        tool_call = completion.choices[0].message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        
        try:
            script = arguments.get('script')
            result = execute_python_script(script)
            if result is None:
                result = "Code execution failed"
        except:
            result = "Code execution failed"
            
        function_message = {
            "role": "tool",
            "content": f"Execution result: {result}",
            "tool_call_id": tool_call.id
        }

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
            completion.choices[0].message,
            function_message
        ]

        final_completion = client.chat.completions.create(
            model=mdl_name,
            messages=messages,
            temperature=temperature
        )
        return final_completion.choices[0].message.content
    
    return completion.choices[0].message.content

if __name__ == "__main__":
    client = OpenAI()
    mdl_name = "gpt-4o-mini" 
    
    # Example usage
    prompt = "Analysez les données de mon fichier data.csv et créez un graphique"
    response = kaggle_agent(prompt, client, mdl_name)
    print(response)

    
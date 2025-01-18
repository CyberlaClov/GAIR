import os
import subprocess


def execute_python_script(script: str, venv_path: str) -> str:
    """### Description
    Executes a python script in a sandboxed environment. Return the stdout printed by the script.

    ### Parameters
    - script: str
        The python script to execute.

    ### Return:
    - str: The stdout printed by the script.
    """
    time_out = 5

    # Determine the Python executable path based on the OS
    if os.name == "nt":  # Windows
        python_executable = os.path.join(venv_path, "Scripts", "python.exe")
    else:  # Linux/Mac
        python_executable = os.path.join(venv_path, "bin", "python")

    # Check if the Python executable exists
    if not os.path.exists(python_executable):
        raise FileNotFoundError(f"Python interpreter not found at: {python_executable}")

    with open("sandbox_script.py", "w") as file:
        file.write(script)

    try:
        result = subprocess.run(
            [python_executable, "sandbox_script.py"],
            capture_output=True,
            text=True,
            timeout=time_out,
        )
        # print("Output:", result.stdout)
        os.remove("sandbox_script.py")
        if result.stdout == "":
            return None
        else:
            return result.stdout.rstrip("\n")
    except subprocess.TimeoutExpired:
        os.remove("sandbox_script.py")
        print("Script took too long to execute.")
        return None
    except Exception as e:
        os.remove("sandbox_script.py")
        print(f"An error occurred: {e}")
        return None


tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_python_script",
            "description": "Executes a python script in a sandboxed environment. Return the stdout printed by the script.",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": ["script"],
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "The python script to execute.",
                    }
                },
                "additionalProperties": False,
            },
        },
    }
]

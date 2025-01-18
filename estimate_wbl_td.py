import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, weibull_min


def estimate_wbl(data, censored, alpha=0.05):
    """### Description
    Estimate the shape and scale parameters of a Weibull distribution using maximum likelihood estimation (MLE).

    ### Parameters
    - data: array-like
        The observed data points.
    - censored: array-like
        A boolean array indicating which data points are censored (True) or observed (False).
    - alpha: confidence level (default: 0.05)

    ### Returns
    - shape, scale, shape_ci, scale_ci: shape, scale parameters, and their confidence intervals.
    """
    initial_guess = [1.0, 1.0]  # Initial guess for shape and scale
    result = minimize(
        weibull_log_likelihood_censored,
        initial_guess,
        args=(data, censored),
        method="L-BFGS-B",
        bounds=[(1e-6, None), (1e-6, None)],
    )

    mle_shape, mle_scale = result.x

    # Calculate standard errors using the Fisher Information Matrix
    cov_matrix = fisher_information_censored(result.x, data, censored)
    se_shape, se_scale = np.sqrt(np.diag(cov_matrix))

    # 95% Confidence intervals (assuming normal approximation)
    z = norm.ppf(1 - alpha / 2)
    shape_ci = (mle_shape - z * se_shape, mle_shape + z * se_shape)
    scale_ci = (mle_scale - z * se_scale, mle_scale + z * se_scale)

    return mle_shape, mle_scale, shape_ci, scale_ci


# Step 4: Calculate confidence intervals
def fisher_information_censored(params, data, censored):
    shape, scale = params
    n = len(data)
    fisher_info = np.zeros((2, 2))
    fisher_info[0, 0] = n / shape**2  # Approximation: d^2/d(shape)^2
    fisher_info[1, 1] = n / scale**2  # Approximation: d^2/d(scale)^2
    return np.linalg.inv(fisher_info)


# Step 2: Define the log-likelihood function for the Weibull distribution with right censoring
def weibull_log_likelihood_censored(params, data, censored):
    shape, scale = params
    if shape <= 0 or scale <= 0:
        return np.inf  # Return a high cost for invalid parameters
    inverted_list = [not value for value in censored]
    uncensored_log_likelihood = np.sum(
        inverted_list * weibull_min.logpdf(data, shape, scale=scale)
    )
    censored_log_likelihood = np.sum(
        censored * weibull_min.logsf(data, shape, scale=scale)
    )
    return -(
        uncensored_log_likelihood + censored_log_likelihood
    )  # Negative log-likelihood


def llm_runner(sys_prompt, question, client, mdl_name, tools=None, temperature=1):
    # Query the OpenAI API using ChatCompletion
    input_msg = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
    ]

    if tools is not None:
        completion = client.chat.completions.create(
            model=mdl_name,
            messages=input_msg,
            temperature=temperature,
            tools=tools,
            seed=133442,
        )
    else:
        completion = client.chat.completions.create(
            model=mdl_name, messages=input_msg, temperature=temperature, seed=133442
        )

    return completion


if __name__ == "__main__":
    import json

    from openai import OpenAI

    # Define system and usr prompt
    sys_prompt = """You are a data scientist working on analyzing survival data using Weibull distribution.
You have access to a tool that can estimate Weibull parameters from potentially censored data.
The tool requires:
- data: array of testing times
- censored: boolean array indicating censored (True) or observed (False) data points
- alpha: confidence level for parameter estimation"""

    usr_prompt = """Please analyze this survival data using the estimate_wbl tool:
Testing times T = [2.05509436, 4.34957357, 3.4424862, 2.86644082, 1.2355662, 1.23546211, 0.73385892, 4.25453614, 2.87606317, 3.32885124]
Censoring indicators C = [False, True, False, False, False, False, False, False, False, False]
Use alpha = 0.05 for the confidence intervals.
Please estimate the Weibull shape and scale parameters along with their confidence intervals."""

    # Provide a sandbox for evaluating the generated script.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "estimate_wbl",
                "description": "Estimate the shape and scale parameters of a Weibull distribution using maximum likelihood estimation (MLE).",
                "parameters": {
                    "type": "object",
                    "required": ["data", "censored", "alpha"],
                    "properties": {
                        "data": {
                            "type": "array",
                            "description": "The observed data points.",
                            "items": {"type": "number"},
                        },
                        "censored": {
                            "type": "array",
                            "description": "A boolean array indicating which data points are censored (True) or observed (False).",
                            "items": {"type": "boolean"},
                        },
                        "alpha": {
                            "type": "number",
                            "description": "Confidence level (default: 0.05)",
                        },
                    },
                },
            },
        }
    ]

    # Generate the analysis.
    client = OpenAI()
    mdl_name = "gpt-4o-mini"
    completion = llm_runner(sys_prompt, usr_prompt, client, mdl_name, tools)

    # Execute the python script to get the result.
    script_executed = False
    if completion.choices[0].finish_reason == "tool_calls":
        # Extract the arguments
        # Note this code assumes we have already determined that the model generated a function call. See below for a more production ready example that shows how to check if the model generated a function call
        tool_call = completion.choices[0].message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)

        try:
            # Etract the paraemetrs.
            data = arguments["data"]
            censored = arguments["censored"]

            # Call the tool:

            mle_shape, mle_scale, shape_ci, scale_ci = estimate_wbl(
                data, censored, alpha=0.05
            )
            script_executed = True

        except:  # Handled exception
            mle_shape, mle_scale, shape_ci, scale_ci = None, None, None, None

        # Create a message containing the result of the function call
        function_call_result_message = {
            "role": "tool",
            "content": f"mle_shape: {mle_shape}, mle_scale: {mle_scale}, shape_ci: {shape_ci}, scale_ci: {scale_ci}",
            "tool_call_id": completion.choices[0].message.tool_calls[0].id,
        }

        # Prepare the chat completion call payload

        completion_payload = {
            "model": mdl_name,
            "messages": [
                {"role": "system", "content": f"{sys_prompt}"},
                {"role": "user", "content": f"{usr_prompt}"},
                completion.choices[0].message,
                function_call_result_message,
            ],
        }

        # Call the OpenAI API's chat completions endpoint to send the tool call result back to the model
        try:
            completion_final = client.chat.completions.create(
                model=completion_payload["model"],
                messages=completion_payload["messages"],
            )
        except:
            completion_final = completion
            script_executed = False
    else:
        completion_final = completion

    # Output results
    print(completion_final.choices[0].message.content)
    print("Script executed successfully:", script_executed)

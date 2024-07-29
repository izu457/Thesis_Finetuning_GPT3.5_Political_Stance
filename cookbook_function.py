import json
import os
import numpy as np
import tiktoken
from collections import defaultdict
import openai
import pandas as pd
import matplotlib.pyplot as plt
import requests
import base64
import logging
import itertools
from tenacity import retry, stop_after_attempt, wait_random_exponential

def check_format(dataset):
    if dataset is None:
        print("Dataset is empty")
        return
    # Format error checks
    format_errors = defaultdict(int)
    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
            
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
            
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            function_call = message.get("function_call", None)
                
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
        
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")

# HELPER FUNCTIONS

# cl100k_base is the tokenizer used by ChatGPT3.5 and ChatGPT4
encoding = tiktoken.get_encoding("cl100k_base")

# not exact!
# simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            # .encode returns the list of tokens (so that you can count them with len())
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_base64_padding(base64_str):
    """Fix the padding of a Base64 encoded string."""
    missing_padding = len(base64_str) % 4
    if missing_padding:
        base64_str += '=' * (4 - missing_padding)
    return base64_str

def get_ft_results(file_id):
    """
    Given a result_files id of a finished fine-tuning job, a request is made to the OpenAI API
    to retrieve the content of the file. Content is decoded from Base64 and saved to a 
    csv file, which can be later loaded as a pandas DataFrame.
    """
    headers = {'Authorization': f'Bearer sk-proj-GL73kbRwhRpgN3EmXz1YT3BlbkFJEMJhTsinxQDel42BZdNz'}
    try:
        response = requests.get(f"https://api.openai.com/v1/files/{file_id}/content", headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx and 5xx)
        logger.info("Received response for file content.")
        if response.content:
            try:
                # Parse the JSON content
                decoded_content = base64.b64decode(response.content).decode('utf-8')
                decoded_content = fix_base64_padding(decoded_content)
                logger.info("Parsed JSON content successfully.")
                with open("decoded_content.csv", "w") as f:
                    f.write(decoded_content)
                logger.info("File 'decoded_content.csv' written successfully.")
            except (ValueError, base64.binascii.Error) as e:
                # Handle the case where the response is not valid JSON
                logger.error("Response content could not be parsed as JSON")
                logger.error(f"Exception: {e}")
                logger.error(response.content)
        else:
            logger.error("Response content is empty")
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as err:
        logger.error(f"Error occurred: {err}")
    return "decoded_content.csv"

def get_checkpoint_results(job_id):
    """
    Given a job_id of a finished fine-tuning job, a request is made to the OpenAI API
    to retrieve the checkpoints which have been saved during the fine-tuning process. 
    Content is decoded from Base64 and saved to a csv file, which can be later loaded 
    as a pandas DataFrame.
    """
    headers = {'Authorization': f'Bearer sk-proj-GL73kbRwhRpgN3EmXz1YT3BlbkFJEMJhTsinxQDel42BZdNz'}
    try:
        response = requests.get(f"https://api.openai.com/v1/fine_tuning/jobs/{job_id}/checkpoints", headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx and 5xx)
        if response.content:
            try:
                # Decode the json encoded content
                decoded_content = response.content.decode('utf-8')
                logger.info("Decoded response content successfully.")
                # Normalize the Json content to a df
                decoded_content_df = json.dumps(json.loads(decoded_content), indent=4)
            except (ValueError, json.JSONDecodeError) as e:
                # Handle the case where the response is not valid JSON
                logger.error("Response content could not be parsed as JSON")
                logger.error(f"Exception: {e}")
                logger.error(f"Content: {response.content}")
        else:
            logger.error("Response content is empty")
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as err:
        logger.error(f"Error occurred: {err}")
    return decoded_content_df


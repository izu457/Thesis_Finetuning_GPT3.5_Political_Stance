import json
import os
import numpy as np
import time
import tiktoken
from collections import defaultdict
import openai
import pandas as pd
import requests
import base64
import random
import logging
import itertools

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Authenticate
os.environ['TestKey3'] = 'sk-proj-GL73kbRwhRpgN3EmXz1YT3BlbkFJEMJhTsinxQDel42BZdNz' 
client = openai.OpenAI(api_key=os.environ['TestKey3'])
headers = {
    "Authorization": f"Bearer {os.environ['TestKey3']}"
}

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

def fix_base64_padding(base64_str):
    """
    For strings that cannot be decoded from Base64 without
    a padding problem.
    Fixes the padding of a Base64 encoded string, returns the
    fixed string.
    """
    missing_padding = len(base64_str) % 4
    if missing_padding:
        base64_str += '=' * (4 - missing_padding)
    return base64_str

def validate_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error on line {line_number}: {e}")
        print("Valid jsonl file")

def process_jobs(lr_multiplier, n_epochs, batch_size, train_upload, val_upload, rate_limit, max_retries=3):
    """
    Takes as input lists of hyperparameters and generates all possible combinations
    to loop through and try as hyperparameters for a fine-tuning job.
    Takes as input the file id of the training file and the rate limit given in 
    possible requests per minute. 
    If rate limit is exceeded, maximum retries default to 3. 
    Processes each combination by creating a fine-tuning job with the specified 
    hyperparameters. Respects the rate limit by waiting between requests. 
    Returns a list of job ids of the finished jobs.
    """
    # Generate all combinations of hyperparameters
    combinations = list(itertools.product(lr_multiplier, n_epochs, batch_size))
    logger.info(len(combinations), "hyperparameter combinations in total")
    logger.info(combinations)
    # Rate limit settings
    # code snippets from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
    request_interval = 60 / rate_limit  # Interval in seconds
    logger.info(f"Standard request interval is set to {request_interval} seconds.")
    last_request_time = time.time() - request_interval  # Initialize to allow immediate first request
    # Loop through each combination of hyperparameters while using batches
    all_job_ids = []
    for lr_multiplier, epoch, batch in combinations:
        # Retry loop to handle rate limiting
        for attempt in range(max_retries):
            # Setting time to wait between requests
            logger.info(f"Processing hyperparameters (lr={lr_multiplier}, epoch={epoch}, batch={batch})")
            current_time = time.time()
            elapsed_time = current_time - last_request_time
            # if time that the processing took is less than the request interval, 
            # add the difference in time as sleep_time    
            if elapsed_time < request_interval:
                sleep_time = request_interval - elapsed_time 
                logger.info(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            try:
                # save response of ft-job after creating it with as much metadata as possible
                response = client.fine_tuning.jobs.create(
                    training_file=train_upload.id,  # file id returned after upload to API
                    validation_file=val_upload.id, # file id returned after upload to API
                    model="gpt-3.5-turbo",
                    suffix="mig_gen",
                    seed=124,
                    hyperparameters={
                        "n_epochs": epoch,
                        "batch_size": batch,
                        "learning_rate_multiplier": lr_multiplier
                    }
                )
                job_id = response.id
                all_job_ids.append(job_id)
                logger.info(f"Job created with ID {job_id}")
                last_request_time = time.time()
                break  # Break out of the retry loop if the request is successful
            except Exception as e:
                if "429" in str(e):  # Check if the error is a rate limiting error
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.error(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"An error occurred: {e}")
                    break
    logger.info("All batches processed.")
    return all_job_ids

def monitor_jobs(job_id, limit=10):
    """
    Takes as input a single job id of a finished fine tuning job and 
    returns a list of up to limit (default limit is 10) saved events 
    of the fine-tuning job.
    """
    ft_events = []
    event_ls = [client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=job_id,
            limit=limit
                )
            ]
    for event in event_ls:
        #print(event)
        for ft_event in event:
            #print(ft_event)
            ft_events.append(ft_event.id)
    return ft_events

def extract_job_info(all_job_ids):
    """
    Takes as input a list of job ids from created fine-tuning jobs.
    Creates a list of dictionaries with information about each finished 
    fine-tuning job, including hyperparameters, events, and the result file
    name.
    Converts the list of dictionaries to a dataframe and returns the dataframe.
    """
    results = []
    for job_id in all_job_ids:
        # Access OpenAI API to retreive job information
        job_results = client.fine_tuning.jobs.retrieve(job_id)
        results.append({
            "job_id": job_results.id,
            "learning_rate_multiplier": job_results.hyperparameters.learning_rate_multiplier,
            "n_epochs": job_results.hyperparameters.n_epochs,
            "batch_size": job_results.hyperparameters.batch_size,
            "status": job_results.status,
            "event_ids": monitor_jobs(job_id),
            "result_file_name": job_results.result_files
        })
    # convert results to a pandas DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def get_ft_results(file_id):
    """
    Given a result files id of a finished fine-tuning job, a request is made to the OpenAI API
    to retrieve the content of the file. Content is decoded from Base64 and saved to a 
    csv file, which can be later loaded as a pandas DataFrame.
    """
    headers = {'Authorization': 'Bearer sk-proj-GL73kbRwhRpgN3EmXz1YT3BlbkFJEMJhTsinxQDel42BZdNz'}
    try:
        response = requests.get(f"https://api.openai.com/v1/files/{file_id}/content", headers=headers)
        # print the url of the request
        print(response.url)
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
    Content is decoded from Base64 and saved to a string.
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
                # Normalize the Json content to a string
                decoded_content = json.dumps(json.loads(decoded_content), indent=4)
            except (ValueError, json.JSONDecodeError) as e:
                # Handle the case where the response is not valid JSON
                logger.error("Response content could not be parsed as JSON")
                logger.error(f"Exception: {e}")
                logger.error(f"Content: {response.content}")
        else:
            logger.error("Response content is empty")
    #except requests.exceptions.HTTPError as http_err:
    #    logger.error(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as err:
        logger.error(f"Error occurred: {err}")
    return decoded_content

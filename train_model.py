import openai
import time
import logging


openai.api_key = "sk-proj-GL73kbRwhRpgN3EmXz1YT3BlbkFJEMJhTsinxQDel42BZdNz"


def configure_logging():
    """
    Configures logging settings.
    """
    logging.basicConfig(filename='output.log', level=logging.INFO,
                        format='%(asctime)s [%(levelname)s]: %(message)s')
    return logging.getLogger()


def upload_file(file_name):
    """
    Uploads a file to OpenAI for fine-tuning.

    :param file_name: Path to the file to be uploaded.
    :return: Uploaded file object.
    """
    # Note: For a 400KB train_file, it takes about 1 minute to upload.
    uploaded_files = []
    for file_name in file_names:
        logger.info(f"Uploading file: {file_name}")
        try:
            # Upload the file
            train_file_upload = openai.File.create(file=open(file_name, "rb"), purpose="fine-tune")
            logger.info(f"Uploaded file with id: {train_file_upload.id}")

            # Wait for the file to be processed
            while True:
                logger.info("Waiting for file to process...")
                file_handle = openai.File.retrieve(id=train_file_upload.id)

                if file_handle.status == "processed":
                    logger.info("File processed")
                    break
                time.sleep(60)

            uploaded_files.append(train_file_upload)
        except Exception as e:
            logger.error(f"Error uploading file {file_name}: {e}")

    return uploaded_files


if __name__ == '__main__':
    # Configure logger
    logger = configure_logging()

    train_file_name = "data/finetuning_data/small_training_data.jsonl"
    val_file_name = "data/finetuning_data/small_validation_data.jsonl"
    file_names = [train_file_name, val_file_name]
    uploaded_files = upload_file(file_names)

    logger.info(uploaded_files)
    job = openai.FineTuningJob.create(
        training_file = uploaded_files[0].id,
        validation_file = uploaded_files[1].id, # file id returned after upload to API
        model="gpt-3.5-turbo",
        suffix="small",
        seed=124,
        hyperparameters={
            "n_epochs": 3,
	        "batch_size": 1,
	        "learning_rate_multiplier": 0.01
        }
        )
    logger.info(f"Job created with id: {job.id}")

    # Note: If you forget the job id, you can use the following code to list all the models fine-tuned.
    # result = openai.FineTuningJob.list(limit=10)
    # print(result)


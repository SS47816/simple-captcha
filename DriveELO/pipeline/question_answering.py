from pathlib import Path
import string
from tqdm import tqdm
import pandas as pd
import hashlib
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def generate_mmlu_question_ids(subject: str, question: str) -> str:
    question_hash = hashlib.md5(question.encode()).hexdigest()
    return f"{subject}_{question_hash}"

def index_to_letter(index: int) -> str:
    return chr(index + ord('A')) if 0 <= index <= 25 else "?"

def evaluate_mmlu(prediction_result_folder: Path, task="all", split='test', batch_size=16, device="cuda"):
    """
    Evaluates a language model on the MMLU dataset and saves results to a pickle file.

    This function loads a pre-trained language model, evaluates it on a specified MMLU task,
    and saves the results to a pickle file. It handles the entire evaluation pipeline including
    data loading, model inference, and result saving.

    Args:
        task (str): Specific MMLU task to evaluate (e.g., "computer_security"). Use "all" for full evaluation.
        split (str): The dataset split to use for evaluation ('train', 'val', 'test').
        batch_size (int): Number of samples to process in each batch during evaluation.
        device (str): The device to run the evaluation on ("cuda" or "cpu").

    Returns:
        float: Accuracy score on the given MMLU task.

    Note:
        This function uses a list of fixed models and saves results to a pickle file.
    """

    # Load MMLU dataset
    dataset = load_dataset("cais/MMLU", task, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    choice_labels = list(string.ascii_uppercase[:26])  # Generate A, B, C... based on choice count

    # Define models
    models_dict = {
        "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",

        "BLOOM-7B1": "bigscience/bloom-7b1",
        "BLOOM-3B": "bigscience/bloom-3b",
        "BLOOM-1B7": "bigscience/bloom-1b7",
        "BLOOM-1B1": "bigscience/bloom-1b1",
        "BLOOM-560M": "bigscience/bloom-560m",

        "BLOOMZ-7B1": "bigscience/bloomz-7b1",
        "BLOOMZ-3B": "bigscience/bloomz-3b",
        "BLOOMZ-1B7": "bigscience/bloomz-1b7",
        "BLOOMZ-1B1": "bigscience/bloomz-1b1",
        "BLOOMZ-560M": "bigscience/bloomz-560m",

        # "Llama-2-7B": "meta-llama/Llama-2-7b-hf",
        # "Llama-2-13B": "meta-llama/Llama-2-13b-hf",

        # "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",

        # "Mistral-7B": "mistralai/Mistral-7B",
        # "Falcon-7B": "tiiuae/falcon-7b",
        # "GPT-J-6B": "EleutherAI/gpt-j-6B",
        # "GPT-NeoX-20B": "EleutherAI/gpt-neox-20b",
    }

    question_dict = {}
    # Evaluation loop
    with torch.no_grad():
        # Iterate over each model
        for model_name, model_url in models_dict.items():
            print(f"Processing model: {model_name}")
            # Load quantized model
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(model_url, quantization_config=quant_config, trust_remote_code=True, device_map="auto", torch_dtype="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_url, trust_remote_code=True)
            is_8bit = hasattr(model, "is_loaded_in_8bit") and model.is_loaded_in_8bit
            model.eval()

            all_preds = []
            all_labels = []
            results_dict = {}  # List to store results for pickle file

            # Iterate over batches
            for batch_id, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {task}")):
                subjects = np.array(batch['subject'])
                questions = np.array(batch['question'])
                choices_list = np.array(batch['choices']).transpose()
                answers = np.array(batch['answer'])

                # Hash and save the hash to question pairs
                question_ids = []
                for subject, question, choices, answer in zip(subjects, questions, choices_list, answers):
                    question_id = generate_mmlu_question_ids(subject, question)
                    question_ids.append(question_id)
                    if question_id not in question_dict:
                        question_dict[question_id] = {
                            'id': question_id,
                            'question': question,
                            'subject': subject,
                            'choices': choices,
                            'answer': answer
                        }

                # Format inputs for multiple-choice questions
                inputs = []
                for q, choices in zip(questions, choices_list):
                    formatted_choices = "\n".join(
                        [f"{label}) {choice}" for label, choice in zip(choice_labels, choices)]
                    )
                    inputs.append(f"Question: {q}\n{formatted_choices}\nAnswer:")

                # Get filenames for the current batch
                start_idx = batch_id * batch_size
                end_idx = start_idx + len(batch)

                # Model inference
                try:
                    # Tokenize inputs
                    tokenized_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
                    if not is_8bit:
                        tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}

                    outputs = model(**tokenized_inputs)

                    # Handling different model types
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits  # If model outputs logits
                    else:
                        logits = outputs[0]  # For some models like GPT-style models

                    preds = torch.argmax(logits, dim=-1)  # Predict token index for each question

                except Exception as e:
                    logger.error(f"Error processing batch {start_idx} to {end_idx} for model {model_name}: {str(e)}")
                    continue

                # Ensure preds matches the number of questions
                preds = preds[:, -1] if preds.dim() > 1 else preds  # Select last token only if needed

                # Convert predictions to human-readable format
                decoded_preds = [tokenizer.decode([pred]).strip() for pred in preds.cpu().numpy()]
                decoded_labels = [index_to_letter(label) for label in answers]

                # Save results for each test case
                for question_id, true_label, predicted in zip(question_ids, decoded_labels, decoded_preds):
                    results_dict[question_id] = {"Test Case": question_id, "True Label": true_label, model_name: predicted}

                # Store predictions and correct labels
                all_preds.extend(decoded_preds)
                all_labels.extend(decoded_labels)

            del model
            del tokenizer
            torch.cuda.empty_cache()

            # Save results to a pickle file
            df = pd.DataFrame(results_dict.values())
            prediction_file_name = prediction_result_folder.joinpath(f"{model_name}.pkl")
            df.to_pickle(prediction_file_name)
            print(f"Results saved to {prediction_file_name}")

            # Compute accuracy
            accuracy = accuracy_score(all_labels, all_preds)
            print(f"{model_name} Final Accuracy: {accuracy:.2%}")

        # Save hashed dataset to a pickle file
        df = pd.DataFrame(question_dict.values())
        filename = prediction_result_folder.parent.joinpath(f"mmlu_hashed.pkl")
        df.to_pickle(filename)
        print(f"Hashed MMLU dataset saved to {filename}")

    return


def main():
    # Dataset Selection (Change Here)
    dataset_name = "MMLU"
    dataset_task = 'all' # 'high_school_physics', 'all'
    dataset_split_name = "test" # "val", "test", "dev", "auxiliary_train"

    # Input and Output Data folders
    output_folder = Path(f"./data/question_answering/{dataset_name}/{dataset_split_name}/")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Step 1: Run inference on ImageNet
    prediction_result_folder = output_folder.joinpath("predictions/")
    prediction_result_folder.mkdir(parents=True, exist_ok=True)
    evaluate_mmlu(prediction_result_folder, task=dataset_task, split=dataset_split_name, batch_size=16)


if __name__ == "__main__":
    main()

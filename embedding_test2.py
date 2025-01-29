import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import json
import os
import random
from datasets import load_dataset
from transformers import set_seed
from sentence_transformers import SentenceTransformer

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)
set_seed(42)

# Important parameters - size of evaluation data, loading chunksize of dataset, counter parameter, trained embedding model path, 
# initial dataset containing the XML, the questions and the correct corresponding chunks

evaluate_range = 1000
chunk_size = 50
evaluated_lines = 0
model_path = "F:/VSprograms/models/trained-embedding"
second_dataset_path = "F:/VSprograms/XML_testing_dataset.jsonl"
model = SentenceTransformer(model_path, device='cuda')
model.max_seq_length = 512
results= []

def read_jsonl_in_chunks(file, chunk_size):
    current_chunk = []
    for line in file:
        current_chunk.append(json.loads(line))
        if len(current_chunk) == chunk_size:
            yield current_chunk
            current_chunk = []
    if current_chunk:
        yield current_chunk

def chunk_text(text, lines_per_chunk = 5):
    lines = text.splitlines()
    return ["\n".join(lines[i:i + lines_per_chunk]) for i in range (0, len(lines), lines_per_chunk)]

def cdata_chunks(chunks, indices_list):
    filtered_chunks = [(i, chunk) for i, chunk in enumerate(chunks) if i not in indices_list[1:]] # All chunks that are not in the indices list
    cdata_indices = [original_idx for original_idx, chunk in filtered_chunks if "CDATA" in chunk] # All cdata chunks from the filtered chunk list
    if cdata_indices:
        return cdata_indices
    else:
        return None

def evaluate_chunks(question, chunks_xml, model, top_k=3):
    """
    Retrieve the top-k most similar chunks for the given question.

    Args:
        question (str): The query question.
        chunks_xml (list): List of document chunks (strings).
        model (object): The embedding model with an `.encode()` method.
        top_k (int): Number of top similar chunks to retrieve.

    Returns:
        list: Indices of the top-k most similar chunks.
    """
    # Embed chunks & documents
    question_embedding = model.encode(question, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunks_xml, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = cosine_similarity(
        question_embedding.cpu().numpy().reshape(1, -1),
        chunk_embeddings.cpu().numpy()
    )[0]

    # Get the indices of the top-k most similar chunks
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_k_indices

def calculate_accuracy(retrieved_indices, ground_truth_indices, top_k = 3):
    """
    Calculate accuracy metrics for retrieved chunks, accounting for cases where
    the number of retrieved or correct chunks is limited.

    Args:
        retrieved_indices (list): Indices of retrieved chunks.
        ground_truth_indices (list): Indices of the correct chunks.
        top_k (int): The number of chunks retrieved (e.g., 3).

    Returns:
        dict: Precision, Recall, and F1-Score.
    """
    # Convert lists to sets for easier comparison
    retrieved_set = set(retrieved_indices)
    ground_truth_set = set(ground_truth_indices)
    # Correct = number of chunks that match with the correct ones, Precision = number of correct chunks accounting for top_k and the correct data length
    correct = len(retrieved_set.intersection(ground_truth_set))
    precision = correct / min(len(ground_truth_set), top_k)

    return {
        "precision": precision,
        "correct": correct,
        "retrieved": len(retrieved_set),
        "relevant": len(ground_truth_set)
    }

def calculate_mean_metrics(results):
    """
    Calculate mean precision, recall, and F1 score from a list of results.

    Args:
        results (list of dict): Each dictionary contains "precision", "recall", and "f1_score".

    Returns:
        dict: Mean precision, recall, and F1 score.
    """
    total_precision = 0
    num_results = len(results)

    for result in results:
        res = result["accuracy"]
        total_precision += res["precision"]

    mean_precision = total_precision / num_results if num_results > 0 else 0
    print(f"The mean precision is: {mean_precision:.4f}")
    return

# Dataset reader
with tqdm(total=evaluate_range, desc="Processing entries", unit="lines") as progress_bar:
    with open(second_dataset_path, 'r', encoding='utf-8') as file:
        for chunk in read_jsonl_in_chunks(file, chunk_size):
            for entry in chunk:
                # Stop once evaluation range is reached
                if evaluated_lines >= evaluate_range:
                    break
                xml_data = entry.get("xml_data")
                qaci_pairs = entry.get("qaci_pairs", {}) #Question Answer Chunk_Index pairs
                if not xml_data:
                    print("problem") # Debug
                    continue
                chunks_xml = chunk_text(xml_data, lines_per_chunk=5) #Chunking

                # Loop through questions and qaci pairs
                for question, details in qaci_pairs.items():
                    eval_chunk_pos = []
                    correct_indices = []
                    unseen_bool = False
                    if not details or len(details) < 2: # len(details) should be at least 2 since one entry is always the section ID
                        continue
                    chunk_answer = details[0]  # The chunk content
                    chunk_indices = details[1]  # The chunk indices
                    content = []
                    e_check = chunk_answer.strip() # Check for empty data
                    if not e_check:
                        continue
                    for index in chunk_indices: # Builds lists of the correct chunk content and indices
                        if str(index).isdigit(): # section ID handling                       
                            eval_chunk_pos.append(chunks_xml[index])
                            correct_indices.append(index)
                    if eval_chunk_pos and correct_indices:
                        top_answer_indices = evaluate_chunks(question, chunks_xml, model, 3)
                        if top_answer_indices.size > 0:
                            accuracy = calculate_accuracy(top_answer_indices, chunk_indices, top_k=3) # Called accuracy but actually just metric dictionary
                            results.append({ # Dictionary with questions and "accuracy" dictionary that contains the precision
                                "question": question,
                                "accuracy": accuracy
                            })
                            evaluated_lines += 1 # Progress bar parameter
                            progress_bar.update(1)
calculate_mean_metrics(results) # Final metric calculation of precision




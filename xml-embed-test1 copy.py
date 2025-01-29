import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import json
import random
from datasets import load_dataset
from transformers import set_seed
from sentence_transformers import (
    SentenceTransformer
)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)
set_seed(42)

evaluate_range = 1000
chunk_size = 50
evaluated_lines = 0
model_path = "F:/VSprograms/models/trained-embedding"
#model_path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# dataset_path = "F:/VSprograms/embedding_testing_dataset.jsonl"
# dataset_path = "F:/VSprograms/embedding_testing_unseen_dataset.jsonl"
second_dataset_path = "F:/VSprograms/XML_testing_dataset.jsonl"
model = SentenceTransformer(model_path, device='cuda')
model.max_seq_length = 512
results= []
# ds = load_dataset('json', data_files=second_dataset_path, split='train')
# ds = ds.select(range(10000))

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
    filtered_chunks = [(i, chunk) for i, chunk in enumerate(chunks) if i not in indices_list[1:]]
    cdata_indices = [original_idx for original_idx, chunk in filtered_chunks if "CDATA" in chunk]
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
    # Encode the question and document chunks
    question_embedding = model.encode(question, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunks_xml, convert_to_tensor=True)

    # Compute cosine similarities between question and all chunks
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

    # Correctly retrieved chunks
    correct = len(retrieved_set.intersection(ground_truth_set))

    # Precision: Fraction of retrieved chunks that are relevant
    # Precision considers only the retrieved chunks
    precision = correct / min(len(ground_truth_set), top_k)

    # Recall: Fraction of relevant chunks that were retrieved
    # Recall considers the total number of relevant chunks
    # recall = correct / len(ground_truth_set)

    # F1-Score: Harmonic mean of precision and recall
    # f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        # "recall": recall,
        # "f1_score": f1_score,
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
    #total_recall = 0
    #total_f1_score = 0
    num_results = len(results)

    for result in results:
        res = result["accuracy"]
        total_precision += res["precision"]

    mean_precision = total_precision / num_results if num_results > 0 else 0
    #mean_recall = total_recall / num_results if num_results > 0 else 0
    #mean_f1_score = total_f1_score / num_results if num_results > 0 else 0

    print(f"The mean precision is: {mean_precision:.4f}")
    #print(f"The mean recall is: {mean_recall}")
    #print(f"The mean f1 score is: {mean_f1_score}")

    return


with tqdm(total=evaluate_range, desc="Processing entries", unit="lines") as progress_bar:
    with open(second_dataset_path, 'r', encoding='utf-8') as file:
        for chunk in read_jsonl_in_chunks(file, chunk_size):
            for entry in chunk:
                if evaluated_lines >= evaluate_range:
                    break
                xml_data = entry.get("xml_data")
                qaci_pairs = entry.get("qaci_pairs", {})
                if not xml_data:
                    print("problem")
                    continue
                chunks_xml = chunk_text(xml_data, lines_per_chunk=5)

                # Process each question-chunk pair dynamically
                for question, details in qaci_pairs.items():
                    eval_chunk_pos = []
                    correct_indices = []
                    unseen_bool = False
                    if not details or len(details) < 2:
                        continue
                    chunk_answer = details[0]  # The chunk content
                    chunk_indices = details[1]  # The chunk indices
                    content = []
                    e_check = chunk_answer.strip()
                    if not e_check:
                        continue
                    for index in chunk_indices:
                        if str(index).isdigit():                        
                            eval_chunk_pos.append(chunks_xml[index])
                            correct_indices.append(index)
                    if eval_chunk_pos and correct_indices:
                        top_answer_indices = evaluate_chunks(question, chunks_xml, model, 3)
                        if top_answer_indices.size > 0:
                            accuracy = calculate_accuracy(top_answer_indices, chunk_indices, top_k=3)
                            results.append({
                                "question": question,
                                "accuracy": accuracy
                            })
                            evaluated_lines += 1
                            progress_bar.update(1)
calculate_mean_metrics(results)


# def preprocess_dataset(example):
#     return {
#         "anchor": example["question"],
#         "positive": example["positive_example"],
#         "negative": example["negative_example"]
#     }
# ds = ds.map(preprocess_dataset)
# ds = ds.remove_columns(['question', 'positive_example', 'negative_example'])
# model = SentenceTransformer(model_path, device='cuda')

# # Define a simple dataset class
# def create_triplet_dataset(dataset, model):
#     """
#     Convert the dataset into tensors of anchor, positive, and negative embeddings.
#     """
#     anchors = []
#     positives = []
#     negatives = []

#     for example in dataset:
#         anchor_embedding = model.encode(example["anchor"], convert_to_numpy=True)
#         positive_embedding = model.encode(example["positive"], convert_to_numpy=True)
#         negative_embedding = model.encode(example["negative"], convert_to_numpy=True)

#         anchors.append(anchor_embedding)
#         positives.append(positive_embedding)
#         negatives.append(negative_embedding)

#     return torch.tensor(anchors), torch.tensor(positives), torch.tensor(negatives)

# # Convert your dataset into triplet tensors
# anchors, positives, negatives = create_triplet_dataset(ds, model)

# # Create a DataLoader
# class TensorDataset(torch.utils.data.Dataset):
#     def __init__(self, anchors, positives, negatives):
#         self.anchors = anchors
#         self.positives = positives
#         self.negatives = negatives

#     def __len__(self):
#         return len(self.anchors)

#     def __getitem__(self, idx):
#         return self.anchors[idx], self.positives[idx], self.negatives[idx]

# # Create dataset and dataloader
# dataset = TensorDataset(anchors, positives, negatives)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# # Evaluation function
# # def evaluate_triplet_model(dataloader):
# #     correct = 0
# #     total = 0

# #     for batch in dataloader:
# #         anchors, positives, negatives = batch

# #         # Compute cosine similarities
# #         anchor_positive_sim = cosine_similarity(anchors, positives)
# #         anchor_negative_sim = cosine_similarity(anchors, negatives)

# #         # Check if positive similarity is greater than negative similarity
# #         correct += (anchor_positive_sim > anchor_negative_sim).sum()
# #         total += len(anchors)

# #     accuracy = correct / total
# #     return accuracy

# def evaluate_triplet_model(dataloader):
#     correct = 0
#     total = 0

#     for batch in dataloader:
#         anchors, positives, negatives = batch

#         # Normalize embeddings if not already normalized
#         anchors = anchors / anchors.norm(dim=1, keepdim=True)
#         positives = positives / positives.norm(dim=1, keepdim=True)
#         negatives = negatives / negatives.norm(dim=1, keepdim=True)

#         # Compute cosine similarities
#         anchor_positive_sim = cosine_similarity(anchors, positives)
#         anchor_negative_sim = cosine_similarity(anchors, negatives)

#         # Check if positive similarity is greater than negative similarity
#         correct += (anchor_positive_sim > anchor_negative_sim).sum().item()
#         total += anchors.size(0)  # Total samples in the batch

#     accuracy = correct / total
#     return accuracy


# # Run the evaluation
# print("Starting evaluation...")
# accuracy = evaluate_triplet_model(dataloader)
# print(f"Triplet Evaluation Accuracy: {accuracy:.2f}%")

# #actions = ["cdata", "random"]
#                 # probabilities = [0.5, 0.5]
#                 # chosen_action = random.choices(actions, probabilities)[0]
#                 # if chosen_action == actions[0]:
#                 #     random_cdata = cdata_chunks(chunks_xml, chunk_indices)
#                 #     if random_cdata:
#                 #         eval_chunk_neg = chunks_xml[random.choice(random_cdata)]
#                 #     else:
#                 #         continue
#                 # else:
#                 #     eval_chunk_neg = chunks_xml[random.choice(negative_indices)]
#                 # for index in chunk_indices:
#                 #     if str(index).isdigit():                        
#                 #         eval_chunk_pos = chunks_xml[index]
#                 #         break
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from transformers import set_seed
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Test 1 - to test the triplet dataset on whether the positive example has higher question similarity than the negative example after embedding

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)
set_seed(42)

# Important parameters - Evaluation size, model path, evaluation dataset path
datasize = 1000
model_path = "F:/VSprograms/models/trained-embedding" 
dataset_path = "F:/VSprograms/embedding_testing_unseen_dataset.jsonl"
ds = load_dataset('json', data_files=dataset_path, split='train')

def preprocess_dataset(example):
    return {
        "anchor": example["question"],
        "positive": example["positive_example"],
        "negative": example["negative_example"]
    }
ds = ds.select(range(datasize))
ds = ds.map(preprocess_dataset)
ds = ds.remove_columns(['question', 'positive_example', 'negative_example'])

model = SentenceTransformer(model_path, device='cuda')
# Limit with this model is 514
model.max_seq_length = 512

def create_triplet_dataset(dataset, model):
    """
    Convert the dataset into tensors of anchor, positive, and negative embeddings.
    """
    anchors = []
    positives = []
    negatives = []

    for example in dataset:
        anchor_embedding = model.encode(example["anchor"], convert_to_numpy=True)
        positive_embedding = model.encode(example["positive"], convert_to_numpy=True)
        negative_embedding = model.encode(example["negative"], convert_to_numpy=True)
        anchors.append(anchor_embedding)
        positives.append(positive_embedding)
        negatives.append(negative_embedding)

    return torch.tensor(anchors), torch.tensor(positives), torch.tensor(negatives)

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, anchors, positives, negatives):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        return self.anchors[idx], self.positives[idx], self.negatives[idx]

anchors, positives, negatives = create_triplet_dataset(ds, model)
dataset = TensorDataset(anchors, positives, negatives)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)



def evaluate_triplet_model(dataloader, model=None):
    correct = 0
    total = 0

    for batch in dataloader:
        anchors, positives, negatives = batch

        # Ensure all inputs are on the same device
        device = anchors.device
        positives, negatives = positives.to(device), negatives.to(device)

        # If embeddings need to be passed through a model
        if model:
            anchors = model(anchors)
            positives = model(positives)
            negatives = model(negatives)

        # Compute cosine similarities
        anchor_positive_sim = F.cosine_similarity(anchors, positives, dim=1)
        anchor_negative_sim = F.cosine_similarity(anchors, negatives, dim=1)

        # Check if positive similarity is greater than negative similarity
        correct += (anchor_positive_sim > anchor_negative_sim).sum().item()
        total += len(anchors)

    accuracy = correct / total if total > 0 else 0
    return accuracy

# Run the evaluation
print("Starting evaluation...")
accuracy = evaluate_triplet_model(dataloader)
print(f"Triplet Evaluation Accuracy: {accuracy:.4f}%")

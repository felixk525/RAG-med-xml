from transformers import AutoTokenizer, AutoModel, set_seed, AutoModelForCausalLM, pipeline, GPTQConfig, EarlyStoppingCallback
import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
import random
import json
import os
from sentence_transformers import losses, SentenceTransformer, models, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, SentenceTransformerModelCardData,
from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers
from datasets import load_dataset
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.models import Pooling

dataset_path = "F:/VSprograms/embedding_training_dataset.jsonl"
eval_ds_path = "F:/VSprograms/embedding_testing_dataset.jsonl"
model_save = "F:/VSprograms/models/trained-embedding"


# Ensure reproducibility across devices (CPU and GPU)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)
set_seed(42)
seed = 42

# Load embedding model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device='cuda')

# Load evaluation and training datasets
ds = load_dataset('json', data_files=dataset_path, split='train')
ds = ds.select(range(100000))
eval_ds = load_dataset('json', data_files=eval_ds_path, split='train')  # Example eval dataset
eval_ds = eval_ds.select(range(500))
model.max_seq_length = 512 # limit 514 because tokenizer
print(model)

def preprocess_dataset(example):
    # remapping for Loss etc.
    return {
        "anchor": example["question"],
        "positive": example["positive_example"],
        "negative": example["negative_example"]
    }

ds = ds.map(preprocess_dataset)
ds = ds.remove_columns(['question', 'positive_example', 'negative_example'])
eval_ds = eval_ds.map(preprocess_dataset)
eval_ds = eval_ds.remove_columns(['question', 'positive_example', 'negative_example'])

print("mapping complete")
print(f"Number of training examples: {len(ds)}")


loss = losses.TripletLoss(model=model)
args = SentenceTransformerTrainingArguments(
    output_dir="F:/VSprograms/models/trained-embedding-checkpoints",
    seed=seed,
    per_device_train_batch_size=16,
    num_train_epochs=1,
    fp16= True,
    warmup_ratio=0.01,
    save_strategy="steps", 
    save_steps=40,
    logging_strategy="steps", 
    logging_steps= 40,
    max_grad_norm= 1.0,
    eval_steps=40,
    eval_strategy="steps",
    save_on_each_node= True,
    save_total_limit=2,
    load_best_model_at_end= True,
    optim= "paged_adamw_8bit"
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=ds,
    eval_dataset=eval_ds,
    loss=loss,
)


print("Starting training...")
trainer.train()
print("Training complete.")

try:
    model.save_pretrained(model_save)# in case of failure
    print("model saved 1")
except Exception as e:
    print(e)

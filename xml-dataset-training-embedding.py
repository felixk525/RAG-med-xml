from transformers import AutoTokenizer, AutoModel, set_seed
import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
import random
import json
import os
from sentence_transformers import losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.models import Pooling
from transformers import AutoModelForCausalLM, pipeline, GPTQConfig, EarlyStoppingCallback

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.manual_seed(42)

# Ensure reproducibility across devices (CPU and GPU)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For multi-GPU
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(42)
set_seed(42)

# Load model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device='cuda')
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device='cuda')
dataset_path = "F:/VSprograms/embedding_training_dataset.jsonl"
ds = load_dataset('json', data_files=dataset_path, split='train')
ds = ds.select(range(100000))#0))


eval_ds_path = "F:/VSprograms/embedding_testing_dataset.jsonl"
eval_ds = load_dataset('json', data_files=eval_ds_path, split='train')  # Example eval dataset
eval_ds = eval_ds.select(range(500))

#pooling_layer = Pooling(model.get_sentence_embedding_dimension(), normalize_embeddings=True)
#model = SentenceTransformer(modules=[model, pooling_layer])
model.max_seq_length = 512 # limit 514 bc tokenizer
print(model)
# save is "F:/VSprograms/models/trained-embedding",
# embedding dimensions = 768
# pretrained backbone = XLMRobertaModel
# pooling strategy = mean pooling
# normalization
# learning rate = 5e-5
# batch_size = 64
# loss function = Triplet Loss
# margin for ranking !
# negative sampling strategy 
# number of epochs = 1
# optimizer = paged_adamw_8bit
# scheduler (default linear)
# gradient clipping 
# dropout rate
# max_How grad_norm = (1.0 per default)
# warmup ratio = 0.1
# train dataset = ...
# preprocessing = ...
# sequence length
# eval strategy "steps", save 500, log 100, eval 100, save limit 2
# mixed precision = fp16
# load_best_model at end
# weight decay = 0 default

def preprocess_dataset(example):
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

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,
    early_stopping_threshold=0.0001 #1
)


loss = losses.TripletLoss(model=model)
args = SentenceTransformerTrainingArguments(
    # Required parameters:
    output_dir="F:/VSprograms/models/trained-embedding-checkpoints",
    seed=42,
    # Training parameters:
    per_device_train_batch_size=16,
    num_train_epochs=1,
    fp16= True,
    warmup_ratio=0.01, #5  
    save_strategy="steps", 
    save_steps=40,#20, 
    logging_strategy="steps", 
    logging_steps= 40,#,20, 
    max_grad_norm= 1.0,
    eval_steps=40,#20,
   eval_strategy="steps",
   save_on_each_node= True, #due to errors
   #metric_for_best_model= "eval_loss",
   save_total_limit=2,
   #save_safetensors= False,
   load_best_model_at_end= True,
    optim= "paged_adamw_8bit"
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=ds,
    eval_dataset=eval_ds,
    #callbacks=[early_stopping_callback],
    loss=loss,
)

# dev_evaluator = TripletEvaluator(
#     anchors=test_dataset["question"],
#     positives=test_dataset["positive_example"],
#     negatives=test_dataset["negative_example"],
#     name="pre-test",
# )
# dev_evaluator(model)

print("Starting training...")
trainer.train()
print("Training complete.")

# test_evaluator = TripletEvaluator(
#     anchors=test_dataset["question"],
#     positives=test_dataset["positive_example"],
#     negatives=test_dataset["negative_example"],
#     name="after-test",
# )
# test_evaluator(model)

try:
    model.save_pretrained("F:/VSprograms/models/trained-embedding")#, safe_serialization=False)
    print("model saved 1")
except Exception as e:
    print(e)
    try:
        model.save("F:/VSprograms/models/paraphrase-multilingual/final")
        print("model saved 2")
    except Exception as e:
        print(e)













# Define contrastive loss function
# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
    
#     def forward(self, positive_similarity, negative_similarity):
#         # Contrastive loss formula
#         return torch.max(positive_similarity - negative_similarity + self.margin, torch.tensor(0.0))

# Example of training with data
# def train_model(dataset_path, batch_size=16, epochs=3):
#     optimizer = optim.Adam(model.parameters(), lr = 0.001)
#     contrastive_loss_fn = ContrastiveLoss()

#     for epoch in range(epochs):
#         total_loss = 0
#         num_pairs = 0

#         with open(dataset_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 pair = json.loads(line)
#                 question = pair['question']
#                 pos_example = pair['positive_example']
#                 neg_example = pair['negative_example']

#                 # Get embeddings for the question, positive, and negative examples
#                 que_embedding = model.encode(question)
#                 pos_embedding = model.encode(pos_example)
#                 neg_embedding = model.encode(neg_example)

#                 # Calculate cosine similarity for positive and negative pairs
#                 positive_similarity = nn.functional.cosine_similarity(que_embedding, pos_embedding)
#                 negative_similarity = nn.functional.cosine_similarity(que_embedding, neg_embedding)

#                 # Calculate contrastive loss
#                 loss = contrastive_loss_fn(positive_similarity, negative_similarity)
#                 total_loss += loss.item()
#                 num_pairs += 1

#                 # Backpropagation
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 if num_pairs % 1000 == 0:
#                     print(f"Epoch {epoch+1}, Processed {num_pairs} pairs, Loss: {total_loss/num_pairs:.4f}")

#         print(f"Epoch {epoch+1} finished, Average Loss: {total_loss/num_pairs:.4f}")

# Train on your dataset
#dataset_path = "D:/Bachelorarbeit/embedding_training_dataset.jsonl"
#train_model(dataset_path)
# {'loss': 0.0076, 'grad_norm': 0.0, 'learning_rate': 1.066704593941118e-07, 'epoch': 1.0} 
# {'train_runtime': 10905.0365, 'train_samples_per_second': 91.701, 'train_steps_per_second': 1.433, 'train_loss': 0.053297019004821776, 'epoch': 1.0}  
# model 1

# print(ds.column_names)
# print(test_dataset.column_names)
# for example in ds.select(range(3)):
#     print(example)
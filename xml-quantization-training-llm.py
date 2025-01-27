from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GPTQConfig, EarlyStoppingCallback
import torch
from torch.utils.data import DataLoader
import transformers
from evaluate import load
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# dataset_path = "F:/VSprograms/generation_training_dataset.jsonl"
# ds = load_dataset('json', data_files=dataset_path, split='train' )#streaming=True)
# ds = ds.select(range(100000))
# #print(f"10th processed entry: {ds[10]}")
# eval_dataset_path = "F:/VSprograms/generation_testing_unseen_dataset.jsonl"
# eval_ds = load_dataset('json', data_files=eval_dataset_path, split='train' )#streaming=True)
# eval_ds = eval_ds.select(range(100))
#print(f"10th processed entry: {eval_ds[10]}")
batch_size_var = 2
# [question, context2, context3, context1, answer]
# device = "cuda"
#  model_path = "TheBloke/falcon-7b-instruct-GPTQ"
tokenizer_model_path = "Qwen/Qwen2-1.5B-Instruct"
#model_path = "Qwen/Qwen2-1.5B-Instruct"
model_path = "Qwen/Qwen2-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path, use_fast = True)
gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer, use_exllama=False)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",max_memory={0: "11GiB", "cpu": "60GiB"}, quantization_config=gptq_config)
# # GPTQ for quantization gguf for quantization
# model.to("cpu")
model.save_pretrained("F:/VSprograms/models/Qwen/Qwen2-1.5B-Instruct-personal4")
#prompt = f"[INST] [/INST]"

# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype="auto",
#     max_memory={0: "12GiB", "cpu": "64GiB"},
#     # torch_dtype=torch.float,
#     device_map="auto",
#     #trust_remote_code = False,
#     #revision = "main"
# )#.to(device)
# # print(model)
# model.train()
# model.gradient_checkpointing_enable()
# model = prepare_model_for_kbit_training(model) # enable quantized training

# config = LoraConfig(
#     r = 4,
#     lora_alpha= 32,
#     target_modules= ["q_proj", "v_proj"],# ? potentially add k_proj and o_proj
#     lora_dropout= 0.05,
#     bias= "none",
#     task_type="CAUSAL_LM"
# ) # ????
# model = get_peft_model(model, config)
# #model.gradient_checkpointing_enable()
# model.print_trainable_parameters()
# #print(tokenizer.eos_token)
# model.config.use_cache = False # Only in fine tuning
# model.gradient_checkpointing_enable() # ??? Catches pytorch warning
# # def create_prompt(example):
# #     prompt = example["question"]
# #     return {"prompt": prompt, "answer": example["answer"]}
# # ds = ds.map(create_prompt)
# # ds = ds.remove_columns(['question', 'context1', 'context2', 'context3'])

# def preprocess_for_training(example):
#     tokenized = tokenizer(
#         example["question"],
#         padding="max_length",#True, #"longest"
#         truncation=True,
#         max_length=2048,#512,  # Adjust max length as needed
#         return_tensors="np",
#         text_target=example["answer"]
#     )
#     return {
#         "input_ids": tokenized["input_ids"],
#         "attention_mask": tokenized["attention_mask"],
#         "labels": tokenized["labels"]
#     }

# accuracy_metric = load("accuracy")
# # def compute_metrics(pred):
# #     logits, labels = pred  # Separate logits (model outputs) and true labels
# #     predictions = logits.argmax(axis=-1)  # Get the predicted token indices
# #     return accuracy_metric.compute(predictions=predictions, references=labels)

# bleu_metric = load("bleu")

# def compute_metrics(eval_preds):
#     """
#     Computes BLEU scores for model evaluation.
    
#     Args:
#         eval_preds: Tuple containing:
#             - predictions: Model predictions (logits or token IDs).
#             - references: Ground-truth token IDs (with padding tokens as -100).

#     Returns:
#         dict: BLEU score in a dictionary.
#     """
#     predictions, references = eval_preds
#     # print(f"Predictions: {predictions[:5]}")  # Log first 5 predictions
#     # print(f"References: {references[:5]}")
#     # !!!!!!!!!!!!! Problem that references are wrong!
#     # If predictions are logits, convert them to token IDs
#     if predictions.ndim > 2:  # [batch_size, seq_len, vocab_size]
#         predictions = torch.argmax(torch.tensor(predictions), dim=-1)
    
#     # Convert references to tensor if necessary
#     if isinstance(references, list):
#         references = torch.tensor(references)
    
#     # Remove padding tokens (-100)
#     predictions = predictions.tolist()  # Remove unnecessary .numpy()
#     references = references.tolist()
    
#     # Decode token IDs
#     decoded_predictions = [
#         tokenizer.decode([token for token in pred if token != -100], skip_special_tokens=True)
#         for pred in predictions
#     ]
#     decoded_references = [
#         [tokenizer.decode([token for token in ref if token != -100]) ] #skip_special_tokens=True)
#         for ref in references
#     ]
#     # log_interval = 5
#     # if len(decoded_predictions) % log_interval == 0:
#     #     print(f"Sample Predictions: {decoded_predictions[:3]}")
#     #     print(f"Sample References: {decoded_references[:3]}")
    
#     # Calculate BLEU
#     bleu_score = bleu_metric.compute(predictions=decoded_predictions, references=decoded_references)
    
#     return {"bleu_score": bleu_score["bleu"] * 100}


# # def preprocess_logits_for_metrics(logits, labels):
# #     if isinstance(logits, tuple):
# #         # Depending on the model and config, logits may contain extra tensors,
# #         # like past_key_values, but logits always come first
# #         logits = logits[0]
# #     # logits should be [bs, seq_len, hidden_size]
# #     return logits[:,0,:] # return CLS embedding

# def preprocess_logits_for_metrics(logits, labels):
#     if isinstance(logits, tuple):
#         # Depending on the model and config, logits may contain extra tensors,
#         # like past_key_values, but logits always come first
#         logits = logits[0]
#     # logits should be [bs, seq_len, hidden_size]
#     predictions = torch.argmax(logits, dim=-1)  # Get the token with the highest probability for each position
#     return predictions


# tokenizer.pad_token = tokenizer.eos_token # use own end of thing token

# ds = ds.map(preprocess_for_training, remove_columns=["question", "answer"], batched = True)
# #print(f"10th processed entry: {ds[10]}")
# eval_ds = eval_ds.map(preprocess_for_training, remove_columns=["question", "answer"], batched=True)

# #print(ds[0])  # Inspect a single dataset entry after preprocessing

# data_collector = transformers.DataCollatorForLanguageModeling(tokenizer)# (batched = True) try for dynamic padding, mlm= False)
# # dataloader = DataLoader(ds, batch_size=batch_size_var, pin_memory=True) # num_workers = 4
# # dataloader = DataLoader(eval_ds, batch_size=batch_size_var, pin_memory=True)

# args = TrainingArguments(
#    output_dir="F:/VSprograms/models/trained-generation-checkpoints",
#    per_device_train_batch_size=batch_size_var, 
#    per_device_eval_batch_size=batch_size_var,
#    gradient_checkpointing=True,
#    num_train_epochs=1,
#    #warmup_ratio=0.1,
#    save_strategy="steps",
#    logging_steps= 20,
#    eval_steps=20,
#    eval_strategy="steps",
#    #eval_accumulation_steps
#    eval_accumulation_steps=5,
#    save_total_limit=2,
#    # fp16 = True,
# #    weight_decay= 0.01,
#     load_best_model_at_end= True,
#    gradient_accumulation_steps=2,
#    metric_for_best_model="bleu_score",  # Metric to determine the best model
#     greater_is_better=True, #0 -> 1
#     optim= "paged_adamw_8bit"  #"adamw_torch"
#    )

# # dataloader = DataLoader(ds, batch_size= args.per_device_train_batch_size, pin_memory=True)
# early_stopping_callback = EarlyStoppingCallback(
#     early_stopping_patience=5,  # Number of evaluation steps without improvement
#     early_stopping_threshold=0.01  # Minimum improvement to count as progress
# )
# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=ds,
#     data_collator= data_collector,
#     eval_dataset= eval_ds,
#     callbacks=[early_stopping_callback],
#     compute_metrics=compute_metrics,
#     preprocess_logits_for_metrics=preprocess_logits_for_metrics
# )

# print("Starting training...")
# #from torch.profiler import profile, record_function, ProfilerActivity

# # with profile(activities=[
# #     ProfilerActivity.CPU, 
# #     ProfilerActivity.CUDA], record_shapes=True) as prof:
# trainer.train()
# # print(prof.key_averages().table(sort_by= "cuda_time_total"))
# print("Training complete.")
# try:
#     model.save_pretrained("F:/VSprograms/models/trained-generation-v1",)
#     print("model saved 1")
# except Exception as e:
#     print(e)
#     try:
#         model.save("F:/VSprograms/models/trained-generation-v1/final")
#         print("model saved 2")
#     except Exception as e:
#         print(e)
# # # Train the embedding model again!
# # # Also find model that supports germans
# # # Remember graph attempt / Add early stopping - sooner evaluation - evaluation dataset - best model saving
# # grad norm???
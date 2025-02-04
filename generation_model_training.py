import datasets
from datasets import load_dataset, Dataset, DatasetDict
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import torch
from torch.utils.data import DataLoader
import transformers
import evaluate
from evaluate import load
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import os
import torch
from transformers import (
    AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline, Trainer,
    HfArgumentParser, TrainingArguments, logging, GPTQConfig, EarlyStoppingCallback,
    DataCollatorWithPadding, AutoProcessor, PreTrainedTokenizerFast, set_seed,
    AutoModel, PreTrainedTokenizerBase
)
import warnings
import logging
from transformers.utils import logging as transformers_logging
import re
import random



os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42) 
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)
set_seed(42)

transformers_logging.enable_progress_bar()

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

alt_model_path = "F:/VSprograms/models/Qwen/Qwen2-1.5B-Instruct-personal4" # The quantized base model
processor_model_path = "Qwen/Qwen2-1.5B-Instruct" # The tokenizer / processor of the base model
new_model =  "F:/VSprograms/models/Qwen/custom_qwen_model_10_basic" # Save path for the trained model


checkpoints = new_model + "/misc"
optimizer = "paged_adamw_8bit"
ev_steps = 15
lg_steps = 15
sv_steps = 15
data_batch_size = 50
batch_size = 2
epochs = 1
grad_steps = 16
weight_decay = 0.001
max_seq_length = 2048
lr_scheduler_type = "cosine"
# QLoRA parameters
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode= False
)

model = AutoModelForCausalLM.from_pretrained(
    alt_model_path,
    device_map="auto",
)

model.train() # Change to training mode

model = prepare_model_for_kbit_training(model)

model = get_peft_model(model, peft_config)
model.config.use_cache = False 
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable() 
model.print_trainable_parameters()

processor = AutoProcessor.from_pretrained(processor_model_path, trust_remote_code=True)
dataset_path = "F:/VSprograms/generation_training_dataset.jsonl" # Training dataset
dataset = load_dataset('json', data_files=dataset_path, split='train' )
dataset = dataset.shuffle(seed=42).select(range(50000))

e2_dataset_path = "F:/VSprograms/generation_testing_dataset.jsonl" # Evaluation dataset
e2_dataset = load_dataset('json', data_files=e2_dataset_path, split='train' )
e2_dataset = e2_dataset.select(range(1000))
e2_dataset = e2_dataset.shuffle(seed=42).select(range(300))
eval_dataset = e2_dataset


def prepare_training_example(i):
    messages = [{"role": "system",
              "content": "Du bist ein Assistent der hilft medizinische Informationen die relevant für die Fragen des nutzers sind aus dem gegebenen Kontext auszulesen. Gib nur die relevanten Informationen an, ohne selber die Frage zu beantworten. Frage nicht nach mehr Kontext und gib die relevanten Informationen kurz an, ohne diese zu verändern und nur falls absolut nichts findbar ist meldest du es.\n"},
              {"role": "user",
              "content": "Die Frage ist: " + dataset[i]['instruction'] + "\n Der Kontext ist: \n " + dataset[i]['input'] + "\n Die Antwort auf die Frage ist: "}
              ]
    # Prompt for training. Uses the collumns instruction and input of the dataset
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    label = dataset[i]['output']

    # Create the training example: prompt + label
    input_text = prompt + label
    return {'input_text': input_text, 'label': label, 'context': (dataset[i]['instruction'] + "\n Der Kontext ist: \n " + dataset[i]['input'])}

# Transform the entire dataset
transformed_data = [prepare_training_example(i) for i in range(len(dataset))]
transformed_edata = [prepare_training_example(i) for i in range(len(eval_dataset))]
# Convert to a Hugging Face Dataset
hf_dataset = Dataset.from_dict({
    'input_text': [example['input_text'] for example in transformed_data],
    'label': [example['label'] for example in transformed_data],
    'context': [example['context'] for example in transformed_data]
})

hf_eval_dataset = Dataset.from_dict({
    'input_text': [example['input_text'] for example in transformed_edata],
    'label': [example['label'] for example in transformed_edata],
    'context': [example['context'] for example in transformed_edata]
})

dataset_dict = DatasetDict({
    'train': hf_dataset.shuffle().select(range(int(len(hf_dataset)))),
    'test': hf_eval_dataset.shuffle().select(range(int(len(hf_eval_dataset))))
})

def tokenize_function(examples):
    return processor(examples['input_text'], truncation=True, padding='max_length', max_length=2048)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
# input_text / label / context / input_ids / attention_mask

# Remove unnecessary columns
tokenized_datasets = tokenized_datasets.remove_columns(['input_text', 'label'])

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,
    early_stopping_threshold=0.0001
)

args = SFTConfig(
    dataset_text_field= "context",
    packing= True,
    optim= optimizer,
    max_seq_length= max_seq_length,
    dataset_batch_size= data_batch_size,
    learning_rate= 1e-5,
    output_dir= checkpoints,
    num_train_epochs= epochs,
    per_device_train_batch_size= batch_size,
    per_device_eval_batch_size= batch_size,
    gradient_accumulation_steps= grad_steps,
    save_steps= sv_steps,
    save_total_limit= 2,
    logging_steps= lg_steps,
    eval_steps= ev_steps,
    eval_strategy="steps",
    weight_decay= weight_decay,
    metric_for_best_model= "eval_loss",
    fp16=True,
    fp16=False,
    bf16=False,
    fp16_full_eval= True,
    seed=42,
    max_grad_norm= 0.1,
    warmup_ratio= 0.3,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    load_best_model_at_end= True,
    report_to="tensorboard" 
)

trainer = SFTTrainer(
    model= model,
    args= args,
    train_dataset= tokenized_datasets['train'],
    eval_dataset= tokenized_datasets['test'],
    processing_class= processor,
    callbacks=[early_stopping_callback],
    peft_config= peft_config,
)
    
trainer.train()
print("Training complete.")
model.save_pretrained(new_model)
print("Succesfull save.")

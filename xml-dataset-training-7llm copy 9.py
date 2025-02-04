
from datasets import load_dataset, Dataset, DatasetDict
from transformers import TrainingArguments, Trainer
from transformers import AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizerBase
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, pipeline, GPTQConfig, EarlyStoppingCallback
from transformers import DataCollatorWithPadding, AutoProcessor, PreTrainedTokenizerFast, set_seed
from trl import SFTConfig, SFTTrainer
import torch
from torch.utils.data import DataLoader
import transformers
from evaluate import load
import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    # update transformers? {'eval_loss': 1.0083656311035156, 'eval_runtime': 107.6928, 'eval_samples_per_second': 2.786, 'eval_steps_per_second': 1.393, 'epoch': 0.02}
    # {'eval_loss': 1.0901910066604614, 'eval_runtime': 103.1645, 'eval_samples_per_second': 2.908, 'eval_steps_per_second': 1.454, 'epoch': 0.02}


# 3%|█████▍                                                                                                                                                              | 250/7500 [37:51<10:03:20,  4.99s/it]Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
#Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
#{'eval_loss': 0.9857473969459534, 'eval_runtime': 111.4107, 'eval_samples_per_second': 2.693, 'eval_steps_per_second': 1.346, 'epoch': 0.03}
#   7%|██████████▊                                                                                                                                                       | 500/7500 [1:18:04<18:13:04,  9.37s/it]
)
import warnings
import logging
from transformers.utils import logging as transformers_logging
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import re
import random
import os
import datasets

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For multi-GPU
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)
set_seed(42)

#transformers_logging.set_verbosity_error()
transformers_logging.enable_progress_bar()

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# TODo

quantized_base = True
# for extraction model training
alt_model_path = "F:/VSprograms/models/Qwen/Qwen2-1.5B-Instruct-personal4"
processor_model_path = "Qwen/Qwen2-1.5B-Instruct"
#old_model = "F:/VSprograms/models/Qwen/Qwen2-7B-Instruct-personal"
new_model = "F:/VSprograms/models/Qwen/custom_qwen_model_e2_basic" #1 for previous prompt

#alt_model_path = "Qwen/Qwen2-7B-Instruct-GPTQ-Int4"
#processor_model_path = "Qwen/Qwen2-7B-Instruct-GPTQ-Int4"
#old_model = "F:/VSprograms/models/Qwen/Qwen2-7B-Instruct-personal"
#new_model = "F:/VSprograms/models/Qwen/Qwen2-7B-Instruct-personal-trained"
if not quantized_base:
    alt_model_path = processor_model_path
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
# Learning rate schedule
lr_scheduler_type = "cosine"
# QLoRA parameters
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",], #trainable params: 4,616,192 || all params: 238,077,440 || trainable%: 1.9389
    r=4,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode= False
)
bnb_4bit_quant_type = "nf4" 
bnb_4bit_compute_dtype = "float16"
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
# bnb_config = BitsAndBytesConfig( # unused
#     load_in_4bit=True,
#     bnb_4bit_quant_type=bnb_4bit_quant_type,
#     bnb_4bit_compute_dtype=compute_dtype,
#     #bnb_4bit_use_double_quant=False,
# )

packing = False

model = AutoModelForCausalLM.from_pretrained(
    alt_model_path,
    #quantization_config=bnb_config,
    device_map="auto",
)
#model.save_pretrained(old_model)
#print("Initial save success")
model.train()

# check why grad norm turned to infinity
# No inf checks were recorded for this optimizer. warning!!!
# NaN or Inf found in input tensor.
# Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
model = prepare_model_for_kbit_training(model)

model = get_peft_model(model, peft_config)
model.config.use_cache = False # ??
model.config.pretraining_tp = 1 # ??
model.gradient_checkpointing_enable() # deepen understanding! reentrant warning???
model.print_trainable_parameters()

processor = AutoProcessor.from_pretrained(processor_model_path, trust_remote_code=True)
dataset_path = "F:/VSprograms/extract_dataset.jsonl"
dataset = load_dataset('json', data_files=dataset_path, split='train' )
dataset = dataset.select(range(50000))#50000)) 10000 !! not 50k

e_dataset_path = "F:/VSprograms/extract_dataset.jsonl"
e_dataset = load_dataset('json', data_files=e_dataset_path, split='train' )
e_dataset = e_dataset.select(range(300))


#Du bist ein Assistent der hilft medizinische Informationen die relevant für die Fragen des nutzers sind aus dem gegebenen Kontext auszulesen. Gib nur die relevanten Informationen an, ohne selber die Frage zu beantworten. Frage nicht nach mehr Kontext und gib die relevanten Informationen kurz an, ohne diese zu verändern und nur falls absolut nichts findbar ist meldest du es.\n
def prepare_training_example(i):
    messages = [{"role": "system",
              "content": "Du bist ein Assistent der hilft HTML und XML in fließenden Text umzuformen und die medizinischen Informationen aus dem gegebenen Kontext auszulesen. Gib als Antwort nur die enthaltenen Informationen an, ohne selber Text hinzuzufügen. Frage nicht nach mehr Kontext und gib die relevanten Informationen kurz an.\n"},
              {"role": "user",
              "content": "Der Kontext ist: \n " + dataset[i]['input'] + "\n Der extrahierte Kontext ist: "}
              ]
    #prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    label = dataset[i]['output']

    # Create the training example: prompt + label
    input_text = prompt + label

    return {'input_text': input_text, 'label': label, 'context': ("Der Kontext ist: \n " + dataset[i]['input'])}

# Transform the entire dataset
transformed_data = [prepare_training_example(i) for i in range(len(dataset))]
transformed_edata = [prepare_training_example(i) for i in range(len(e_dataset))]
# Convert to a Hugging Face Dataset
hf_dataset = Dataset.from_dict({
    'input_text': [example['input_text'] for example in transformed_data],
    'label': [example['label'] for example in transformed_data],
    'context': [example['context'] for example in transformed_data]
})
# print(hf_dataset)
# print(hf_dataset[0])
hf_eval_dataset = Dataset.from_dict({
    'input_text': [example['input_text'] for example in transformed_edata],
    'label': [example['label'] for example in transformed_edata],
    'context': [example['context'] for example in transformed_edata]
})
# Split into train/test sets
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
    learning_rate= 1e-5, # New addition # half for final test
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
    seed=42,
    bf16=False,
    fp16_full_eval= True, # potentially change necessary
    max_grad_norm= 0.1,
    warmup_ratio=0.3,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    load_best_model_at_end= True,
    report_to="tensorboard" # visual representation of logs etc - see docs on how to view
)

trainer = SFTTrainer(
    model= model,
    args= args,
    #data_collator=
    train_dataset= tokenized_datasets['train'],#['train']
    eval_dataset= tokenized_datasets['test'],
    processing_class= processor,
    callbacks=[early_stopping_callback],
    peft_config= peft_config,
)
    
trainer.train()
print("Training complete.")
model.save_pretrained(new_model)
print("Succesfull save.")

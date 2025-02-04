
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
    #update transformers? {'eval_loss': 1.0882426500320435, 'eval_runtime': 114.3116, 'eval_samples_per_second': 2.624, 'eval_steps_per_second': 1.312, 'epoch': 0.02}  
    # 4%|██████▍                                                                                                                                                          | 500/12500 [1:19:15<31:42:21,  9.51s/it]
    #   2%|███▎                                                                                                                                                               | 250/12500 [38:18<17:37:00,  5.18s/it]
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

quantized_base = True

alt_model_path = "F:/VSprograms/models/Qwen/Qwen2-1.5B-Instruct-personal4"
processor_model_path = "Qwen/Qwen2-1.5B-Instruct"
#old_model = "F:/VSprograms/models/Qwen/Qwen2-7B-Instruct-personal"
new_model =  "F:/VSprograms/models/Qwen/custom_qwen_model_10_basic"#  9->10 9 is actual 10 is new experiment       "F:/VSprograms/models/Qwen/custom_qwen_model_5_basic" five was the attempt with larger batchsize#"F:/VSprograms/models/Qwen/custom_qwen_model_3_basic" #1 for previous prompt

#alt_model_path = "Qwen/Qwen2-7B-Instruct-GPTQ-Int4"
#processor_model_path = "Qwen/Qwen2-7B-Instruct-GPTQ-Int4"
#old_model = "F:/VSprograms/models/Qwen/Qwen2-7B-Instruct-personal"
#new_model = "F:/VSprograms/models/Qwen/Qwen2-7B-Instruct-personal-trained"
if not quantized_base:
    alt_model_path = processor_model_path
checkpoints = new_model + "/misc"
optimizer = "adamw_torch"# unapplied"paged_adamw_8bit"
# tr_steps = 5
ev_steps = 15#40#25
lg_steps = 15#40#10
sv_steps = 15#40#25
data_batch_size = 50
batch_size = 2
epochs = 1
grad_steps = 16# 32 # 8
weight_decay = 0.001
max_seq_length = 2048
# Learning rate schedule
lr_scheduler_type = "linear" #unapplied#"cosine"
# QLoRA parameters
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",], #trainable params: 4,616,192 || all params: 238,077,440 || trainable%: 1.9389
    r=8,
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
#25 hours runtime

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
dataset_path = "F:/VSprograms/generation_training_dataset.jsonl"
dataset = load_dataset('json', data_files=dataset_path, split='train' )
#dataset = dataset.select(range(50000))#50000)) 10000
dataset = dataset.shuffle(seed=42).select(range(50000))

# e_dataset_path = "F:/VSprograms/generation_testing_unseen_dataset.jsonl"
# e_dataset = load_dataset('json', data_files=e_dataset_path, split='train' )
# e_dataset = e_dataset.select(range(150))

e2_dataset_path = "F:/VSprograms/generation_testing_dataset.jsonl"
e2_dataset = load_dataset('json', data_files=e2_dataset_path, split='train' )
e2_dataset = e2_dataset.select(range(1000))
e2_dataset = e2_dataset.shuffle(seed=42).select(range(300))
eval_dataset = e2_dataset#datasets.concatenate_datasets([e_dataset, e2_dataset])


def prepare_training_example(i):
    messages = [{"role": "system",
              "content": "Du bist ein Assistent der hilft medizinische Informationen die relevant für die Fragen des nutzers sind aus dem gegebenen Kontext auszulesen. Gib nur die relevanten Informationen an, ohne selber die Frage zu beantworten. Frage nicht nach mehr Kontext und gib die relevanten Informationen kurz an, ohne diese zu verändern und nur falls absolut nichts findbar ist meldest du es.\n"},
              {"role": "user",
              "content": "Die Frage ist: " + dataset[i]['instruction'] + "\n Der Kontext ist: \n " + dataset[i]['input'] + "\n Die Antwort auf die Frage ist: "}
              ]
    #prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    early_stopping_patience=10,  # 10,
    early_stopping_threshold=0.0001
)

args = SFTConfig(
    dataset_text_field= "context",
    packing= True,
    optim= optimizer,
    max_seq_length= max_seq_length,
    dataset_batch_size= data_batch_size,
    learning_rate= 5e-5, # unapplied 5e-6, # New addition
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
    #fp16=True, # unapplied change
    fp16=False,
    bf16=False,
    fp16_full_eval= True,
    seed=42,
    max_grad_norm= 0.1,
    warmup_ratio= 0.05, # unapplied 0.1,#0.01,
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

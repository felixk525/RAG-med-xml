from datasets import load_dataset
import transformers
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, GPTQConfig, EarlyStoppingCallback
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from torch.utils.data import DataLoader
from evaluate import load
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Quantization of Qwen2-1.5B Instruct via GPTQ. - watch out for the max memory parameter

save_path = "F:/VSprograms/models/Qwen/Qwen2-1.5B-Instruct-personal4"
tokenizer_model_path = "Qwen/Qwen2-1.5B-Instruct"
model_path = "Qwen/Qwen2-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path, use_fast = True)
gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer, use_exllama=False)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",max_memory={0: "11GiB", "cpu": "60GiB"}, quantization_config=gptq_config)
model.save_pretrained(save_path)

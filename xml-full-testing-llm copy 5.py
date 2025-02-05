import datasets
from datasets import load_dataset, Dataset, DatasetDict
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import html
from torch.utils.data import DataLoader
import torch.nn.functional as F
import transformers
from tqdm import tqdm
import evaluate
from evaluate import load
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import random
import os
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser,
    GPTQConfig, EarlyStoppingCallback, DataCollatorWithPadding, AutoProcessor,
    PreTrainedTokenizerFast, set_seed, TrainingArguments, pipeline, logging,
    Trainer, AutoModel, PreTrainedTokenizerBase
)
import re
from xml.dom import minidom

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
random.seed(42)
set_seed(42)

path = "output_file.xml" # The path for the xml file to use for the RAG pipeline
top_k = 3

def preprocessing(file):
    css_pattern = re.compile(r'(BODY|TD|TH|P|DIV|UL|OL|BLOCKQUOTE|BUTTON|INPUT|SELECT|TEXTAREA|FONT|MARGIN|COLOR|BACKGROUND)[^}]*}', re.IGNORECASE)
    file = re.sub(css_pattern, "", file.strip())
    file = html.unescape(file)
    return file

def chunk_text(text, lines_per_chunk = 5):
    lines = text.splitlines()
    return ["\n".join(lines[i:i + lines_per_chunk]) for i in range (0, len(lines), lines_per_chunk)]

def get_outputs(model, inputs, max_new_tokens=200):
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1,
        eos_token_id=processor.eos_token_id,
        pad_token_id=processor.pad_token_id,
        **generation_kwargs
    )
    return outputs

alt_model_path = "F:/VSprograms/models/Qwen/custom_qwen_model_9_basic"
processor_model_path = "Qwen/Qwen2-1.5B-Instruct"
emb_model = SentenceTransformer("F:/VSprograms/models/trained-embedding")
emb_model.max_seq_length = 512 
xml = ""
with open(path, "r", encoding="utf-8") as data:
    xml = data.read()
    try:
        dom = minidom.parseString(xml)
        xml = dom.toprettyxml(indent= "  ")
        xml = preprocessing(xml)
    except Exception as e:
        print(e)
        print("Error due to XML parser")

chunks = chunk_text(xml, 5)
chunk_embeddings = emb_model.encode(chunks, convert_to_tensor=True)
question = input("Stell eine Frage: ")
question_embedding = emb_model.encode(question, convert_to_tensor=True)
similarities = cosine_similarity(
    question_embedding.cpu().numpy().reshape(1, -1),
    chunk_embeddings.cpu().numpy()
)[0]
top_k_indices = np.argsort(similarities)[-top_k:][::-1]
top_k_chunks = [chunks[idx] for idx in top_k_indices]
full_context = ""
for i, chunk in enumerate(top_k_chunks, start=1):
    full_context += f'Kontext {i}: "{chunk}"\n'


generation_kwargs = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "num_beams": 1,
    "early_stopping": False,
}

model = AutoModelForCausalLM.from_pretrained(
    alt_model_path,
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(processor_model_path, trust_remote_code=True)

messages = [{"role": "system",
        "content": "Du bist ein Assistent der hilft für Fragen relevanten Kontext zu finden um die Frage zu beantworten. Sollte nichts die Frage beantworten antwortest du enstprechend.\n"},
        {"role": "user",
        "content": "Die Frage ist: " + question + "\n Der Kontext ist: \n " + full_context + "\n Die Antwort auf die Frage ist: "}
        ]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_input = processor(prompt, return_tensors="pt").to('cuda')
foundational_outputs_sentence = get_outputs(model, model_input, max_new_tokens=600)
output = processor.batch_decode(foundational_outputs_sentence, skip_special_tokens=True)
if 'assistant' in output[0]:
    if len(output[0].split('assistant')) >= 2:
        res = output[0].split('assistant')[1]
        res = res.replace("\n", "").replace("'", "")
    else:
        res = ""
else:
    if 'Die Antwort auf die Frage ist:' in output[0]:
        if len(output[0].split('Die Antwort auf die Frage ist:')) >= 2:
            res = output[0].split('Die Antwort auf die Frage ist:')[1]
            res = res.replace("\n", "").replace("'", "")
    else:
        res = ""
res_c = res.strip()

print(f"Answer was {res_c}")
print("Generation phase finished")


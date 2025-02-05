import datasets
from datasets import load_dataset, Dataset, DatasetDict
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import torch
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
import warnings
import re

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
random.seed(42)
set_seed(42)

# File to evaluate the extraction model

datasize = 1000 # Number of datapoints used for evaluation

embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
embed_model2 = SentenceTransformer("F:/VSprograms/models/trained-embedding")
similarity_scores = []  # To store the similarity scores
similarity_scores2 = []

bleu_eval = evaluate.load("bleu")
rouge_eval = evaluate.load("rouge")

quantized_base = True
alt_model_path = "F:/VSprograms/models/Qwen/custom_qwen_model_e2_basic"
processor_model_path = "Qwen/Qwen2-1.5B-Instruct"
dataset_path = "F:/VSprograms/extract_dataset.jsonl"

generation_kwargs = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "num_beams": 1,
    "early_stopping": False,
}


device_map="auto"

model = AutoModelForCausalLM.from_pretrained(
    alt_model_path,
    torch_dtype="auto",
    device_map=device_map,
)

model.eval() # Put the model into evaluation mode

processor = AutoProcessor.from_pretrained(processor_model_path, trust_remote_code=True)

def get_outputs(model, inputs, max_new_tokens=600):
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


dataset = load_dataset('json', data_files=dataset_path, split='train' )
dataset = dataset.select(range(len(dataset) - datasize, len(dataset)))




bleu_scores = []
rouge_scores = []
skipped = 0
num_of_samples = datasize

for i in tqdm(range(num_of_samples), desc="Evaluating Samples"):

    messages = [{"role": "system",
              "content": "Du bist ein Assistent der hilft HTML und XML in fließenden Text umzuformen und die medizinischen Informationen aus dem gegebenen Kontext auszulesen. Gib als Antwort nur die enthaltenen Informationen an, ohne selber Text hinzuzufügen. Frage nicht nach mehr Kontext und gib die relevanten Informationen kurz an.\n"},
              {"role": "user",
              "content": "Der Kontext ist: \n " + dataset[i]['input'] + "\n Der extrahierte Kontext ist: "}
              ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_input = processor(prompt, return_tensors="pt").to('cuda')
    foundational_outputs_sentence = get_outputs(model, model_input, max_new_tokens=1000)
    output = processor.batch_decode(foundational_outputs_sentence, skip_special_tokens=True)
    label = dataset[i]["output"]
    # Extract model output
    if 'assistant' in output[0]:
        if len(output[0].split('assistant')) >= 2:
            res = output[0].split('assistant')[1]
            res = res.replace("\n", "").replace("'", "")
        else:
            res = ""
    else:
        if 'Der extrahierte Kontext ist:' in output[0]:
            if len(output[0].split('Der extrahierte Kontext ist:')) >= 2:
                res = output[0].split('Der extrahierte Kontext ist:')[1]
                res = res.replace("\n", "").replace("'", "")
        else:
            res = ""
    res_c = res.strip()
    label_c = label.strip()
    if not res_c or not label_c or len(res_c) <= 2 or len(label_c) <= 2:
        skipped += 1
        continue
    try:
        bleu_result = bleu_eval.compute(predictions=[res], references=[label]) 
        bleu_scores.append(bleu_result['bleu'])

        # Compute ROUGE score
        rouge_result = rouge_eval.compute(predictions=[res], references=[label])
        rouge_scores.append(rouge_result)

        res_embedding = embed_model.encode(res_c, convert_to_tensor=True)
        label_embedding = embed_model.encode(label_c, convert_to_tensor=True)
        similarity = cosine_similarity(res_embedding.cpu().numpy().reshape(1, -1), 
                                        label_embedding.cpu().numpy().reshape(1, -1))
        similarity_scores.append(similarity[0][0])
        res_embedding = embed_model2.encode(res_c, convert_to_tensor=True)
        label_embedding = embed_model2.encode(label_c, convert_to_tensor=True)
        similarity = cosine_similarity(res_embedding.cpu().numpy().reshape(1, -1), 
                                        label_embedding.cpu().numpy().reshape(1, -1))
        similarity_scores2.append(similarity[0][0])
    except ZeroDivisionError as e:
        print(f"Skipping due to an error: res='{res}', label='{label}'")
        skipped += 1
        continue

# Final results
avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
avg_rouge = {
    key: sum([score[key] for score in rouge_scores]) / len(rouge_scores)
    for key in rouge_scores[0].keys()
} if rouge_scores else {}

print(f"Skipped {skipped} values")
print("Final Evaluation Results:")
print(f"Average BLEU Score: {avg_bleu:.4f}")
print("Average ROUGE Scores:")
for key, value in avg_rouge.items():
    print(f"{key}: {value:.4f}")
print(f"Average Similarity Score: {np.mean(similarity_scores):.4f}")
print(f"Average trained Similarity Score: {np.mean(similarity_scores2):.4f}")

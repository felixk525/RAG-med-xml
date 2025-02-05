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

# This file is used to get the answer and optimal answer (and the original question) of one data entry as example
entry_x = 10 # The dataentry used for a single RAG example

# Load the embedding model
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
embed_model2 = SentenceTransformer("F:/VSprograms/models/trained-embedding")
eval_embed_model = embed_model2 # The model that should be used for evaluation

eval_embed_model.max_seq_length = 512 
similarity_scores = [] 
similarity_scores2 = []
bleu_eval = evaluate.load("bleu")
rouge_eval = evaluate.load("rouge")

quantized_base = True
alt_model_path = "F:/VSprograms/models/Qwen/custom_qwen_model_9_basic"
processor_model_path = "Qwen/Qwen2-1.5B-Instruct"
dataset_path = "F:/VSprograms/overall_testing_dataset.jsonl"

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

#model.eval()

processor = AutoProcessor.from_pretrained(processor_model_path, trust_remote_code=True)

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

dataset = load_dataset('json', data_files=dataset_path, split='train' )
entry = dataset[entry_x]
Correctly_done = 0


bleu_scores = []
rouge_scores = []
skipped = 0
chunk_size = 50
evaluated_lines = 0
results= []

def chunk_text(text, lines_per_chunk = 5):
    lines = text.splitlines()
    return ["\n".join(lines[i:i + lines_per_chunk]) for i in range (0, len(lines), lines_per_chunk)]

def evaluate_chunks(question, chunks_xml, model, top_k=3):
    """
    Retrieve the top-k most similar chunks for the given question.

    Args:
        question (str): The query question.
        chunks_xml (list): List of document chunks (strings).
        model (object): The embedding model with an `.encode()` method.
        top_k (int): Number of top similar chunks to retrieve.

    Returns:
        list: Indices of the top-k most similar chunks.
    """
    # Encode the question and document chunks
    question_embedding = model.encode(question, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunks_xml, convert_to_tensor=True)

    # Compute cosine similarities between question and all chunks
    similarities = cosine_similarity(
        question_embedding.cpu().numpy().reshape(1, -1),
        chunk_embeddings.cpu().numpy()
    )[0]

    # Get the indices of the top-k most similar chunks
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    top_k_chunks = [chunks_xml[idx] for idx in top_k_indices]

    return top_k_indices, top_k_chunks


xml_data = entry.get("xml")
question = entry.get("instruction")
answer = entry.get("output")
if not xml_data:
    print("problem")
chunks_xml = chunk_text(xml_data, lines_per_chunk=5)
chunk_answer = answer
e_check = chunk_answer.strip()
if not e_check:
    print("empty answer data")
top_answer_indices, top_answer_chunks = evaluate_chunks(question, chunks_xml, eval_embed_model, 3)
if top_answer_indices.size > 0:
    results.append({
        "question": question,
        "answer_chunks": top_answer_chunks,
        "optimal_answer": chunk_answer
    })
print("Embedding phase completed")
print("Starting genration phase")

for result in results:
    question = result["question"]
    context = result["answer_chunks"]
    answer = result["optimal_answer"]
    full_context = ""
    for i, chunk in enumerate(context, start=1):
        full_context += f'Kontext {i}: "{chunk}"\n'

    messages = [{"role": "system",
            "content": "Du bist ein Assistent der hilft für Fragen relevanten Kontext zu finden um die Frage zu beantworten. Sollte nichts die Frage beantworten antwortest du enstprechend.\n"},
            {"role": "user",
            "content": "Die Frage ist: " + question + "\n Der Kontext ist: \n " + full_context + "\n Die Antwort auf die Frage ist: "}
            ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_input = processor(prompt, return_tensors="pt").to('cuda')
    foundational_outputs_sentence = get_outputs(model, model_input, max_new_tokens=600)
    output = processor.batch_decode(foundational_outputs_sentence, skip_special_tokens=True)
    label = answer
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
    label_c = label.strip()

    if not res_c or not label_c or len(res_c) <= 2 or len(label_c) <= 2:
        skipped += 1
        continue
    try:
        print(f"Question was {question}")
        print(f"Answer was {res_c}")
        print(f"Best answer was {label_c}")
        bleu_result = bleu_eval.compute(predictions=[res], references=[label]) # [[label]]
        bleu_scores.append(bleu_result['bleu'])

        # Compute ROUGE score
        rouge_result = rouge_eval.compute(predictions=[res], references=[label])
        rouge_scores.append(rouge_result)

        res_embedding = F.normalize(embed_model.encode(res_c, convert_to_tensor=True), p=2, dim=-1)
        label_embedding = F.normalize(embed_model.encode(label_c, convert_to_tensor=True), p=2, dim=-1)
        similarity = torch.nn.functional.cosine_similarity(res_embedding, label_embedding, dim=-1)
        similarity_scores.append(similarity.item())

        res_embedding = F.normalize(embed_model2.encode(res_c, convert_to_tensor=True), p=2, dim=-1)
        label_embedding = F.normalize(embed_model2.encode(label_c, convert_to_tensor=True), p=2, dim=-1)

        similarity = torch.nn.functional.cosine_similarity(res_embedding, label_embedding, dim=-1)
        similarity_scores2.append(similarity.item())
    except ZeroDivisionError as e:
        print(f"Skipping due to an error: res='{res}', label='{label}'")
        skipped += 1
        continue


avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
avg_rouge = {
    key: sum([score[key] for score in rouge_scores]) / len(rouge_scores)
    for key in rouge_scores[0].keys()
} if rouge_scores else {}

print("Generation phase finished")
print(f"Skipped {skipped} values")
print("Final Generaton Evaluation Results:")
print(f"Average BLEU Score: {avg_bleu}")
print("Average ROUGE Scores:")
for key, value in avg_rouge.items():
    print(f"{key}: {value}")
print(f"Average Similarity Score: {np.mean(similarity_scores):.4f}")
print(f"Average Similarity Score: {np.mean(similarity_scores2):.4f}")

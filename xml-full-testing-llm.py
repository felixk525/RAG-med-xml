
from datasets import load_dataset, Dataset, DatasetDict
from transformers import TrainingArguments, Trainer
from transformers import AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizerBase
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, pipeline, GPTQConfig, EarlyStoppingCallback
from transformers import DataCollatorWithPadding, AutoProcessor, PreTrainedTokenizerFast, set_seed
from trl import SFTConfig, SFTTrainer
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import transformers
from evaluate import load
from tqdm import tqdm
import evaluate
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import random
import os
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    #update transformers?
)
import warnings
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import re
import os
torch.manual_seed(42)

# Ensure reproducibility across devices (CPU and GPU)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
random.seed(42)
set_seed(42)
# embedding dimensions, pretrained backbone, pooling strategy, normalization
# learning rate, batch_size, loss function, margin for ranking, negative sampling strategy, number of epochs, optimizer, scheduler, gradient clipping, dropout rate

#from sacrebleu import corpus_bleu
#from rouge_score import rouge_scorer
datasize = 1000# 1000
# number of samples tokens or documents
# preprocessing
# how is it divided?
# task type & metrics
# prompt design
# temperature, top-k & top-p, max tokens, stop sequences, batch size, evaluation speed

# fine tuning learning rate, optimizer, number of epochs, batch size, random seed, error handling, post processing, hyperparameters?
# max tokens beam search epochs weight decay gradient clipping, learn rate scheduler, dropout rate, warmup steps, evaluation frequency, early stopping, checkpointing

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the embedding model
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
embed_model2 = SentenceTransformer("F:/VSprograms/models/trained-embedding")
eval_embed_model = embed_model2 # The model that should be used for evaluation
#test_embedding_model = SentenceTransformer(embedding_model_path, device= "cuda")
#test_embedding_model.max_seq_length = 512
embed_model.max_seq_length = 512
similarity_scores = []  # To store the similarity scores
similarity_scores2 = []
bleu_eval = evaluate.load("bleu")
rouge_eval = evaluate.load("rouge")
# !!!!!!!!!accuracy.citation
# print(bleu_eval.features)
# print(rouge_eval.features)
#"Qwen/Qwen2-1.5B-Instruct"#
#"F:/VSprograms/models/Qwen/Qwen2-7B-Instruct-personal"
quantized_base = True
alt_model_path = "F:/VSprograms/models/Qwen/custom_qwen_model_9_basic"#"F:/VSprograms/models/Qwen/Qwen2-1.5B-Instruct-personal4"#"Qwen/Qwen2-7B-Instruct-GPTQ-Int4"#"F:/VSprograms/models/Qwen/Qwen2-1.5B-Instruct-personal4"
processor_model_path = "Qwen/Qwen2-1.5B-Instruct"#"Qwen/Qwen2-7B-Instruct-GPTQ-Int4"#"Qwen/Qwen2-1.5B-Instruct"
dataset_path = "F:/VSprograms/overall_testing_dataset.jsonl"
# if not quantized_base:
#     alt_model_path = processor_model_path
#new_model = "F:/VSprograms/models/Qwen/custom_qwen_model_1_basic"
#"F:/VSprograms/models/Qwen/Qwen2-1.5B-Instruct-personal"
generation_kwargs = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "num_beams": 1,
    "early_stopping": False,
}
# generation_kwargs = {
#     "do_sample": False,  # Disable sampling
#     "num_beams": 1,  # Greedy decoding
#     "temperature": None,  # Not needed for deterministic
#     "top_p": None,  # Not needed for deterministic
#     "top_k": None,  # Not needed for deterministic
#     "early_stopping": False,  # Not relevant for num_beams=1
# }

# QLoRA parameters
# peft_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.05,
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
#     r=4,
#     bias="none",
#     task_type="CAUSAL_LM",
#     inference_mode= False
# )
device_map="auto"

model = AutoModelForCausalLM.from_pretrained(
    alt_model_path,
    torch_dtype="auto",
    #quantization_config=bnb_config,
    device_map=device_map,
)
#model.save_pretrained("F:/VSprograms/models/Qwen/Qwen2-7B-Instruct-personal")
model.eval()
#model = prepare_model_for_kbit_training(model)

processor = AutoProcessor.from_pretrained(processor_model_path, trust_remote_code=True)

def get_outputs(model, inputs, max_new_tokens=200):
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1,
        #early_stopping=True, #Can stop before reach the max_length
        eos_token_id=processor.eos_token_id,
        pad_token_id=processor.pad_token_id,
        #do_sample=False, #potentially switch to keep static but high reliability
        **generation_kwargs
    )
    return outputs

dataset = load_dataset('json', data_files=dataset_path, split='train' )
dataset = dataset.select(range(datasize))
Correctly_done = 0


def prepare_training_example(i):
    qaci_pairs = dataset[i]['qaci_pairs']
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

def tokenize_function(examples):
    return processor(examples['input_text'], truncation=True, padding='max_length', max_length=2048)

# tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
# # input_text / label / context / input_ids / attention_mask

# # Remove unnecessary columns
# tokenized_datasets = tokenized_datasets.remove_columns(['input_text', 'label'])
bleu_scores = []
rouge_scores = []
num_of_samples = 10000000
skipped = 0
max_samples = len(dataset)
num_of_samples = min(num_of_samples, max_samples)
bleu_eval = evaluate.load("bleu")
rouge_eval = evaluate.load("rouge")
evaluate_range = datasize
chunk_size = 50
evaluated_lines = 0
results= []

def read_jsonl_in_chunks(file, chunk_size):
    current_chunk = []
    for line in file:
        current_chunk.append(json.loads(line))
        if len(current_chunk) == chunk_size:
            yield current_chunk
            current_chunk = []
    if current_chunk:
        yield current_chunk

def chunk_text(text, lines_per_chunk = 5):
    lines = text.splitlines()
    return ["\n".join(lines[i:i + lines_per_chunk]) for i in range (0, len(lines), lines_per_chunk)]

def cdata_chunks(chunks, indices_list):
    filtered_chunks = [(i, chunk) for i, chunk in enumerate(chunks) if i not in indices_list[1:]]
    cdata_indices = [original_idx for original_idx, chunk in filtered_chunks if "CDATA" in chunk]
    if cdata_indices:
        return cdata_indices
    else:
        return None

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

def calculate_accuracy(retrieved_indices, ground_truth_indices, top_k = 3):
    """
    Calculate accuracy metrics for retrieved chunks, accounting for cases where
    the number of retrieved or correct chunks is limited.

    Args:
        retrieved_indices (list): Indices of retrieved chunks.
        ground_truth_indices (list): Indices of the correct chunks.
        top_k (int): The number of chunks retrieved (e.g., 3).

    Returns:
        dict: Precision, Recall, and F1-Score.
    """
    # Convert lists to sets for easier comparison
    retrieved_set = set(retrieved_indices)
    ground_truth_set = set(ground_truth_indices)

    # Correctly retrieved chunks
    correct = len(retrieved_set.intersection(ground_truth_set))

    # Precision: Fraction of retrieved chunks that are relevant
    # Precision considers only the retrieved chunks
    precision = correct / min(len(ground_truth_set), top_k)

    # Recall: Fraction of relevant chunks that were retrieved
    # Recall considers the total number of relevant chunks
    # recall = correct / len(ground_truth_set)

    # F1-Score: Harmonic mean of precision and recall
    # f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        # "recall": recall,
        # "f1_score": f1_score,
        "correct": correct,
        "retrieved": len(retrieved_set),
        "relevant": len(ground_truth_set)
    }

def calculate_mean_metrics(results):
    """
    Calculate mean precision, recall, and F1 score from a list of results.

    Args:
        results (list of dict): Each dictionary contains "precision", "recall", and "f1_score".

    Returns:
        dict: Mean precision, recall, and F1 score.
    """
    total_precision = 0
    #total_recall = 0
    #total_f1_score = 0
    num_results = len(results)

    for result in results:
        res = result["embed_accuracy"]
        total_precision += res["precision"]

    mean_precision = total_precision / num_results if num_results > 0 else 0
    #mean_recall = total_recall / num_results if num_results > 0 else 0
    #mean_f1_score = total_f1_score / num_results if num_results > 0 else 0

    print(f"The mean precision is: {mean_precision}")
    #print(f"The mean recall is: {mean_recall}")
    #print(f"The mean f1 score is: {mean_f1_score}")

    return


with tqdm(total=evaluate_range, desc="Processing entries", unit="lines") as progress_bar:
    with open(dataset_path, 'r', encoding='utf-8') as file:
        for chunk in read_jsonl_in_chunks(file, chunk_size):
            for entry in chunk:
                if evaluated_lines >= evaluate_range:
                    break
                xml_data = entry.get("xml")
                question = entry.get("instruction")
                answer = entry.get("output")
                if not xml_data:
                    print("problem")
                    continue
                chunks_xml = chunk_text(xml_data, lines_per_chunk=5)
                chunk_answer = answer
                e_check = chunk_answer.strip()
                if not e_check:
                    continue
                top_answer_indices, top_answer_chunks = evaluate_chunks(question, chunks_xml, eval_embed_model, 3)
                if top_answer_indices.size > 0:
                    results.append({
                        "question": question,
                        "answer_chunks": top_answer_chunks,
                        "optimal_answer": chunk_answer
                    })
                    evaluated_lines += 1
                    progress_bar.update(1)
print("Embedding phase completed")
print("Starting genration phase")

evaluated_lines = 0
with tqdm(total=evaluate_range, desc="Processing entries", unit="lines") as progress_bar:
    for result in results:
        if evaluated_lines >= evaluate_range:
            break
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
            #print(f"Skipping sample {i + 1} due to empty prediction or reference.")
            skipped += 1
            continue
        try:
            bleu_result = bleu_eval.compute(predictions=[res], references=[label]) # [[label]]
            bleu_scores.append(bleu_result['bleu'])

            # Compute ROUGE score
            rouge_result = rouge_eval.compute(predictions=[res], references=[label])
            rouge_scores.append(rouge_result)

            res_embedding = F.normalize(embed_model.encode(res_c, convert_to_tensor=True), p=2, dim=-1)
            label_embedding = F.normalize(embed_model.encode(label_c, convert_to_tensor=True), p=2, dim=-1)
            similarity = torch.nn.functional.cosine_similarity(res_embedding, label_embedding, dim=-1)
            # similarity = cosine_similarity(res_embedding.cpu().numpy().reshape(1, -1), 
            #                                 label_embedding.cpu().numpy().reshape(1, -1))
            similarity_scores.append(similarity.item())

            res_embedding = F.normalize(embed_model2.encode(res_c, convert_to_tensor=True), p=2, dim=-1)
            label_embedding = F.normalize(embed_model2.encode(label_c, convert_to_tensor=True), p=2, dim=-1)

            # res_embedding = embed_model2.encode(res_c, convert_to_tensor=True)
            # label_embedding = embed_model2.encode(label_c, convert_to_tensor=True)
            similarity = torch.nn.functional.cosine_similarity(res_embedding, label_embedding, dim=-1)
            similarity_scores2.append(similarity.item())
        except ZeroDivisionError as e:
            print(f"Skipping due to an error: res='{res}', label='{label}'")
            skipped += 1
            continue
        evaluated_lines +=1
        progress_bar.update(1)


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
# # Final results
# avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
# avg_rouge = {
#     key: sum([score[key] for score in rouge_scores]) / len(rouge_scores)
#     for key in rouge_scores[0].keys()
# } if rouge_scores else {}

# print(f"Skipped {skipped} values")
# print("Final Evaluation Results:")
# print(f"Average BLEU Score: {avg_bleu}")
# print("Average ROUGE Scores:")
# for key, value in avg_rouge.items():
#     print(f"{key}: {value}")
# print(f"Average Similarity Score: {np.mean(similarity_scores):.4f}")
    #print(res)
    #print(dataset[i]['output'])

#     # Retrieve the target label
#     label = dataset_dict['test']['label'][i]

#     # Calculate BLEU score
#     bleu_score = corpus_bleu([res], [[label]]).score
#     bleu_scores.append(bleu_score)

#     # Calculate ROUGE-L score
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     rouge_scores = scorer.score(label, res)
#     rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)

#     # Debugging outputs for misclassified cases
#     if bleu_score == 0 or rouge_scores['rougeL'].fmeasure == 0:
#         print(f"Sample {i}: Model output: {res}, Target: {label}")

# # Calculate average BLEU and ROUGE-L scores
# avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
# avg_rouge_l_score = sum(rouge_l_scores) / len(rouge_l_scores)

# print(f"Average BLEU Score: {avg_bleu_score:.2f}")
# print(f"Average ROUGE-L Score: {avg_rouge_l_score:.2f}")

# trainer = SFTTrainer(
#     model= model,
#     #data_collator=
#     train_dataset= tokenized_datasets['train'],#['train']
#     eval_dataset= tokenized_datasets['test'],
#     processing_class= processor,
#     #compute_metrics= compute_metrics,
#     #optimizers= optimizer,
#     peft_config= peft_config,
# )
    

# # Train model
# trainer.predict(tokenized_datasets["test"])
# print("Testing complete.")


# from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu


# def bleu(ref, gen):
#     ''' 
#     calculate pair wise bleu score. uses nltk implementation
#     Args:
#         references : a list of reference sentences 
#         candidates : a list of candidate(generated) sentences
#     Returns:
#         bleu score(float)
#     '''
#     ref_bleu = []
#     gen_bleu = []
#     for l in gen:
#         gen_bleu.append(l.split())
#     for i,l in enumerate(ref):
#         ref_bleu.append([l.split()])
#     cc = SmoothingFunction()
#     score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
#     return score_bleu

#rouge scores for a reference/generated sentence pair
#source google seq2seq source code.

# import itertools

# #supporting function
# def _split_into_words(sentences):
#   """Splits multiple sentences into words and flattens the result"""
#   return list(itertools.chain(*[_.split(" ") for _ in sentences]))

# #supporting function
# def _get_word_ngrams(n, sentences):
#   """Calculates word n-grams for multiple sentences.
#   """
#   assert len(sentences) > 0
#   assert n > 0

#   words = _split_into_words(sentences)
#   return _get_ngrams(n, words)

# #supporting function
# def _get_ngrams(n, text):
#   """Calcualtes n-grams.
#   Args:
#     n: which n-grams to calculate
#     text: An array of tokens
#   Returns:
#     A set of n-grams
#   """
#   ngram_set = set()
#   text_length = len(text)
#   max_index_ngram_start = text_length - n
#   for i in range(max_index_ngram_start + 1):
#     ngram_set.add(tuple(text[i:i + n]))
#   return ngram_set

# def rouge_n(reference_sentences, evaluated_sentences, n=2):
#   """
#   Computes ROUGE-N of two text collections of sentences.
#   Source: http://research.microsoft.com/en-us/um/people/cyl/download/
#   papers/rouge-working-note-v1.3.1.pdf
#   Args:
#     evaluated_sentences: The sentences that have been picked by the summarizer
#     reference_sentences: The sentences from the referene set
#     n: Size of ngram.  Defaults to 2.
#   Returns:
#     recall rouge score(float)
#   Raises:
#     ValueError: raises exception if a param has len <= 0
#   """
#   if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
#     raise ValueError("Collections must contain at least 1 sentence.")

#   evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
#   reference_ngrams = _get_word_ngrams(n, reference_sentences)
#   reference_count = len(reference_ngrams)
#   evaluated_count = len(evaluated_ngrams)

#   # Gets the overlapping ngrams between evaluated and reference
#   overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
#   overlapping_count = len(overlapping_ngrams)

#   # Handle edge case. This isn't mathematically correct, but it's good enough
#   if evaluated_count == 0:
#     precision = 0.0
#   else:
#     precision = overlapping_count / evaluated_count

#   if reference_count == 0:
#     recall = 0.0
#   else:
#     recall = overlapping_count / reference_count

#   f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

#   #just returning recall count in rouge, useful for our purpose
#   return recall
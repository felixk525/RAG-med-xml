
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
from tqdm import tqdm
import evaluate
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
# pip install optimum, auto-gptq
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
datasize = 1000#1000
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
similarity_scores = []  # To store the similarity scores
similarity_scores2 = []
# Set the environment variable to suppress advisory warnings
# os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"
bleu_eval = evaluate.load("bleu")
rouge_eval = evaluate.load("rouge")
# !!!!!!!!!accuracy.citation
# print(bleu_eval.features)
# print(rouge_eval.features)
#"Qwen/Qwen2-1.5B-Instruct"#
#"F:/VSprograms/models/Qwen/Qwen2-7B-Instruct-personal"
quantized_base = True
alt_model_path = "F:/VSprograms/models/Qwen/custom_qwen_model_e2_basic"#"Qwen/Qwen2-7B-Instruct-GPTQ-Int4"#"F:/VSprograms/models/Qwen/Qwen2-1.5B-Instruct-personal4"
processor_model_path = "Qwen/Qwen2-1.5B-Instruct"#"Qwen/Qwen2-7B-Instruct-GPTQ-Int4"#"Qwen/Qwen2-1.5B-Instruct"
dataset_path = "F:/VSprograms/extract_dataset.jsonl"#"F:/VSprograms/generation_testing_dataset.jsonl"
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
# bnb_4bit_quant_type = "nf4" 
# bnb_4bit_compute_dtype = "float16"
# compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
# bnb_config = BitsAndBytesConfig( # unused
#     load_in_4bit=True,
#     bnb_4bit_quant_type=bnb_4bit_quant_type,
#     bnb_4bit_compute_dtype=compute_dtype,
#     #bnb_4bit_use_double_quant=False,
# )

# packing = False

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

#model = get_peft_model(model, peft_config)
#model.config.use_cache = False # ??
#model.config.pretraining_tp = 1 # ??
#model.gradient_checkpointing_enable() # deepen understanding! reentrant warning???
#model.print_trainable_parameters()

processor = AutoProcessor.from_pretrained(processor_model_path, trust_remote_code=True)

def get_outputs(model, inputs, max_new_tokens=600):
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

# dataset = load_dataset('json', data_files=dataset_path, split='train' )
# dataset = dataset.select(range(len(dataset) - datasize, len(dataset)))
dataset = load_dataset('json', data_files=dataset_path, split='train' )
dataset = dataset.select(range(len(dataset) - datasize, len(dataset)))
#50000)) # Last X entries
# ohne selber die Frage zu beantworten



bleu_scores = []
rouge_scores = []
num_of_samples = 10000000
skipped = 0
num_of_samples = datasize
bleu_eval = evaluate.load("bleu")
rouge_eval = evaluate.load("rouge")

for i in tqdm(range(num_of_samples), desc="Evaluating Samples"):

    messages = [{"role": "system",
              "content": "Du bist ein Assistent der hilft HTML und XML in fließenden Text umzuformen und die medizinischen Informationen aus dem gegebenen Kontext auszulesen. Gib als Antwort nur die enthaltenen Informationen an, ohne selber Text hinzuzufügen. Frage nicht nach mehr Kontext und gib die relevanten Informationen kurz an.\n"},
              {"role": "user",
              "content": "Der Kontext ist: \n " + dataset[i]['input'] + "\n Der extrahierte Kontext ist: "}
              ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Inference original model
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
        #print(f"Skipping sample {i + 1} due to empty prediction or reference.")
        skipped += 1
        continue
    try:
        bleu_result = bleu_eval.compute(predictions=[res], references=[label]) # [[label]]
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

    #print(f"Sample {i + 1}/{num_of_samples}")
    # print(f"Prediction: {res}")
    # print(f"Reference: {label}")
    # print(f"BLEU Score: {bleu_result['bleu']}")
    # print(f"ROUGE Scores: {rouge_result}\n")

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
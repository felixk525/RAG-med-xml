import ollama
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from sentence_transformers import losses, SentenceTransformer, InputExample
import json
import random

# Code for the full pipeline test dataset. Mostly directly uses the initial dataset

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
chunk_size = 50  # Number of rows to process at a time
train_data_path = "D:/Bachelorarbeit/XML_training_dataset.jsonl" # Initial dataset
output_file_path = "D:/Bachelorarbeit/overall_testing_dataset.jsonl" # Dataset to create
malformed = 0
malformed2 = 0
total_pairs_written = 0
lines_processed = 0
positive_negative_pairs = []
unseen_pair = 0
negative_pair = 0
normal_pair = 0
cdata_pair = 0
empty_catch = 0
chunks_processed = 0
# Tracking variables
with open(output_file_path, "w", encoding="utf-8") as f:
    pass
# Overwrite existing files

def chunk_text(text, lines_per_chunk=5):
    lines = text.splitlines()
    return ["\n".join(lines[i:i + lines_per_chunk]) for i in range(0, len(lines), lines_per_chunk)]

def read_jsonl_in_chunks(file, chunk_size):
    """Reads a JSONL file in chunks from an already opened file object."""
    current_chunk = []
    for line in file:
        current_chunk.append(json.loads(line))  # Parse each JSONL line
        if len(current_chunk) == chunk_size:
            yield current_chunk  # Yield the full chunk
            current_chunk = []
    if current_chunk:  # Yield the last chunk if it's not empty
        yield current_chunk

def cdata_chunks(chunks, indices_list):
    filtered_chunks = [(i, chunk) for i, chunk in enumerate(chunks) if i not in indices_list[1:]]
    cdata_indices = [original_idx for original_idx, chunk in filtered_chunks if "CDATA" in chunk]
    if cdata_indices:
        return cdata_indices
    else:
        return None


with open(output_file_path, 'a', encoding='utf-8') as output_file:
    with open(train_data_path, 'r', encoding='utf-8') as file:
        for chunk in read_jsonl_in_chunks(file, chunk_size):
            if chunks_processed > 100:
                break
            chunks_processed += 1
            for entry in chunk:
                lines_processed += 1
                if lines_processed % 1000 == 0:
                    print(f"Processed {lines_processed} JSON lines.")
                xml_data = entry.get("xml_data")
                qaci_pairs = entry.get("qaci_pairs", {})
                chunks_xml = chunk_text(xml_data, lines_per_chunk=5)

                # Process each question-chunk pair dynamically
                for question, details in qaci_pairs.items():
                    if not details or len(details) < 2:
                        malformed += 1
                        continue
                    
                    chunk_answer = details[0]  # The chunk content
                    chunk_indices = details[1]  # The chunk indices
                    # Negative example: Choose a random chunk not in chunk_indices
                    all_indices = set(range(len(chunks_xml)))
                    negative_indices = list(all_indices - set(chunk_indices))
                    content = []
                    counter = 0
                    empty_bool = False
                    if negative_indices:  # Ensure there are negative examples available
                        if len(negative_indices) < len(chunk_indices):
                            malformed2 += 1
                            continue
                        e_check = chunk_answer.strip()
                        if not e_check:
                            empty_bool = True
                            empty_catch += 1
                            continue

                    pair = {
                                "output": f'{chunk_answer}\n',
                                "instruction": f'{question}\n',
                                "xml"  : xml_data
                                

                            }
                    output_file.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    total_pairs_written += 1

print(f"Malformed entries without sufficient details: {malformed}")
print(f"Malformed entries with insufficient negative chunks: {malformed2}")
print(f"Total valid positive-negative pairs written: {total_pairs_written}")
print(f"Negative cases: {negative_pair}")
print(f"Cdata cases: {cdata_pair}")
print(f"Normal (random) cases: {normal_pair}")
print(f"Amount of empty Cdata cases: {empty_catch}")

import ollama
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from sentence_transformers import losses, SentenceTransformer, InputExample
import json
import random

# Used to create the embedding model training dataset

# model = 'paraphrase-multilingual'
chunk_size = 50  # Number of rows to process at a time
train_data_path = "D:/Bachelorarbeit/XML_training_dataset.jsonl"
output_file_path = "D:/Bachelorarbeit/embedding_training_dataset.jsonl"
malformed = 0
malformed2 = 0
total_pairs_written = 0
lines_processed = 0
positive_negative_pairs = []
unseen_pair, cdata_pair, normal_pair = 0, 0, 0

with open(output_file_path, "w", encoding="utf-8") as f:
    pass


def chunk_text(text, lines_per_chunk=5):
    lines = text.splitlines()
    return ["\n".join(lines[i:i + lines_per_chunk]) for i in range(0, len(lines), lines_per_chunk)]

def cdata_chunks(chunks, indices_list):
    filtered_chunks = [(i, chunk) for i, chunk in enumerate(chunks) if i not in indices_list[1:]]
    cdata_indices = [original_idx for original_idx, chunk in filtered_chunks if "CDATA" in chunk]
    if cdata_indices:
        return random.choice(cdata_indices)
    else:
        return None


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

# Example usage
with open(output_file_path, 'a', encoding='utf-8') as output_file:
    with open(train_data_path, 'r', encoding='utf-8') as file:
        for chunk in read_jsonl_in_chunks(file, chunk_size):
            for entry in chunk:
                lines_processed += 1
                if lines_processed % 1000 == 0:
                    print(f"Processed {lines_processed} JSON lines.")
                xml_data = entry.get("xml_data")
                qaci_pairs = entry.get("qaci_pairs", {})
                chunks_xml = chunk_text(xml_data, lines_per_chunk=5)

                # Process each question-chunk pair dynamically
                for question, details in qaci_pairs.items():
                    unseen_bool = False
                    positive_examples = []
                    negative_examples = []
                    if not details or len(details) < 2:
                        malformed += 1
                        continue
                    
                    chunk_answer = details[0]  # The chunk content
                    chunk_indices = details[1]  # The chunk indices

                    # Negative example: Choose a random chunk not in chunk_indices
                    all_indices = set(range(len(chunks_xml)))
                    negative_indices = list(all_indices - set(chunk_indices))

                    if negative_indices:  # Ensure there are negative examples available
                        if len(negative_indices) < len(chunk_indices):
                            malformed2 += 1
                            continue
                        for index in chunk_indices:
                            if str(index).isdigit():
                                if int(index) < len(chunks_xml):

                                    actions = ["Cdata_negative", "Normal"] # Sample 50/50 between CDATA and completely random
                                    probabilities = [0.5, 0.5]
                                    chosen_action = random.choices(actions, probabilities)[0]
                                    if chosen_action == actions[1]:
                                        if unseen_bool == False:
                                            positive_examples.append(chunks_xml[index])
                                            negative_examples.append(chunks_xml[random.choice(negative_indices)])
                                            normal_pair += 1
                                        else:
                                            unseen_pair +=1
                                    elif chosen_action == actions[0]:
                                        negative_cdata_index = cdata_chunks(chunks_xml, chunk_indices)
                                        if negative_cdata_index is not None:
                                            if unseen_bool == False:
                                                positive_examples.append(chunks_xml[index])
                                                negative_examples.append(chunks_xml[negative_cdata_index])
                                                cdata_pair += 1
                                            else:
                                                unseen_pair +=1

                            else:
                                if any(excluded_section in index for excluded_section in ["Vormedikation", "Therapie"]):
                                    unseen_bool = True


                    # Store the question, positive, and negative examples
                    if positive_examples:
                        for pos_example, neg_example in zip(positive_examples, negative_examples):
                            if unseen_bool == False:
                                pair = {
                                            "question": question,
                                            "positive_example": pos_example,
                                            "negative_example": neg_example
                                        }
                                output_file.write(json.dumps(pair, ensure_ascii=False) + "\n")
                                total_pairs_written += 1

print(f"Malformed entries without sufficient details: {malformed}")
print(f"Malformed entries with insufficient negative chunks: {malformed2}")
print(f"Total valid positive-negative pairs written: {total_pairs_written}")
print(f"Total valid unseen pairs written: {unseen_pair}")
print(f"Total JSON lines processed: {lines_processed}")
print(f"Total valid CDATA pairs written: {cdata_pair}")
print(f"Total valid normal pairs written: {normal_pair}")
# Control outputs

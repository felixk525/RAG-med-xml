import ollama
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from sentence_transformers import losses, SentenceTransformer, InputExample
import json
import random

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

chunk_size = 50  # Number of rows to process at a time
train_data_path = "D:/Bachelorarbeit/XML_training_dataset.jsonl"
output_file_path = "D:/Bachelorarbeit/generation_training_dataset.jsonl"
# train_data_path = "D:/Bachelorarbeit/XML_testing_dataset.jsonl"
# output_file_path = "D:/Bachelorarbeit/generation_testing_dataset.jsonl"
# unseen_file_path = "D:/Bachelorarbeit/generation_testing_unseen_dataset.jsonl"
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
with open(output_file_path, "w", encoding="utf-8") as f:
    pass

# with open(unseen_file_path, "w", encoding="utf-8") as f:
#     pass

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

# Example usage
with open(output_file_path, 'a', encoding='utf-8') as output_file:#, \
     #open(unseen_file_path, 'a', encoding='utf-8') as unseen_file:
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
                    # unseen_bool = False
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
                        if len(chunk_indices) <= 4: # 4 because section ID is included
                            actions = ["cdata_fill", "normal", "irrelevant"]
                            probabilities = [0.6, 0.3, 0.1]
                            chosen_action = random.choices(actions, probabilities)[0]
                            
                            if chosen_action == actions[2] or empty_bool:
                                negative_pair += 1
                                chunk_answer = "Leider konnte ich keine relevanten Informationen finden"
                                while counter < 3:
                                    content.append(chunks_xml[random.choice(negative_indices)])
                                    counter += 1
                            else:
                                for index in chunk_indices:
                                    if str(index).isdigit():
                                        if int(index) < len(chunks_xml):
                                            counter += 1
                                            content.append(chunks_xml[index]) # positive
                                if counter < 3:
                                    if chosen_action == actions[1]:
                                        normal_pair += 1
                                        while counter < 3:
                                            content.append(chunks_xml[random.choice(negative_indices)])
                                            counter += 1
                                    elif chosen_action == actions[0]:
                                        cdata_pair += 1
                                        negative_cdata_index = cdata_chunks(chunks_xml, chunk_indices)
                                        while counter < 3:
                                            if negative_cdata_index:
                                                choosen = random.choice(negative_cdata_index)
                                                content.append(chunks_xml[choosen])
                                                negative_cdata_index.remove(choosen)
                                            else:
                                                content.append(chunks_xml[random.choice(negative_indices)])
                                                malformed2 += 1
                                            counter += 1

                                    
                            # else:
                            #     if any(excluded_section in index for excluded_section in ["Vormedikation", "Therapie"]):
                            #         unseen_bool = True
                            #     else:
                            #         unseen_bool = False
                    # Store the question, positive, and negative examples
                    if len(content) == 3:
                        #for example in content:#zip(positive_examples, negative_examples):
# add stip handling to evade empty cdata
                        context = (
                            f'Kontext 1: "{content[0]}"\n'
                            f'Kontext 2: "{content[1]}"\n'
                            f'Kontext 3: "{content[2]}"\n'
                        )
                        prompt = (
                            f'Du bist ein Assistent der hilft medizinische Informationen die relevant für die Fragen des nutzers sind aus dem gegebenen Kontext auszulesen. Gib nur die relevanten Informationen an, ohne selber die Frage zu beantworten. Frage nicht nach mehr Kontext und gib die relevanten Informationen kurz an, ohne diese zu verändern und nur falls absolut nichts findbar ist meldest du es.\n'
                            f'-----\n'
                            f'Anfrage: {question}\n'
                            f'-----\n'
                            f'{context}'
                            f'-----\n'
                            f'Antwort: '
                        )
                        tokenized = tokenizer(prompt, return_tensors="pt")  # `return_tensors` is optional if you just want token count
                        num_tokens = len(tokenized.input_ids[0])
                        if num_tokens < 2048:

                            pair = {
                                        "output": f'{chunk_answer}\n',
                                        "input" : context,
                                        "instruction": f'{question}\n',
                                        "text"  : prompt
                                        

                                    }
                            output_file.write(json.dumps(pair, ensure_ascii=False) + "\n")
                            total_pairs_written += 1
                        else:
                            malformed += 1
                        # if unseen_bool == True:
                        #     unseen_file.write(json.dumps(pair, ensure_ascii=False) + "\n")
                        #     unseen_pair +=1
                        # else:
                        #     seen_pair +=1
print(f"Malformed entries without sufficient details: {malformed}")
print(f"Malformed entries with insufficient negative chunks: {malformed2}")
print(f"Total valid positive-negative pairs written: {total_pairs_written}")#, seen {seen_pair}")
# print(f"Total valid unseen pairs written: {unseen_pair}")
print(f"Total JSON lines processed: {lines_processed}")
print(f"Negative cases: {negative_pair}")
print(f"Cdata cases: {cdata_pair}")
print(f"Normal (random) cases: {normal_pair}")
print(f"Amount of empty Cdata cases: {empty_catch}")

# generation dataset
# Malformed entries without sufficient details: 160662
# Malformed entries with insufficient negative chunks: 139
# Total valid positive-negative pairs written: 864503
# Total JSON lines processed: 189183
# Negative cases: 102522
# Cdata cases: 600643
# Normal (random) cases: 299106
# Amount of empty Cdata cases: 30215
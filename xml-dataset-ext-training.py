import re
import html
import json
import pandas as pd
import xml.dom.minidom
import random
import transformers
from transformers import AutoTokenizer
# 1. replace tag name in attribute cases
# 2. remove closing tags without attributes
# 3. Capitalize opening tags and add :
# 4. Remove excessive whitespace
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
file_path = "D:/Bachelorarbeit/XML_training_dataset.jsonl"
chunk_factor = 500 # facor * 50 = entries processed
clength = 9
entry_index = 0 
failures2 = 0
failures3 = 0
output_file = 'D:/Bachelorarbeit/extract_train_test_dataset.jsonl'
output_file_long = 'D:/Bachelorarbeit/extract_train_test_dataset_long.jsonl'
malformed = 0
malformed2 = 0
empty_catch = 0
normal_pair = 0
cdata_pair = 0
total_pairs_written = 0
malformed3 = 0
lines_processed = 0
labor_a = 0
labor_b = 0
long_pairs_written = 0
# Processing finished. Failures: 5096
# Processing finished. Failures2: 0
# Processing finished. Failures3: 1446
# Processing finished. Written: 9780

with open(output_file, 'w', encoding='utf-8') as file:
    pass
with open(output_file_long, 'w', encoding='utf-8') as long_file:
    pass
def chunk_text(text, lines_per_chunk=5):
    lines = text.splitlines()
    return ["\n".join(lines[i:i + lines_per_chunk]) for i in range(0, len(lines), lines_per_chunk)]

def xml_chunk_to_text(xml_chunk):
    lines = xml_chunk.splitlines()
    attribute_trigger_list = ["name", "descriptor", "ID"]
    result = []
    for line in lines:
        
        line = line.strip()
        tag = ""
        closing_tag = re.match(r"</(\w+)>", line)
        self_closing_tag = re.match(r"<(\w+)(.*?)/>", line)
        opening_tag = re.match(r"<(\w+)(.*?)>", line)
        cdata = re.match(r"<!\[CDATA\[(.*?)\]\]>", line)
        if cdata:
            line = cdata.group(1)
            line = re.sub(r'<.*?>', ' ', line)
                #line = re.sub(r'<[^<>]*>$', ' ', line)
        elif closing_tag:
            line = ""
        elif self_closing_tag:
            tag = self_closing_tag.group(1) + ": "
            line = self_closing_tag.group(2).strip()
            # Remove the trailing `/` left by the self-closing tag
            line = re.sub(r"/$", "", line).strip()
            # Parse attributes to check for specific triggers in attribute names
            for attr in re.finditer(r'(\w+)=["\'](.*?)["\']', line):
                attr_name, attr_value = attr.groups()
                if attr_name in attribute_trigger_list:  # Check if attribute name matches
                    tag = attr_value + ": "  # Replace the tag with the value of the matched attribute
                    # Remove the matched attribute from the line
                    line = re.sub(rf'\b{attr_name}=["\'].*?["\']', '', line).strip()
                    break
        elif opening_tag:
            tag = opening_tag.group(1) + ": " # Capitalize
            line = opening_tag.group(2).strip()


        line = tag + line
        line = re.sub(r'"', '', line).strip()
        line = html.unescape(line)
        line = re.sub(r'\s+', ' ', line).strip()  # Reduce excessive whitespace
        if line:
            result.append(line)
    return "\n".join(result)

#Entferne ""
#Kapitalize
# def cdata_chunks(chunks):
#     cdata_indices = [original_idx for original_idx, chunk in chunks if "CDATA" in chunk]
#     if cdata_indices:
#         return cdata_indices
#     else:
#         return []
# def cdata_chunks(chunks):
#     return [chunk for chunk in chunks if "CDATA" in chunk]

def cdata_chunks(chunks, indices_list):
    filtered_chunks = [(i, chunk) for i, chunk in enumerate(chunks) if i not in indices_list[1:]]
    cdata_indices = [original_idx for original_idx, chunk in filtered_chunks if "CDATA" in chunk]
    if cdata_indices:
        return cdata_indices
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

with open(output_file, 'a', encoding='utf-8') as output_file, \
    open(output_file_long, 'w', encoding='utf-8') as long_file:
    with open(file_path, 'r', encoding='utf-8') as rfile:
        print("Processing started")
        chunk_size = 50
        xml_content = []
        extracted_content = []
        for chunk in read_jsonl_in_chunks(rfile, chunk_size):
            chunk_factor = chunk_factor - 1
            if chunk_factor < 0:
                break           
            for entry in chunk:
                lines_processed += 1
                if lines_processed % 1000 == 0:
                    print(f"Processed {lines_processed} JSON lines.")
                xml_str = entry.get("xml_data")
                qaci_pairs = entry.get("qaci_pairs", {})
                chunks_xml = chunk_text(xml_str, lines_per_chunk=5)
                for question, details in qaci_pairs.items():
                    long_content = False
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
                        if len(chunk_indices) <= (clength+1):
                            if len(chunk_indices) >= 5:
                                long_content = True
                            actions = ["cdata_fill", "normal"]
                            probabilities = [0.7, 0.3]
                            chosen_action = random.choices(actions, probabilities)[0]
                                    
                            for index in chunk_indices:
                                if str(index).isdigit():
                                    if int(index) < len(chunks_xml):
                                        counter += 1
                                        xml_fluent_chunk = xml_chunk_to_text(chunks_xml[index])
                                        content.append(xml_fluent_chunk) # positive
                                else:
                                    section_id = str(index)
                                    if section_id:
                                        if "Labor" in section_id:
                                            labor_b += 1
                            if counter < clength:
                                if chosen_action == actions[1]:
                                    normal_pair += 1
                                    while counter < clength:
                                        xml_fluent_chunk = xml_chunk_to_text(chunks_xml[random.choice(negative_indices)])
                                        content.append(xml_fluent_chunk)
                                        counter += 1
                                elif chosen_action == actions[0]:
                                    cdata_pair += 1
                                    negative_cdata_index = cdata_chunks(chunks_xml, chunk_indices)
                                    while counter < clength:
                                        if negative_cdata_index:
                                            choosen = random.choice(negative_cdata_index)
                                            xml_fluent_chunk = xml_chunk_to_text(chunks_xml[choosen])
                                            content.append(xml_fluent_chunk)
                                            negative_cdata_index.remove(choosen)
                                        else:
                                            xml_fluent_chunk = xml_chunk_to_text(chunks_xml[random.choice(negative_indices)])
                                            content.append(xml_fluent_chunk)
                                            failures2 += 1
                                        counter += 1

                    if len(content) == clength:
                        context = (
                            f'Kontext: {content[0]}\n'
                            f'{content[1]}\n'
                            f'{content[2]}\n'
                            f'{content[3]}\n'
                            f'{content[4]}\n'
                            f'{content[5]}\n'
                            f'{content[6]}\n'
                            f'{content[7]}\n'
                            f'{content[8]}\n'
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
                            if long_content:
                                long_file.write(json.dumps(pair, ensure_ascii=False) + "\n")
                                long_pairs_written += 1
                            total_pairs_written += 1
                        else:
                            if section_id:
                                if "Labor" in section_id:
                                    labor_a += 1
                                    labor_b -= 1
                            failures3 += 1
                    else:
                        malformed3 +=1

#print(xml_chunk_to_text(xml_input))
print("Processing finished. Not enough cdata: " + str(failures2))
print("Processing finished. Too long: " + str(failures3))
print(f"Malformed entries without sufficient details: {malformed}")
print(f"Malformed entries with insufficient negative chunks: {malformed2}")
print(f"Total valid pairs written: {total_pairs_written}")
print(f"Total long pairs written: {long_pairs_written}")
print(f"Cdata cases: {cdata_pair}")
print(f"Normal (random) cases: {normal_pair}")
print(f"Amount of empty Cdata cases: {empty_catch}")
print(f"Lines processed: {lines_processed}")
print(f"denied Labor {labor_a} accepted Labor {labor_b}")

# Processing finished. Not enough cdata: 18806
# Processing finished. Too long: 16950
# Malformed entries without sufficient details: 0
# Malformed entries with insufficient negative chunks: 0
# Total valid pairs written: 138616
# Total long pairs written: 16695
# Cdata cases: 107880
# Normal (random) cases: 46822
# Amount of empty Cdata cases: 4018
# Lines processed: 25000
# denied Labor 3360 accepted Labor 16148

# Processing finished. Not enough cdata: 18745
# Processing finished. Too long: 17088
# Malformed entries without sufficient details: 0
# Malformed entries with insufficient negative chunks: 0
# Total valid pairs written: 138478
# Cdata cases: 108135
# Normal (random) cases: 46567
# Amount of empty Cdata cases: 4018
# Lines processed: 25000



# Transform the malformed XML-like data
# result = parse_malformed_xml(malformed_xml)
# print(result)


# import ollama
# from transformers import AutoTokenizer, AutoModel
# from torch.utils.data import DataLoader
# from sentence_transformers import losses, SentenceTransformer, InputExample
# import json
# import random

# # model = 'paraphrase-multilingual'
# chunk_size = 50  # Number of rows to process at a time
# train_data_path = "D:/Bachelorarbeit/extract_dataset.json"
# output_file_path = "D:/Bachelorarbeit/extraction_training_dataset.jsonl"
# malformed = 0
# malformed2 = 0
# total_pairs_written = 0
# lines_processed = 0
# positive_negative_pairs = []
# unseen_pair, cdata_pair, normal_pair = 0, 0, 0
# # debug = 0
# with open(output_file_path, "w", encoding="utf-8") as f:
#     pass

# def chunk_text(text, lines_per_chunk=5):
#     lines = text.splitlines()
#     return ["\n".join(lines[i:i + lines_per_chunk]) for i in range(0, len(lines), lines_per_chunk)]

# def cdata_chunks(chunks, indices_list):
#     filtered_chunks = [(i, chunk) for i, chunk in enumerate(chunks) if i not in indices_list[1:]]
#     cdata_indices = [original_idx for original_idx, chunk in filtered_chunks if "CDATA" in chunk]
#     if cdata_indices:
#         return random.choice(cdata_indices)
#     else:
#         return None


# def read_jsonl_in_chunks(file, chunk_size):
#     current_chunk = []
#     for line in file:
#         current_chunk.append(json.loads(line))  # Parse each JSONL line
#         if len(current_chunk) == chunk_size:
#             yield current_chunk  # Yield the full chunk
#             current_chunk = []
#     if current_chunk:  # Yield the last chunk if it's not empty
#         yield current_chunk

# def cdata_chunks(chunks):
#     cdata_indices = [original_idx for original_idx, chunk in chunks if "CDATA" in chunk]
#     if cdata_indices:
#         return cdata_indices
#     else:
#         return None

# with open(output_file_path, 'a', encoding='utf-8') as output_file:
#     with open(train_data_path, 'r', encoding='utf-8') as file:
#         for chunk in read_jsonl_in_chunks(file, chunk_size):
#             for entry in chunk:
#                 lines_processed += 1
#                 if lines_processed % 1000 == 0:
#                     print(f"Processed {lines_processed} JSON lines.")
#                 xml_data = entry.get("xml_data")
#                 chunks_xml = chunk_text(xml_data, lines_per_chunk=5)
#                 xmltext_examples = []
#                 fluent_examples = []
#                 actions = ["Cdata_negative", "Normal"]
#                 probabilities = [0.5, 0.5]
#                 chosen_action = random.choices(actions, probabilities)[0]
#                 xmltext_examples.append(chunk)
#                 fluent_examples.append(chunk)
#                 #if chosen_action
#                 cdata_chunks(chunks_xml)
#                 malformed += 1
#                 for chunk in chunks_xml.items():
                    

#                     # Negative example: Choose a random chunk not in chunk_indices
#                     all_indices = set(range(len(chunks_xml)))



# print(f"Malformed entries without sufficient details: {malformed}")
# print(f"Malformed entries with insufficient negative chunks: {malformed2}")
# print(f"Total valid positive-negative pairs written: {total_pairs_written}")
# print(f"Total valid unseen pairs written: {unseen_pair}")
# print(f"Total JSON lines processed: {lines_processed}")
# print(f"Total valid CDATA pairs written: {cdata_pair}")
# print(f"Total valid normal pairs written: {normal_pair}")
import re
import html
import json
import pandas as pd
import xml.dom.minidom
import random
import transformers
from transformers import AutoTokenizer

# Dataset for the transformation of XML content to semi fluent content

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
file_path = 'D:/Bachelorarbeit/Arztbriefe_Export_WITH_XML.csv'
chunk_factor = 5000
# chunk_factor * 50 = parsed entries
entry_index = 0 
failures2 = 0
failures3 = 0
written = 0
output_xml = 'D:/Bachelorarbeit/extract_dataset.jsonl'
failures = 0

with open(output_xml, 'w', encoding='utf-8') as file:
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
            tag = opening_tag.group(1) + ": "
            line = opening_tag.group(2).strip()


        line = tag + line
        line = re.sub(r'"', '', line).strip()
        line = html.unescape(line)
        line = re.sub(r'\s+', ' ', line).strip()  # Reduce excessive whitespace
        if line:
            result.append(line)
    return "\n".join(result)


def cdata_chunks(chunks):
    return [chunk for chunk in chunks if "CDATA" in chunk]


with open(output_xml, 'a', encoding='utf-8') as file:
    print("Processing started")
    chunk_size = 50
    chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size)
    xml_content = []
    extracted_content = []
    for i in range(0,chunk_factor):
        first_chunk = next(chunk_iterator)
        if (i + 1) % 50 == 0:
            print("Chunk " + str(i+1))
        for a in range(0, chunk_size):
            first_row_value = first_chunk.iloc[a].values[0]  # Extract only the first value (ignoring column names)
            xml_str = f"{first_row_value}"
            try:
                xml_dom = xml.dom.minidom.parseString(xml_str)
                pretty_xml = xml_dom.toprettyxml(indent="  ")
                css_pattern = re.compile(r'(BODY|TD|TH|P|DIV|UL|OL|BLOCKQUOTE|BUTTON|INPUT|SELECT|TEXTAREA|FONT|MARGIN|COLOR|BACKGROUND)[^}]*}', re.IGNORECASE)
                pretty_xml = re.sub(css_pattern, '', pretty_xml.strip())
                chunks_xml = chunk_text(pretty_xml, lines_per_chunk=5)
                actions = ["Cdata_negative", "Normal"]
                probabilities = [0.5, 0.5]
                chosen_action = random.choices(actions, probabilities)[0]
                if chosen_action == actions[0]:
                    chunks_xml = cdata_chunks(chunks_xml)
                    if chunks_xml:
                        xml_chunk = random.choice(chunks_xml)
                    else:
                        xml_chunk = random.choice(chunks_xml)
                        failures2 += 1
                else:
                    xml_chunk = random.choice(chunks_xml)
                xml_str = xml_chunk_to_text(xml_chunk)
                if len(xml_content) !=3:
                        xml_content.append(xml_chunk)
                        extracted_content.append(xml_str)
                else: # Prompt for extraction
                    xml_prompt = (f'Du bist ein Assistent der hilft HTML und XML in fließenden Text umzuformen und die medizinischen Informationen aus dem gegebenen Kontext auszulesen. Gib als Antwort nur die enthaltenen Informationen an, ohne selber Text hinzuzufügen. Frage nicht nach mehr Kontext und gib die relevanten Informationen kurz an.\n'
                                f'-----\n'
                                f'Kontext 1: "{xml_content[0]}"\n'
                                f'Kontext 2: "{xml_content[1]}"\n'
                                f'Kontext 3: "{xml_content[2]}"\n'
                                f'-----\n'
                                f'Der extrahierte Kontext ist:'
                    )
                    tokenized = tokenizer(xml_prompt, return_tensors="pt")
                    num_tokens = len(tokenized.input_ids[0])
                    if num_tokens < 2048:
                        fluent_prompt = (
                                f'{extracted_content[0]}\n\n'
                                f'{extracted_content[1]}\n\n'
                                f'{extracted_content[2]}\n\n'
                                 
                        )
                        context = (f'Kontext 1: "{xml_content[0]}"\n'
                                f'Kontext 2: "{xml_content[1]}"\n'
                                f'Kontext 3: "{xml_content[2]}"\n')
                            
                        json.dump({
                            "input": context, 
                            "output": fluent_prompt
                        }, file, ensure_ascii=False)
                        file.write("\n")
                        written += 1
                    else:
                        failures3 += 1
                    xml_content = []
                    extracted_content = []
                    
            except Exception as e:
                    failures += 1
                    continue

print("Processing finished. Exceptions: " + str(failures))
print("Processing finished. Not enough cdata: " + str(failures2))
print("Processing finished. Too long: " + str(failures3))
print("Processing finished. Written: " + str(written))

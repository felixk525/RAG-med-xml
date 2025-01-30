# Define a dictionary mapping sections to lists of questions
section_questions = {
    "Diagnosis": [
        "Welche Diagnose wurde beim Patienten gestellt?",
        "Was war die medizinische Beurteilung des Zustands des Patienten?",
        "Welche Erkrankung oder welches Krankheitsbild liegt bei dem Patienten vor?",
        "Welche gesundheitlichen Probleme wurden diagnostiziert?"
    ],
    "Procedures": [
        "Welche medizinischen Verfahren wurden durchgeführt?",
        "Welche Maßnahmen oder Behandlungen hat der Patient erhalten?",
        "Was für Operationen oder Eingriffe wurden vorgenommen?",
        "Welche Verfahren kamen während der Behandlung zum Einsatz?"
    ],
    "Histologie": [
        "Welche Ergebnisse lieferte die histologische Untersuchung?",
        "Welche Erkenntnisse wurden aus der Gewebeprobe gewonnen?",
        "Was zeigen die histopathologischen Befunde?",
        "Was ergab die mikroskopische Analyse des Gewebes?"
    ],
    "Anamnese": [
        "Was sind Vorerkrankungen des Patienten?",
        "Welche Vorerkrankungen oder Beschwerden wurden erfasst?",
        "Welche relevanten medizinischen Daten wurden in der Anamnese dokumentiert?",
        "Was sagt die Anamnese?"
    ],
    "Vormedikation": [
        "Welche Medikamente hat der Patient vor der aktuellen Behandlung eingenommen?",
        "Welche Arzneimittel wurden vorab verordnet oder genutzt?",
        "Was für Medikamente sind dem Patienten momentan verschrieben?",
        "Welche Medikamente muss der Patient zu sich nehmen?"
    ],
    "Untersuchung": [
        "Welche Untersuchungen wurden beim Patienten durchgeführt?",
        "Was ergab die Untersuchung?",
        "Welche Ergebnisse lieferten die durchgeführten Untersuchungen?",
        "Welche Tests oder Untersuchungsmethoden kamen zum Einsatz?"
    ],
    "Konsile": [
        "Welche Konsile wurden für den Patienten angefordert?",
        "Welche Ratschläge haben Fachärzte gegeben?",
        "Was für Behandlungen wurden für den Patienten empfohlen?",
        "Welche Empfehlungen wurden im Rahmen der Konsile gegeben?"
    ],
    "Epikrise": [
        "Welche Schlussfolgerungen enthält die Epikrise?",
        "Was ist die Zusammenfassung der Behandlung?",
        "Welche Erkenntnisse wurden im Abschlussbericht dokumentiert?",
        "Was sagt der Abschlussbericht?"
    ],
    "EntlMedikation": [
        "Welche Medikamente wurden dem Patienten bei der Entlassung verordnet?",
        "Welche Arzneimittel wurden im Entlassungsplan aufgeführt?",
        "Was für Medikamente sind dem Patienten zum Schluss verschrieben worden?",
        "Welche Medikamente soll der Patient nach der Entlassung einnehmen?"
    ],
    "Therapie": [
        "Welche therapeutischen Maßnahmen wurden durchgeführt?",
        "Welche Behandlungen standen im Fokus der Therapie?",
        "Welche Verfahren oder Therapien wurden zur Behandlung eingesetzt?",
        "Welche Strategien wurden zur Verbesserung des Gesundheitszustands verfolgt?"
    ],
    "ZusatzUnt": [
        "Welche Zusatzuntersuchungen wurden durchgeführt?",
        "Gab es ergänzende diagnostische Maßnahmen?",
        "Welche weiteren Tests wurden zur Klärung der Diagnose veranlasst?",
        "Welche Zusatzuntersuchungen lieferten relevante Ergebnisse?"
    ],
    "Labor": [
        "Welche Laborwerte wurden ermittelt?",
        "Gab es auffällige Laborbefunde?",
        "Welche Erkenntnisse ergaben sich aus den Blutuntersuchungen?",
        "Welche Labordiagnostik wurde angewendet?"
    ],
    "Unfalldaten": [
        "Welche Informationen liegen zu dem Unfall vor?",
        "Was wird in den Unfalldaten dokumentiert?",
        "Welche Details zum Unfallhergang wurden festgehalten?",
        "Hatte der Patient einen Unfall?"
    ],
    "Bemerkungen": [
        "Welche zusätzlichen Anmerkungen wurden dokumentiert?",
        "Welche Bemerkungen ergänzen die Patientenakte?",
        "Gibt es wichtige Hinweise oder Kommentare in diesem Abschnitt?",
        "Wurden irgendwelche Bemerkungen gemacht?"
    ],
    "Empfehlung": [
        "Welche Empfehlungen wurden für die weitere Behandlung gegeben?",
        "Welche Ratschläge erhielt der Patient zur Genesung?",
        "Welche Maßnahmen wurden dem Patienten empfohlen?",
        "Welche ärztlichen Empfehlungen wurden ausgesprochen?"
    ],
    "Entlassstatus": [
        "Welcher Entlassstatus wurde für den Patienten dokumentiert?",
        "Wie wurde der Zustand des Patienten bei der Entlassung beschrieben?",
        "Gab es besondere Hinweise zum Entlassstatus?",
        "Welche Schlussbewertung wurde bei der Entlassung abgegeben?"
    ]
}

import re
from xml.dom import minidom
import pandas as pd
from collections import defaultdict
import random
import json
import html
#from lxml import etree

file_path = 'D:/Bachelorarbeit/Arztbriefe_Export_WITH_XML.csv'

def get_random_questions(section):
    """Get random questions for a section."""
    if section in section_questions:
        return section_questions[section]
    else:
        return ["What information can you extract from this section?"]

def process_cdata(cdata_text):
    """Process and clean CDATA content."""
    # Remove unwanted CSS/style blocks
    #css_pattern = re.compile(r'(BODY|TD|TH|P|DIV|UL|OL|BLOCKQUOTE|BUTTON|INPUT|SELECT|TEXTAREA|FONT|MARGIN|COLOR|BACKGROUND)[^}]*}', re.IGNORECASE)
    #cleaned_cdata = re.sub(css_pattern, '', cdata_text.strip())

    # Decode HTML entities
    #decoded_cdata = html.unescape(cleaned_cdata)

    # Replace tags with a single space and normalize whitespace
    stripped_content = re.sub(r'<[^>]*>', ' ', cdata_text)  # Replace tags with a space
    stripped_content = re.sub(r'\s+', ' ', stripped_content).strip()  # Normalize excessive whitespace

    return stripped_content

def chunking(file, chunk_size = 5):
    
    current_chunk = []
    chunk_list = []
    file = file.splitlines()

    for i, line in enumerate(file):
        current_chunk.append(line.strip())# Add the current line to the chunk, stripping whitespace
        if len(current_chunk) == chunk_size or i == len(file) - 1:  # Full chunk or document end
            combined_chunk = ' '.join(current_chunk)  # Combine the lines into a single string
            chunk_list.append(combined_chunk)
            current_chunk = []  # Reset the chunk
    return chunk_list

# 100 recommended
chunk_size = 100
chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size)
failures = 0
empty = 0
exclude_sections = True
output_file = "D:/Bachelorarbeit/XML_testing_dataset.jsonl" # Saving in a JSONL (JSON Lines) file

# Open the file for writing (if it already exists, it will be overwritten)
with open(output_file, "w", encoding="utf-8") as f:
    pass

# Process 1000 chunks for a total of 100000 files
num_chunks = 1000  
chunks_to_skip = 3000
with open(output_file, "a", encoding="utf-8") as f:
    # Write each entry as a JSON object on a new line
    for a in range(chunks_to_skip):
        try:
            next(chunk_iterator)  # Skip over the chunks you don't want to process
            if ((a + 1) % 200 == 0):
                print(f"{a+1} chunks skipped")
        except StopIteration:
            print("End of iterator reached before skipping required chunks.")
            break
    for chunk_idx in range(num_chunks):
        if (chunk_idx + 1) % 10 == 0:
            print(f"Chunk {chunk_idx + 1} from {num_chunks} chunks currently processing")

        try:
            chunk = next(chunk_iterator) # Get the next document chunk

            # Loop through each row/xml in the chunk
            for i in range(0, chunk_size):
                first_row_value = chunk.iloc[i].values[0]  # Get the xml of the row's first column
                xml_str = f"{first_row_value}"
                # Preprocessing steps
                css_pattern = re.compile(r'(BODY|TD|TH|P|DIV|UL|OL|BLOCKQUOTE|BUTTON|INPUT|SELECT|TEXTAREA|FONT|MARGIN|COLOR|BACKGROUND)[^}]*}', re.IGNORECASE)
                xml_str = re.sub(css_pattern, '', xml_str.strip())
                xml_str = html.unescape(xml_str)

                try:
                    # Parsing
                    dom = minidom.parseString(xml_str)
                    xml_str = dom.toprettyxml(indent="  ")
                    chunks_xml = chunking(xml_str)
                except Exception as e:
                    failures += 1
                    continue  # Skip this xml if parsing fails

                # Dictionary to store questions, answers, relevant chunk indexes for this XML
                qaci_pairs = {}

                # Find all xml sections
                sections = dom.getElementsByTagName("section")
                empty_bool = True
                for section in sections:
                    section_id = section.getAttribute("ID") # Section name

                    all_cdata_contents = [] # Collect all CDATA per section
                
                    paragraphs = section.getElementsByTagName("paragraph") # Find all <paragraph> elements within this section
                    chunk_indexes_per_paragraph = [section_id]

                    for paragraph in paragraphs:
                        
                        content_elements = paragraph.getElementsByTagName("content") # Find the <content> element within this paragraph
                        if content_elements:
                            content_element = content_elements[0]  # Assuming one <content> per paragraph

                            # Collect all CDATA
                            for node in content_element.childNodes:
                                
                                if node.nodeType == minidom.Node.CDATA_SECTION_NODE:
                                    for chunk_idx, chunk_content in enumerate(chunks_xml):
                                        # Find the chunk that has the CDATA ! Uses embedding chunk length
                                        #normalized_chunk = chunk.strip().replace('\r\n', '\n').replace('\r', '\n')
                                        if node.data in chunk_content:
                                            chunk_indexes_per_paragraph.append(chunk_idx)
                                            break
                                    processed_content = process_cdata(node.data)
                                    all_cdata_contents.append(processed_content)

                    # If the section has content, generate questions and answers
                    if all_cdata_contents:
                        # If content check whether the section ID is registered - do we have questions for it?
                        if any(key in section_id for key in section_questions.keys()):
                            empty_bool = False
                            matched_key = next(key for key in section_questions.keys() if key in section_id)
                            combined_content = "\n".join(all_cdata_contents)
                            
                            # Select one random question for the matched section
                            question = random.choice(section_questions[matched_key])
                            qaci_pairs[question] = [combined_content,chunk_indexes_per_paragraph]
                if empty_bool:
                    empty += 1 # Counter of unused documents because no question was found

                if qaci_pairs:
                    # Open the JSONL file and append the current chunk's data
                    json.dump({
                        "xml_data": xml_str, #add optimal embedding chunk (embedded & normal)
                        "qaci_pairs": qaci_pairs,
                    }, f, ensure_ascii=False)
                    f.write("\n")

        except StopIteration:
            # In case there are fewer than the assumed amount of chunks in the file
            print(f"Less than {num_chunks} chunks in the file, chunk {chunk_idx} not there, exiting.")
            break

print(f"Dataset created and saved to {output_file}")
print(f"{failures} failures, {empty} empty from {num_chunks * chunk_size} entries")
# 180 000 training entries
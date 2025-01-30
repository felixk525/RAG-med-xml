# import pandas as pd
# import xml.etree.ElementTree as ET

# # Load the Excel file
# path = 'D:\Bachelorarbeit\Arztbriefe_Export_WITH_XML.csv'
# import os

# def first_test(fpath):
    
#     file_path = fpath  # Replace with your file path
#     #print(os.path.exists(file_path))



#     # Process the file in smaller chunks
#     # chunk_size = 100  # Number of rows per chunk
#     # i = 0
#     # for chunk in pd.read_csv(file_path, chunksize=chunk_size, encoding="ISO-8859-1"):
#     # # Process each chunk here
#     #     if i == 0:
#     #         print(chunk.head())
#     #     i += 1
#     #     break

#     # with open(file_path, 'r') as file:
#     #     print(file.readline())
#     # file_path = "path/to/your/file.csv"

#     with open(file_path, 'r', encoding="utf-8-sig") as file:
#         current_entry = []
#         found_entry = False

#         for line in file:
#             if "<levelone>" in line:  # Detect the start of an XML entry
#                 found_entry = True

#             if found_entry:
#                 current_entry.append(line.strip())
#                 if "</levelone>" in line:  # Detect the end of the XML entry
#                     break  # Exit the loop once the entry is captured

#     # Combine the lines into a single XML string
#     xml_string = "\n".join(current_entry)
#     if xml_string.startswith('"<?xml'):
#         # Find the position of the first "<levelone>"
#         start_index = xml_string.find('<levelone>')
#         # Slice the string from the first "<levelone>"
#         xml_string = xml_string[start_index:]
#     # Remove a trailing quotation mark at the end
#     xml_string = xml_string.rstrip('"')

#     print(xml_string[-10:])  # Print the last 10 characters to confirm the fix

#     print(xml_string)
#     import re

#     # Remove BOM and leading/trailing whitespace
#     #xml_string = xml_string.lstrip("\ufeff").strip()

#     # Optionally remove other invalid characters (e.g., control characters)
#     #xml_string = re.sub(r"[^\x20-\x7E]", "", xml_string)  # Remove non-printable ASCII

#     if not xml_string.startswith("<levelone>") or not xml_string.endswith("</levelone>"):
#         print("Warning: XML string is incomplete or invalid.")
#         if not xml_string.startswith("<levelone>"):
#             print("error1")
#         if not xml_string.endswith("</levelone>"):
#             print("error2")
#             # Print the last 10 characters of the string
#             print(xml_string[-10:])

#     else:
#         print("XML string looks valid.")



#     try:
#         root = ET.fromstring(xml_string)
#         print("Root tag:", root.tag)
#         # Example: Iterate over children
#         for child in root:
#             print("Child tag:", child.tag, "Attributes:", child.attrib)
#     except ET.ParseError as e:
#         print("Error parsing XML:", e)


    # Print the first value to verify
    #first_value = df.iat[0]  # Access the value in the first row and first column
    #print("First value:", first_value)


#first_test(path)

#konvertieren zu xml! string zu xml

#faiss-cpu module
#//////////////////////////////////////////////
#pip install llama-index
#default gpt-3.5-turbo text gen / text-embedding-ada-002 retrieval & embeddings
# this needs an OPENAI_API_KEY set up as an environment variable

#local example
# pip install llama-index-core 
# pip llama-index-readers-file 
# pip llama-index-llms-ollama 
# pip llama-index-embeddings-huggingface
# pip install -r requirements.txt
# rust & cargo or faiss-cpu/faiss-gpu installation

from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
#3.10.0 not compatible - torch etc
# documents = SimpleDirectoryReader("data").load_data()

# # bge-base embedding model
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# # ollama
# Settings.llm = Ollama(model="llama3", request_timeout=360.0) #replace with huggingface

# index = VectorStoreIndex.from_documents(
#     documents,
# )

# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do growing up?")
# print(response)

# from llama_index.core import VectorStoreIndex
# from llama_index.core import SimpleDirectoryReader
# from llama_index.core import Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.huggingface import HuggingFaceLLM # corresponding pip install

# # Load documents
# documents = SimpleDirectoryReader("data").load_data()

# # Set up embedding model (bge-base embedding model)
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# # Replace Ollama with a HuggingFace LLM
# Settings.llm = HuggingFaceLLM(model_name="google/flan-t5-large", tokenizer_name="google/flan-t5-large")

# # Create the index
# index = VectorStoreIndex.from_documents(documents)

# # Set up the query engine
# query_engine = index.as_query_engine()

# # Query the model
# response = query_engine.query("What did the author do growing up?")
# print(response)

# from transformers import pipeline

# mxqas = pipeline("question-answering") #feature-extraction
# output = mxqas(
#     question="How does bulletproof glass work?",
#     context="You are a helpfull chatbot",
# )
# print(output)

# from transformers import pipeline

# # Initialize a text-generation pipeline with a general-purpose model
# qa_pipeline = pipeline("text-generation", model="google/flan-t5-base")

# # Ask a general knowledge question
# question = "How does bulletproof glass work?"

# # Generate an answer
# result = qa_pipeline(question,  num_return_sequences=1)

# # Print the answer
# print(f"Question: {question}")
# print(f"Answer: {result[0]['generated_text']}")

# from transformers import pipeline

# qa_model = pipeline("question-answering", "timpal0l/mdeberta-v3-base-squad2")
# question = "Where do I live?"
# context = "My name is Tim and I live in Sweden."
# qa_model(question = question, context = context)
# # {'score': 0.975547730922699, 'start': 28, 'end': 36, 'answer': ' Sweden.'}

# #embedding model

#ollama & rust & cargo are installed

dataset = []
with open('demo_simple_rag_py/cat-facts.txt', 'r', encoding='utf-8') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')

import ollama
#EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
#EMBEDDING_MODEL = 'bge-m3'

#LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

EMBEDDING_MODEL = 'paraphrase-multilingual'
LANGUAGE_MODEL = 'qwen2:1.5b-instruct'

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
VECTOR_DB = []

def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

chunk_size = 5  # Number of lines per chunk
current_chunk = []

for i, line in enumerate(dataset):
    current_chunk.append(line.strip())  # Add the current line to the chunk, stripping whitespace
    if len(current_chunk) == chunk_size or i == len(dataset) - 1:  # When the chunk is full or at the end of the dataset
        combined_chunk = ' '.join(current_chunk)  # Combine the lines into a single string
        add_chunk_to_database(combined_chunk)  # Add the combined chunk to the database
        current_chunk = []  # Reset the chunk
        print(f'Added chunk {i+1}/{len(dataset)} to the database')

# for i, chunk in enumerate(dataset):
#   add_chunk_to_database(chunk)
#   print(f'Added chunk {i+1}/{len(dataset)} to the database')

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
    # Generate the embedding for the query
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    # Temporary list to store (chunk, similarity) pairs
    similarities = []
    # Iterate through VECTOR_DB to calculate similarities
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    # Sort by similarity in descending order, because higher similarity means more relevant chunks
    similarities.sort(key=lambda x: x[1], reverse=True)
    # Finally, return the top N most relevant chunks
    return similarities[:top_n]
input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query)

print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
    print(f' - (similarity: {similarity:.2f}) {chunk}')

instruction_prompt = (
    f"You are a helpful chatbot.\n"
    "Use only the following pieces of context to answer the question. Don't make up any new information:\n"
    + '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])
)

stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': instruction_prompt},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)

# print the response from the chatbot in real-time
print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)








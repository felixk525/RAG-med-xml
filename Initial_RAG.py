from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
# check if ollama & rust & cargo are installed

# initial RAG somewhat adapted to the use-case but still in english

dataset = []
with open('demo_simple_rag_py/cat-facts.txt', 'r', encoding='utf-8') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')

import ollama

EMBEDDING_MODEL = 'paraphrase-multilingual'
LANGUAGE_MODEL = 'qwen2:1.5b-instruct'

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
VECTOR_DB = []

def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

chunk_size = 5  # Number of lines per chunk
current_chunk = []

for i, line in enumerate(dataset):
    current_chunk.append(line.strip())
    if len(current_chunk) == chunk_size or i == len(dataset) - 1:  # When the chunk is full or at the end of the dataset
        combined_chunk = ' '.join(current_chunk)  # Combine the lines into a single string
        add_chunk_to_database(combined_chunk) 
        current_chunk = [] 
        print(f'Added chunk {i+1}/{len(dataset)} to the database')

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
    # Generate the embedding for the query
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    # Sort by similarity in descending order, because higher similarity means more relevant chunks
    similarities.sort(key=lambda x: x[1], reverse=True)
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








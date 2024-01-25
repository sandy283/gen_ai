from langchain.document_loaders import DirectoryLoader
from langchain.docstore.document import Document
from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
directory = 'data'
loader = DirectoryLoader(directory)
documents = loader.load()
file_path = 'data/output.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()
sentences = [sentence.strip() for sentence in content.split('\n\n')]
embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en")
documents = [Document(page_content=sentence) for sentence in sentences]
pinecone.init(api_key="64a4e1b4-f374-40e2-90b4-d5902f0c74d1", environment="gcp-starter")
index_name = "youseai"
index = Pinecone.from_documents(documents, embeddings, index_name=index_name)


import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)
directory = 'data'
loader = DirectoryLoader(directory)
documents = loader.load()
file_path = 'data/output.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()
sentences = [sentence.strip() for sentence in content.split('\n\n')]
embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en")
documents = [Document(page_content=sentence) for sentence in sentences]
documents
vector_store = Chroma.from_documents(documents, embeddings, collection_metadata = {'hnsw:space': 'cosine'},persist_directory = 'chroma')
from fastapi import FastAPI, Request, Form
from flask import Flask, request, jsonify, render_template
import os
import json
from langchain import PromptTemplate, LLMChain
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

local_llm = "llama-2-7b.Q5_K_M.gguf"

config = {
    'max_new_tokens': 1024,
    'repetition_penalty': 1.1,
    'temperature': 0.1,
    'top_k': 50,
    'top_p': 0.9,
    'stream': True,
    'thread': int(os.cpu_count()/ 2)
}

llm = CTransformers(
    model = local_llm,
    model_type = 'llama',
    lib = 'avx2',
    **config
)
print("LLM initialized")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to  make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

prompt = PromptTemplate(template = prompt_template, input_variables = ['context', 'question'])
load_vector_store = Chroma(persist_directory = 'chroma', embedding_function = embeddings)
retriever = load_vector_store.as_retriever(search_kwargs = {"k": 3})
retriever

def get_response(query: str = Form(...)):
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
    response = qa(query)
    print(response)
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    response_data = {"answer": answer, "source_document": source_document, "doc": doc}

    return JSONResponse(content=jsonable_encoder(response_data))


# Example usage:
query_example = "What is the price of blooms flower ?"
response_example = get_response(query_example)
response_example
# reformat
import json
response_example = response_example.body.decode()
response_example = json.loads(response_example)
response_example.keys()
print("Answer: \t",response_example['answer'],"\n\n")
print("Answer: \t",response_example['source_document'],"\n\n")
print("Answer: \t",response_example['doc'],"\n\n")
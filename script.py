from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import timeit
import sys
load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY','64a4e1b4-f374-40e2-90b4-d5902f0c74d1')
PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV', 'gcp-starter')

def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents

extracted_data=load_pdf_file(data='data')

def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=text_split(extracted_data)
print("Length of Text Chunks", len(text_chunks))


def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='BAAI/bge-large-en')
    return embeddings

start = timeit.default_timer()
embeddings = download_hugging_face_embeddings()

query_result = embeddings.embed_query("Hello world")
print("Length", len(query_result))

pinecone.init(api_key='64a4e1b4-f374-40e2-90b4-d5902f0c74d1',
              environment='gcp-starter')

index_name="youseai"
docsearch=Pinecone.from_existing_index(index_name, embeddings)


query = "What are different types of flower avilable to you ?"


prompt_template="""
Use the following pieces of information to answer the question asked by the user.
If you don't know the answer, just say that you don't know.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}
llm=CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.3})

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={'k': 3}),return_source_documents=True, chain_type_kwargs=chain_type_kwargs)
while True:
    user_input=input(f"Input Prompt:")
    if user_input=='exit':
        print('Exiting')
        sys.exit()
    if user_input=='':
        continue
    result=qa({"query": user_input})
    print("Response : ", result["result"])
    #print("Source Documents : ", result["source_documents"])
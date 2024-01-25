from langchain.document_loaders import DirectoryLoader
directory = 'data'

def load_docs(directory):
    loader = DirectoryLoader(directory)
    document = loader.load()
    return document

document = load_docs(directory)

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(document, chunk_overlap = 20, chunk_size = 300):
    text_splitter = RecursiveCharacterTextSplitter(chunk_overlap = chunk_overlap, chunk_size = chunk_size)
    docs = text_splitter.split_documents(document)
    return docs

docs = split_docs(document)
print(len(docs))

from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name = "BAAI/bge-large-en")

import pinecone
from langchain.vectorstores import Pinecone
pinecone.init(
    api_key="64a4e1b4-f374-40e2-90b4-d5902f0c74d1",
    environment="gcp-starter"
)

index = "youseai"
index = Pinecone.from_documents(docs, embeddings, index_name = index)

def get_similar_docs(query, k=3, score = False):
    if score:
        similar_docs = index.similarity_search_with_score(query,k)
    else:
        similar_docs = index.similarity_search(query, k)
    return similar_docs

query = "What is flower colour ?"
similar_docs = get_similar_docs(query)
similar_docs


from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *

st.title("Langchain chatbot")
if 'responses' not in st.session_state:
    st.session_state['response'] = ["How can i assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name= "gpt-3.5-turbo",
                 openai_api_key='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')

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory = ConversationBufferMemory(
                 k=3, return_messages=True
            )
system_msg_template = SystemMessagePromptTemplate.from_template(template="""
Answer the question given by the customer using the provided context,
and if the answer is not contained within the text below, say 'I am sorry but i don't know about it'""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template  = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)
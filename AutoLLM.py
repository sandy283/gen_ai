from autollm import AutoQueryEngine
from autollm.utils.document_reading import read_github_repo_as_documents, read_files_as_documents
import gradio as gr
import os

#os.environ["OPENAI_API_KEY"]
git_repo_url = "https://github.com/langchain-ai/langchain.git"
relative_folder_path = "docs"
required_exts = [".md"]
documents = read_github_repo_as_documents(git_repo_url=git_repo_url, relative_folder_path=relative_folder_path, required_exts=required_exts)
documents
query_engine = AutoQueryEngine.from_defaults(documents = documents)

system_prompt = "You are an AI assistant that helps user find the most relevant and answer to the questions."
query_wrapper_prompt = """
The document information is below.
----------------------------------
{context_str}
----------------------------------
using the document information and mostly relying on it,
answer the query.
Query: {query_str}
Answer:
"""

enable_cost_calculator = True
model = "gpt-3.5-turbo"
vector_store_type = "LanceDBVectorStore"
uri = "tmp/lancedb"
table_name = "vectors"
chunk_size = 1024 
similarity_top_k = 5
llm_params = {"model": model}
vector_store_params = {"vector_store_type":vector_store_type, "uri": uri, "table_name":table_name}
service_context_params = {"chunk_size": chunk_size}
query_engine_params = {"similarity_top_k": similarity_top_k}
query_engine_new = AutoQueryEngine.from_defaults(documents=documents, system_prompt=system_prompt, query_wrapper_prompt=query_wrapper_prompt, enable_cost_calculator=enable_cost_calculator, llm_params = llm_params, vector_store_params = vector_store_params, service_context_params = service_context_params, query_engine_params = query_engine_params)
query_engine_new
response = query_engine_new.query("What do we mean by chain in langchain?")
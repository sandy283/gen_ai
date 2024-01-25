from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name = 'NeuML/pubmedbert-base-embeddings')
print(embeddings)

loader = DirectoryLoader('F:/gen_ai/meditron/data', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap = 500
)

texts = text_splitter.split_documents(documents)
texts
# start the docker and run using: docker run -p 6333:6333 qdrant/qdrant
url = "http://localhost:6333/dashboard"
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url = url,
    prefer_grpc = False,
    collection_name = "lung_cancer"
)

print("vector db created")
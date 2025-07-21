from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os

load_dotenv()
qdrant_key = os.getenv("QDRANT_KEY")
qdrant_url = os.getenv("QDRANT_URL")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 1000,
    chunk_overlap = 50,
    disallowed_special=()

)

pdf_loader = DirectoryLoader("",glob="**/*.pdf",loader_cls=PyMuPDFLoader)
docs = pdf_loader.load()
for doc in docs:
    doc.metadata['source'] = doc.metadata.get("source","unknown")
    doc.metadata['page'] = doc.metadata.get("page","unknown")
split_docs = text_splitter.split_documents(docs)
split_docs = [doc for doc in split_docs if len(doc.page_content.strip())>30]

vector_store = QdrantVectorStore.from_documents(
    split_docs,
    OpenAIEmbeddings(model="text-embedding-3-small"),
    url = qdrant_url,
    api_key = qdrant_key,
    batch_size = 31,
    collection_name= "Centanary_Celebrations"
)


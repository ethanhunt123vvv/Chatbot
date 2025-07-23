from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os,pinecone

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")  
index_name = "centenary-celebration"


pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

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

vector_store = PineconeVectorStore.from_documents(
    documents = split_docs,
    embeddings = embeddings,
    index_name = index_name
)


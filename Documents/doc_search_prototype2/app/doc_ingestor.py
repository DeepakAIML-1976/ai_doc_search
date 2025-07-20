import os
import pickle
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load .env file containing OPENAI_API_KEY
load_dotenv()

# Constants
SOURCE_DIR = "data/documents"
DB_FAISS_PATH = "data/vector_store"
INDEX_METADATA_PATH = "data/metadata.pkl"

def load_documents(source_dir):
    loaders = [
        (".txt", TextLoader),
        (".pdf", PyPDFLoader),
        (".docx", Docx2txtLoader),
    ]

    all_docs = []
    for ext, LoaderClass in loaders:
        loader = DirectoryLoader(source_dir, glob=f"**/*{ext}", loader_cls=LoaderClass, show_progress=True)
        docs = loader.load()
        all_docs.extend(docs)

    return all_docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def ingest_documents():
    print("Loading documents...")
    raw_documents = load_documents(SOURCE_DIR)

    print(f"Loaded {len(raw_documents)} documents. Splitting into chunks...")
    documents = split_documents(raw_documents)

    print("Creating embeddings and FAISS index...")
    embeddings = OpenAIEmbeddings()  # Uses OPENAI_API_KEY from env
    vectorstore = FAISS.from_documents(documents, embeddings)

    print(f"Saving FAISS index to {DB_FAISS_PATH}...")
    vectorstore.save_local(DB_FAISS_PATH)

    print(f"Saving metadata to {INDEX_METADATA_PATH}...")
    metadata = [doc.metadata for doc in documents]
    with open(INDEX_METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("Ingestion completed!")

if __name__ == "__main__":
    ingest_documents()

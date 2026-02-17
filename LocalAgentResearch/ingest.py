import os
import shutil
import stat
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
DATA_PATH = "./data"
DB_PATH = "./vectorstore"


def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def create_vector_db():
    # 1. Check if data folder exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created {DATA_PATH}. Please put .txt files there.")
        return

    # 2. Clear old database (optional, good for fresh experiments)
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH, onexc=remove_readonly)

    # 3. Load Documents
    print("Loading documents...")
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    if not documents:
        print("No documents found in ./data")
        return
    print(f"Loaded {len(documents)} document(s).")

    # 4. Split Text (Chunking)
    # Critical for RAG: We break text into smaller pieces so the LLM isn't overwhelmed.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # How many characters per chunk
        chunk_overlap=50  # Overlap to preserve context between chunks
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # 5. Create Embeddings & Store in Chroma
    # We use a standard, lightweight embedding model suitable for laptops.
    print("Creating embeddings (this may take a moment)...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # This line does the heavy lifting: Embeds -> Stores -> Persists
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    print(f"Database successfully created at {DB_PATH}")


if __name__ == "__main__":
    create_vector_db()

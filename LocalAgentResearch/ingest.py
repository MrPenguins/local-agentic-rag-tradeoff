import json
import os
import shutil
import stat
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
DATA_PATH = "./data"
DB_PATH = "./vectorstore"
DATASET_PATH = "./dataset/hotpot_dev_distractor_v1.json"
GOLDEN_DATASET_PATH = "./dataset/clean_benchmark.json"


def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)


# Create a vector database from text files in the data directory
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


# This function would read the dataset, extract relevant text, and create the vector DB.
def create_vector_db_from_dataset():
    # Clear old database
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH, onexc=remove_readonly)

    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)

    documents = []
    golden_dataset = []

    # Iterate and create Document-level chunks
    for item in data:
        q_id = item['_id']
        # CRITICAL: Ensure we only take questions with exactly 10 contexts (as per dataset structure)
        if len(item['context']) != 10:
            continue

        golden_dataset.append(item)
        for context_list in item['context']:
            title = context_list[0]
            paragraph_text = "".join(context_list[1])
            chunk_text = f"Title: {title}\nText: {paragraph_text}"

            # CRITICAL: Bind the chunk to the specific question
            metadata = {
                "question_id": q_id,
                "title": title
            }

            # Create LangChain Document (NO TextSplitter used)
            documents.append(Document(page_content=chunk_text, metadata=metadata))

    print(f"Created {len(documents)} distinct paragraph chunks (10 per question).")

    with open(GOLDEN_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(golden_dataset, f, indent=2)
    print(f"✅ Saved golden benchmark to '{GOLDEN_DATASET_PATH}'.")

    # Vectorize and Store
    print("Embedding and storing in ChromaDB...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                            model_kwargs={'device': 'cuda'},
                                            show_progress=True)
    Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    print("✅ Ingestion Complete.")


if __name__ == "__main__":
    create_vector_db_from_dataset()

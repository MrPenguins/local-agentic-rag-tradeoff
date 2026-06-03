"""
Shared initialization — module-level singleton, executed once on first import.
Import from here instead of duplicating config + warm-up across pipelines.
"""

import yaml
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
with open("../config.yaml", "r") as f:
    _config = yaml.safe_load(f)

K = _config["database"]["k_retrieval"]
MAX_RETRIES = _config["models"]["max_retries"]

embedding_model = HuggingFaceEmbeddings(
    model_name=_config["models"]["embedding_name"],
    model_kwargs={"device": _config["models"]["device"]},
)
vectorstore = Chroma(
    persist_directory=_config["database"]["path"],
    embedding_function=embedding_model,
)
llm = ChatOllama(
    model=_config["models"]["llm_name"],
    temperature=_config["models"]["llm_temperature"],
)

# --- Warm-up ---
print("Initializing models and warming up GPU...")
print(f"   LLM: {_config['models']['llm_name']}")
llm.invoke("Hi")
embedding_model.embed_query("Warm up the GPU memory pool.")
print("GPU is fully warm. Ready to benchmark.")

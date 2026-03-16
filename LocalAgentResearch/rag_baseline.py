import time

import yaml
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

DB_PATH = config['database']['path']
K = config['database']['k_retrieval']
MODEL_NAME = config['models']['llm_name']
EMBEDDING_NAME = config['models']['embedding_name']
DEVICE = config['models']['device']
LLM_TEMPERATURE = config['models']['llm_temperature']

# GLOBAL INITIALIZATION
print("Initializing models and warming up GPU...")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_NAME,
    model_kwargs={'device': DEVICE}
)
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

llm = ChatOllama(model=MODEL_NAME, temperature=LLM_TEMPERATURE)

# WARM UP THE LLM
print("Sending warm-up ping to Ollama...")
llm.invoke("Hi")

# WARM UP THE EMBEDDING MODEL
print("Sending warm-up ping to Embedding Model...")
embedding_model.embed_query("Warm up the GPU memory pool.")

print("GPU is fully warm. Ready to benchmark.")


def run_standard_rag(question, question_id):
    print(f"\n--- Processing Question: {question} ---")

    start_time = time.time()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": K, "filter": {"question_id": question_id}}
    )

    docs = retriever.invoke(question)

    titles_used = [doc.metadata.get("title", "Unknown Title") for doc in docs]
    print(f"   (Retrieved Sources: {titles_used})")
    context_text = "\n\n".join(doc.page_content for doc in docs)

    template = """
    You are a strict Information Synthesizer.
    Answer the question based ONLY on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()

    print("Thinking...")
    answer = chain.invoke({
        "context": context_text,
        "question": question
    })

    end_time = time.time()
    latency = end_time - start_time

    print(f"Answer: {answer}")
    print(f"📚 Sources Used: {titles_used}")
    print(f"⏱️ Total Latency: {latency:.2f} seconds")

    return answer, latency, titles_used


if __name__ == "__main__":
    run_standard_rag("Were Scott Derrickson and Ed Wood of the same nationality?", "5a8b57f25542995d1e6f1371")

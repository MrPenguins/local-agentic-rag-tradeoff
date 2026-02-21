import time

import yaml
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
print("GPU is warm. Ready to benchmark.")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_standard_rag(question, question_id):
    print(f"--- Question: {question} ---")

    start_time = time.time()

    retriever = vectorstore.as_retriever(search_kwargs={"k": K, "filter": {
        "question_id": question_id}})  # Retrieve top 2 chunks

    template = """Answer the question based ONLY on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    print("Thinking...")
    answer = rag_chain.invoke(question)

    end_time = time.time()
    latency = end_time - start_time

    print(f"Answer: {answer}")
    print(f"⏱️ Total Latency: {latency:.2f} seconds")

    return answer, latency


if __name__ == "__main__":
    run_standard_rag("Were Scott Derrickson and Ed Wood of the same nationality?", "5a8b57f25542995d1e6f1371")

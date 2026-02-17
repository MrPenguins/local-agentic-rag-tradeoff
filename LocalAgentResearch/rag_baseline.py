import time
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
DB_PATH = "./vectorstore"
MODEL_NAME = "llama3.1"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_standard_rag(question):
    print(f"--- Question: {question} ---")

    # 1. Start Timer
    start_time = time.time()

    # 2. Setup Vector DB & Retriever
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Retrieve top 2 chunks

    # 3. Setup LLM
    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    # 4. Define Prompt
    template = """Answer the question based ONLY on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 5. Build Chain
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # 6. Execute (The actual "Generation" phase)
    print("Thinking...")
    answer = rag_chain.invoke(question)

    # 7. Stop Timer
    end_time = time.time()
    latency = end_time - start_time

    print(f"Answer: {answer}")
    print(f"⏱️ Total Latency: {latency:.2f} seconds")

    return answer, latency


if __name__ == "__main__":
    # Test with a question answerable by your dummy data
    run_standard_rag("Why do agentic workflows introduce latency?")
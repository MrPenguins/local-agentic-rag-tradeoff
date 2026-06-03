import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from hybrid_retriever import hybrid_search_with_score
from shared import llm, vectorstore, K


def run_standard_rag(question, question_id):
    print(f"\n--- Processing Question: {question} ---")

    start_time = time.time()

    scored = hybrid_search_with_score(question, question_id, vectorstore, k=K)
    docs = [doc for doc, _ in scored]

    titles_used = [doc.metadata.get("title", "Unknown Title") for doc in docs]
    print(f"   (Retrieved Sources: {titles_used})")
    context_text = "\n\n".join(doc.page_content for doc in docs)

    template = """
    You are a strict Information Synthesizer.
    Answer the question based ONLY on the following context:

    Context:
    {context}

    Question: {question}

    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()

    print("Thinking...")

    # --- TTFT Streaming Logic ---
    answer = ""
    ttft = 0.0

    # Stream the output chunk by chunk
    for chunk in chain.stream({
        "context": context_text,
        "question": question
    }):
        if ttft == 0.0:
            # Capture the exact moment the first token arrives
            ttft = time.time() - start_time
        answer += chunk

    end_time = time.time()
    total_latency = end_time - start_time

    print(f"Answer: {answer}")
    print(f"📚 Sources Used: {titles_used}")
    print(f"⏱️ TTFT: {ttft:.2f} seconds | Total Latency: {total_latency:.2f} seconds")

    return answer, total_latency, titles_used, ttft


if __name__ == "__main__":
    run_standard_rag("Were Scott Derrickson and Ed Wood of the same nationality?", "5a8b57f25542995d1e6f1371")

import time
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from hybrid_retriever import hybrid_search_with_score
from shared import llm, vectorstore, K


# --- 1. Define the State ---
class AgentState(TypedDict):
    question: str
    question_id: str
    raw_docs: list  # Stores the 2*K raw documents and scores
    filtered_docs: list  # Stores the final K documents
    context: str
    answer: str
    retrieved_titles: list
    start_time: float
    ttft: float


# --- Node 1: The Deep Retriever ---
def retriever_node(state: AgentState):
    print("\n--- 🔎 RETRIEVER: Fetching Deep Pool ---")
    fetch_k = K * 2
    print(f"   (Fetching top {fetch_k} documents for evaluation)")

    docs_with_scores = hybrid_search_with_score(
        state["question"], state["question_id"], vectorstore, k=fetch_k
    )

    return {"raw_docs": docs_with_scores}


# --- Node 2: The LLM Filter (Pointwise Evaluator with CoT) ---
def filter_node(state: AgentState):
    print("--- ⚖️ FILTER: Evaluating Document Relevance (CoT) ---")

    template = """You are a strict relevance grader. 
    Analyze the Document against the Question.
    A document is relevant if it contains the final answer OR if it contains essential "bridge" entities (like a specific name, movie, or location) required to research the final answer.

    Document: {document}
    Question: {question}

    Output your response in exactly two lines:
    Rationale: [1 short sentence explaining if it contains the answer or a bridge entity]
    Decision: [YES or NO]"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    relevant_docs = []
    irrelevant_docs = []

    # Evaluate each document individually
    for doc, score in state["raw_docs"]:
        result = chain.invoke({
            "document": doc.page_content,
            "question": state["question"]
        })

        # Stricter parsing to extract the CoT decision safely
        if "DECISION: YES" in result.upper() or "\nYES" in result.upper()[-5:]:
            relevant_docs.append((doc, score))
        else:
            irrelevant_docs.append((doc, score))

    print(f"   (LLM Graded: {len(relevant_docs)} Relevant, {len(irrelevant_docs)} Irrelevant)")

    # Sort both lists by distance score (ascending, lower is better)
    relevant_docs.sort(key=lambda x: x[1])
    irrelevant_docs.sort(key=lambda x: x[1])

    # 1. Take top K from relevant
    final_docs = relevant_docs[:K]

    # 2. Enforce Fallback Padding (if LLM found fewer than K relevant docs)
    if len(final_docs) < K:
        shortfall = K - len(final_docs)
        padding = irrelevant_docs[:shortfall]
        final_docs.extend(padding)
        print(f"   (Padded context with {len(padding)} highest-similarity fallback docs)")

    # Extract final objects and metadata
    unique_docs = [doc for doc, score in final_docs]
    context_text = "\n\n".join(doc.page_content for doc in unique_docs)
    titles = [doc.metadata.get("title", "Unknown Title") for doc in unique_docs]

    print(f"   (Final Context Sources: {titles})")

    return {
        "filtered_docs": unique_docs,
        "context": context_text,
        "retrieved_titles": titles
    }


# --- Node 3: The Generator ---
def generator_node(state: AgentState):
    print("--- 🏃 GENERATOR: Synthesizing Answer ---")

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

    for chunk in chain.stream({
        "context": state["context"],
        "question": state["question"]
    }):
        if ttft == 0.0:
            ttft = time.time() - state["start_time"]
        answer += chunk

    return {
        "answer": answer,
        "ttft": ttft
    }


# --- Build the Graph ---
def build_linear_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("retriever", retriever_node)
    workflow.add_node("filter", filter_node)
    workflow.add_node("generator", generator_node)

    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "filter")
    workflow.add_edge("filter", "generator")
    workflow.add_edge("generator", END)

    return workflow.compile()


linear_agent = build_linear_agent()


# --- Execution Wrapper ---
def run_linear_agent(question: str, question_id: str):
    print(f"\n--- Processing Question: {question} ---")

    start_time = time.time()

    result = linear_agent.invoke({
        "question": question,
        "question_id": question_id,
        "start_time": start_time
    })

    end_time = time.time()
    latency = end_time - start_time

    titles_used = result.get('retrieved_titles', [])
    ttft = result.get('ttft', 0.0)

    print(f"Answer: {result['answer']}")
    print(f"📚 Sources Used: {titles_used}")
    print(f"⏱️ TTFT: {ttft:.2f} seconds | Total Latency: {latency:.2f} seconds")

    return result['answer'], latency, titles_used, ttft


if __name__ == "__main__":
    run_linear_agent("Were Scott Derrickson and Ed Wood of the same nationality?", "5a8b57f25542995d1e6f1371")

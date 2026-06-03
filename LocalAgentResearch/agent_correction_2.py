import time
from typing import TypedDict, List, Tuple
from langgraph.graph import StateGraph, START, END

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from hybrid_retriever import hybrid_search_with_score
from shared import llm, vectorstore, K, MAX_RETRIES


# --- 1. Define the Stateful Memory ---
class AgentState(TypedDict):
    question: str
    question_id: str
    current_query: str  # Single targeted query per loop
    raw_docs: list  # Temporary 2K pool from Deep Fetch
    filtered_docs: list  # Locked-in K docs that survive the filter
    context: str  # String representation for the generator
    answer: str
    feedback: str
    loop_count: int
    retrieved_titles: list
    start_time: float
    ttft: float


# --- Node 1: The Query Rewriter ---
def planner_node(state: AgentState):
    loop = state.get('loop_count', 0)
    print(f"\n--- 🧠 PLANNER: Determining Search Strategy (Loop {loop}) ---")

    if loop == 0 or not state.get("feedback"):
        query = state["question"]
        print(f"   (Initial Search. Using Original Question: '{query}')")
        return {"current_query": query}

    feedback = state["feedback"]
    print(f"   (Analyzing Reviewer Feedback: {feedback})")

    template = """You are a Search Query Generator.
    The previous attempt to answer the user's question failed because information was missing.

    Original Question: {question}
    Missing Information (Reviewer Feedback): {feedback}

    CRITICAL RULE: Generate a SINGLE, highly specific search query designed strictly to find the missing information. Do not output anything other than the query itself.

    Targeted Search Query:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    raw_query = chain.invoke({"question": state["question"], "feedback": feedback})
    clean_query = raw_query.strip().replace('"', '')

    # Fallback to avoid catastrophic API failure
    if not clean_query:
        clean_query = state["question"]

    print(f"   (Generated Targeted Query: '{clean_query}')")
    return {"current_query": clean_query}


# --- Node 2: The Deep Retriever ---
def retriever_node(state: AgentState):
    print("--- 🔎 RETRIEVER: Fetching Deep Pool ---")
    fetch_k = K * 2
    print(f"   (Fetching top {fetch_k} documents for current query)")

    # Returns [(Document, score), ...]
    docs_with_scores = hybrid_search_with_score(
        state["current_query"], state["question_id"], vectorstore, k=fetch_k
    )

    # Embed score into metadata so it persists cleanly in the state list
    for doc, score in docs_with_scores:
        doc.metadata["distance_score"] = score

    docs_only = [doc for doc, score in docs_with_scores]
    return {"raw_docs": docs_only}


# --- Node 3: The Stateful Filter ---
def filter_node(state: AgentState):
    print("--- ⚖️ FILTER: Executing Stateful Evaluation (CoT) ---")

    loop = state.get('loop_count', 0)
    evaluator_chain = llm | StrOutputParser()

    final_docs = []

    # LOOP 0: Standard Blind Fetch with Bridge Prompting
    if loop == 0:
        print("   (Loop 0: Executing CoT relevance check with Bridge logic)")
        template = """You are a strict relevance grader. 
        Analyze the Document against the Question.
        A document is relevant if it contains the final answer OR if it contains essential "bridge" entities (like a specific name, movie, or location) required to research the final answer.

        Document: {document}
        Question: {question}

        Output your response in exactly two lines:
        Rationale: [1 short sentence explaining if it contains the answer or a bridge entity]
        Decision: [YES or NO]"""

        prompt = ChatPromptTemplate.from_template(template)

        relevant = []
        irrelevant = []
        for doc in state["raw_docs"]:
            res = evaluator_chain.invoke(prompt.format(document=doc.page_content, question=state["question"]))

            # Stricter parsing to avoid false positives in the rationale
            if "DECISION: YES" in res.upper() or "\nYES" in res.upper()[-5:]:
                relevant.append(doc)
            else:
                irrelevant.append(doc)

        relevant.sort(key=lambda x: x.metadata.get("distance_score", 999))
        irrelevant.sort(key=lambda x: x.metadata.get("distance_score", 999))

        final_docs = relevant[:K]
        if len(final_docs) < K:
            final_docs.extend(irrelevant[:K - len(final_docs)])

    # LOOP 1+: Feedback-Conditioned Stateful Filtering
    else:
        feedback = state["feedback"]
        print("   (Loop 1+: Executing Force-Purge and Targeted Hunt)")

        # Step A: Force-Purge (Retention)
        retained = []
        retention_prompt = ChatPromptTemplate.from_template("""Does this document provide factual background context that is true and helpful, even if it does not contain the exact missing information? 
        Look for "bridge" entities that connect to the missing information.

        Missing Information: {feedback}
        Document: {document}

        Output your response in exactly two lines:
        Rationale: [1 short sentence explaining your reasoning]
        Decision: [YES or NO]""")

        for doc in state.get("filtered_docs", []):
            if len(retained) >= K - 1:  # Force at least 1 slot open
                break
            res = evaluator_chain.invoke(retention_prompt.format(feedback=feedback, document=doc.page_content))
            if "DECISION: YES" in res.upper() or "\nYES" in res.upper()[-5:]:
                retained.append(doc)

        print(f"   (Retained {len(retained)} background documents from previous loop)")

        # Step B: Targeted Hunt (Acquisition)
        needed_slots = K - len(retained)
        acquired = []
        irrelevant = []

        acquisition_prompt = ChatPromptTemplate.from_template("""Analyze if this document contains the specific MISSING information identified below.

        Missing Information: {feedback}
        Document: {document}

        Output your response in exactly two lines:
        Rationale: [1 short sentence explaining if the exact missing fact is present]
        Decision: [YES or NO]""")

        for doc in state["raw_docs"]:
            res = evaluator_chain.invoke(acquisition_prompt.format(feedback=feedback, document=doc.page_content))
            if "DECISION: YES" in res.upper() or "\nYES" in res.upper()[-5:]:
                acquired.append(doc)
            else:
                irrelevant.append(doc)

        acquired.sort(key=lambda x: x.metadata.get("distance_score", 999))
        irrelevant.sort(key=lambda x: x.metadata.get("distance_score", 999))

        acquired = acquired[:needed_slots]
        print(f"   (Acquired {len(acquired)} new documents targeting missing facts)")

        final_docs = retained + acquired

        # Step C: Fallback Padding
        if len(final_docs) < K:
            padding_needed = K - len(final_docs)
            padding = irrelevant[:padding_needed]
            final_docs.extend(padding)
            print(f"   (Padded {len(padding)} slots with fallback documents)")

    # Extract final context
    context_text = "\n\n".join(doc.page_content for doc in final_docs)
    titles = [doc.metadata.get("title", "Unknown Title") for doc in final_docs]
    print(f"   (Final Filtered Sources: {titles})")

    return {
        "filtered_docs": final_docs,
        "context": context_text,
        "retrieved_titles": titles
    }


# --- Node 4: The Generator ---
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
    ttft = 0.0  # Reset locally on every loop

    for chunk in chain.stream({"context": state["context"], "question": state["question"]}):
        if ttft == 0.0:
            # Overwrites on every loop, ensuring TTFT reflects the final answer's start time
            ttft = time.time() - state["start_time"]
        answer += chunk

    current_loop = state.get("loop_count", 0)
    return {
        "answer": answer,
        "loop_count": current_loop + 1,
        "ttft": ttft
    }


# --- Node 5: The Reviewer ---
def reviewer_node(state: AgentState):
    print("--- 🔎 REVIEWER: Grading Answer ---")

    template = """You are a Critical Reviewer.
    Analyze the Question and Answer. Check for Hallucinations and Missing information.

    - If correct, output exactly "STATUS: PASS".
    - If incorrect or incomplete, start your response with "STATUS: FAIL". Then, concisely state EXACTLY what information is missing.

    Question: {question}
    Context Provided: {context}
    Answer: {answer}
    Result:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    review = chain.invoke({
        "question": state["question"],
        "context": state["context"],
        "answer": state["answer"]
    })

    clean_review = review.strip().upper()

    if clean_review.startswith("STATUS: PASS"):
        print("   ✅ Review: PASSED")
        return {"feedback": None}
    elif clean_review.startswith("STATUS: FAIL"):
        print(f"   ❌ Review: FAILED")
        reason = review.replace("STATUS: FAIL", "").strip()
        return {"feedback": reason}
    else:
        print(f"   ⚠️ Ambiguous Review. Defaulting to FAIL.")
        return {"feedback": "The answer was ambiguous. Ensure all parts of the question are answered."}


# --- The Router Logic ---
def should_continue(state: AgentState):
    feedback = state.get("feedback")
    loop_count = state.get("loop_count", 0)

    if not feedback:
        return "end"

    if loop_count >= MAX_RETRIES:
        print("--- 🛑 Max retries reached. Stopping. ---")
        return "end"

    return "retry"


# --- Build the Graph ---
def build_correction_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("filter", filter_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "filter")
    workflow.add_edge("filter", "generator")
    workflow.add_edge("generator", "reviewer")

    workflow.add_conditional_edges("reviewer", should_continue, {"end": END, "retry": "planner"})

    return workflow.compile()


correction_agent = build_correction_agent()


# --- Execution Wrapper ---
def run_correction_agent(question: str, question_id: str):
    print(f"\n--- Processing Question: {question} ---")
    start_time = time.time()

    result = correction_agent.invoke({
        "question": question,
        "question_id": question_id,
        "loop_count": 0,
        "ttft": 0.0,
        "start_time": start_time
    })

    end_time = time.time()
    total_latency = end_time - start_time
    titles_used = result.get('retrieved_titles', [])
    ttft = result.get('ttft', 0.0)

    print(f"\nFinal Answer: {result['answer']}")
    print(f"📚 Sources Used: {titles_used}")
    print(f"⏱️ TTFT: {ttft:.2f} seconds | Total Latency: {total_latency:.2f} seconds")
    print(f"🔄 Total Loops: {result.get('loop_count', 1)}")

    return result['answer'], total_latency, titles_used, result.get('loop_count', 1), ttft


if __name__ == "__main__":
    run_correction_agent("Were Scott Derrickson and Ed Wood of the same nationality?", "5a8b57f25542995d1e6f1371")
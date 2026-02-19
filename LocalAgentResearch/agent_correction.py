import time
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
DB_PATH = "./vectorstore"
MODEL_NAME = "llama3.1"
MAX_RETRIES = 3  # Safety break to prevent infinite loops


# 1. Update the State
# We add 'feedback' to pass criticism back to the planner
# We add 'loop_count' to track how many times we've tried
class AgentState(TypedDict):
    question: str
    plan: str
    context: str
    answer: str
    feedback: str  # <--- NEW: The Critic's complaints
    loop_count: int  # <--- NEW: Safety counter


# --- Node 1: The Adaptive Planner ---
def planner_node(state: AgentState):
    print("--- 🧠 PLANNER: Generating Strategy ---")
    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    # Check if this is a retry
    feedback = state.get("feedback", "")
    original_question = state["question"]
    if feedback == "Ambiguous review output. Please generate a new plan.":
        # If the Reviewer glitched, we don't blame the plan.
        # We just tell the Planner to try a slightly different angle.
        print("   (System Warning: Reviewer failed to parse. Asking Planner to retry.)")
        template = """You are a Planner Agent.
            The previous attempt was technically successful, but the Reviewer failed to parse it.

            Original Question: {question}

            Action: Create a ROBUST plan that is slightly different from the last one to ensure clear results.
            Plan:"""
    elif feedback:
        print(f"   (Refining plan based on feedback: {feedback})")
        template = """You are a Planner Agent.
        The previous attempt to answer this question FAILED.

        Original Question: {question}
        Previous Plan Failure Reason: {feedback}

        Create a NEW, BETTER plan to find the correct answer. Focus on the missing information.
        Plan:"""
    else:
        # First attempt (Standard)
        template = """You are a Planner Agent.
        Break down the following complex question into a clear, step-by-step search plan.

        Question: {question}

        Plan:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    plan = chain.invoke({"question": original_question, "feedback": feedback})
    return {"plan": plan}


# --- Node 2: The Performer (Same as before) ---
def performer_node(state: AgentState):
    print("--- 🏃 PERFORMER: Executing RAG ---")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    # Retrieve
    docs = retriever.invoke(state["question"])
    context_text = "\n\n".join(doc.page_content for doc in docs)

    # Generate
    template = """You are an Expert Performer. 
    Execute the plan to answer the question using the context.

    Plan: {plan}
    Context: {context}
    Question: {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"plan": state["plan"], "context": context_text, "question": state["question"]})

    # Increment loop count here
    current_loop = state.get("loop_count", 0)
    return {"answer": answer, "context": context_text, "loop_count": current_loop + 1}


# --- Node 3: The Reviewer (NEW) ---
def reviewer_node(state: AgentState):
    print("--- 🔎 REVIEWER: Grading Answer ---")
    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    template = """You are a Critical Reviewer.
    Analyze the following Question and Answer.
    Check for:
    1. Hallucinations (facts not in context).
    2. Missing information (did not answer the full question).
    
    - If the answer is correct, output exactly "STATUS: PASS".
    - If the answer is incorrect or incomplete, start your response with "STATUS: FAIL". Then, provide your reasoning.

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
        # We remove the "STATUS: FAIL" tag so the planner just sees the reason
        reason = review.replace("STATUS: FAIL", "").strip()
        return {"feedback": reason}

    else:
        # Fallback: If the LLM didn't follow instructions (rare with Llama 3), fail safely.
        print(f"   ⚠️ Ambiguous Review. Defaulting to FAIL. (Output: {review[:50]}...)")
        return {"feedback": "Ambiguous review output. Please generate a new plan."}


# --- The Router Logic (Conditional Edge) ---
def should_continue(state: AgentState):
    feedback = state.get("feedback")
    loop_count = state.get("loop_count", 0)

    # Condition 1: If approved (no feedback), Stop.
    if not feedback:
        return "end"

    # Condition 2: If we tried too many times, Stop (give up).
    if loop_count >= MAX_RETRIES:
        print("--- 🛑 Max retries reached. Stopping. ---")
        return "end"

    # Condition 3: Otherwise, Loop back to Planner.
    return "retry"


# --- Build the Graph ---
def build_correction_agent():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("performer", performer_node)
    workflow.add_node("reviewer", reviewer_node)

    # Add Edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "performer")
    workflow.add_edge("performer", "reviewer")

    # Add Conditional Edge
    workflow.add_conditional_edges(
        "reviewer",  # Start at Reviewer
        should_continue,  # Run this function to decide where to go
        {  # Map the function's output to Nodes
            "end": END,
            "retry": "planner"
        }
    )

    return workflow.compile()


# --- Execution ---
if __name__ == "__main__":
    agent = build_correction_agent()
    question = "Why do agentic workflows introduce latency?"

    print(f"Processing: {question}")
    start_time = time.time()

    # Initialize state with loop_count = 0
    result = agent.invoke({"question": question, "loop_count": 0})

    end_time = time.time()
    print(f"\nFinal Answer: {result['answer']}")
    print(f"⏱️ Total Latency: {end_time - start_time:.2f} seconds")
    print(f"🔄 Total Loops: {result['loop_count']}")

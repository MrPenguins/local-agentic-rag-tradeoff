import time
from typing import TypedDict
import yaml
from langgraph.graph import StateGraph, START, END

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
MAX_RETRIES = config['models']['max_retries']

# --- Global Initialization ---
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


# --- 1. Define the State ---
class AgentState(TypedDict):
    question: str
    question_id: str
    plan: str
    context: str
    answer: str
    feedback: str  # The Critic's complaints
    loop_count: int  # Safety counter


# --- Node 1: The Adaptive Planner ---
def planner_node(state: AgentState):
    print("--- 🧠 PLANNER: Generating Strategy ---")

    # Check if this is a retry
    feedback = state.get("feedback", "")
    original_question = state["question"]

    if feedback == "Ambiguous review output. Please generate a new plan.":
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


# --- Node 2: The Performer (Enhanced RAG) ---
def performer_node(state: AgentState):
    print("--- 🏃 PERFORMER: Executing RAG ---")

    # 1. Retrieve Context using Metadata Filtering and global vectorstore
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": K,
            "filter": {"question_id": state["question_id"]}
        }
    )

    docs = retriever.invoke(state["question"])
    context_text = "\n\n".join(doc.page_content for doc in docs)

    # 2. Generate Answer
    template = """You are an Expert Performer. 
    Execute the plan to answer the question using the context.

    Plan: {plan}
    Context: {context}
    Question: {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "plan": state.get("plan", ""),
        "context": context_text,
        "question": state["question"]
    })

    # Increment loop count here
    current_loop = state.get("loop_count", 0)
    return {"answer": answer, "context": context_text, "loop_count": current_loop + 1}


# --- Node 3: The Reviewer ---
def reviewer_node(state: AgentState):
    print("--- 🔎 REVIEWER: Grading Answer ---")

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
        reason = review.replace("STATUS: FAIL", "").strip()
        return {"feedback": reason}

    else:
        print(f"   ⚠️ Ambiguous Review. Defaulting to FAIL. (Output: {review[:50]}...)")
        return {"feedback": "Ambiguous review output. Please generate a new plan."}


# --- The Router Logic (Conditional Edge) ---
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
    workflow.add_node("performer", performer_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "performer")
    workflow.add_edge("performer", "reviewer")

    workflow.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "end": END,
            "retry": "planner"
        }
    )

    return workflow.compile()


correction_agent = build_correction_agent()


# --- Execution Wrapper ---
def run_correction_agent(question: str, question_id: str):
    """
    Executes the self-correction agent workflow and records latency.
    """
    print(f"\n--- Processing Question: {question} ---")
    start_time = time.time()

    # Run the Graph
    result = correction_agent.invoke({
        "question": question,
        "question_id": question_id,
        "loop_count": 0
    })

    end_time = time.time()
    total_latency = end_time - start_time

    print(f"\nFinal Answer: {result['answer']}")
    print(f"⏱️ Total Latency: {total_latency:.2f} seconds")
    print(f"🔄 Total Loops: {result.get('loop_count', 1)}")

    return result['answer'], total_latency, result.get('loop_count', 1)


if __name__ == "__main__":
    run_correction_agent("Were Scott Derrickson and Ed Wood of the same nationality?", "5a8b57f25542995d1e6f1371")

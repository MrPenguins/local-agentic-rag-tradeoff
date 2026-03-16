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

# WARM UP THE EMBEDDING MODEL
print("Sending warm-up ping to Embedding Model...")
embedding_model.embed_query("Warm up the GPU memory pool.")

print("GPU is fully warm. Ready to benchmark.")


# --- 1. Define the State ---
class AgentState(TypedDict):
    question: str
    question_id: str
    search_queries: list
    context: str
    answer: str
    feedback: str
    loop_count: int
    retrieved_titles: list


# --- Node 1: The Adaptive Planner (Query Generator) ---
def planner_node(state: AgentState):
    print(f"\n--- 🧠 PLANNER: Generating Search Queries (Loop {state.get('loop_count', 0)}) ---")

    feedback = state.get("feedback", "")
    original_question = state["question"]

    if feedback == "Ambiguous review output. Please generate a new plan.":
        print("   (System Warning: Reviewer failed to provide clear feedback. Generating fallback queries.)")
        template = """You are a Search Query Generator.
        The previous attempt was technically successful, but the Reviewer failed to parse it.

        CRITICAL RULE: Output ONLY the queries, separated by a pipe character (|). Do not add bullet points, numbers, or introductory text.

        Example Input 1 (Comparison): Which film was released first, Inception or The Matrix?
        Example Output 1: When was the film Inception released? | When was the film The Matrix released?

        Example Input 2 (Bridge): What is the nationality of the director of the movie "Parasite"?
        Example Output 2: Who is the director of the movie Parasite? | What is the nationality of the director of the movie Parasite?

        Action: Create 2 NEW search queries slightly different from the last ones.

        Original Question: {question}
        Queries:"""

    elif feedback:
        print(f"   (Refining queries based on feedback: {feedback})")
        template = """You are a Search Query Generator.
        The previous attempt to answer this question FAILED.

        CRITICAL RULE 1: Output ONLY the queries, separated by a pipe character (|). Do not add bullet points, numbers, or introductory text.
        CRITICAL RULE 2: Your revised queries MUST cover the ENTIRE original question. You must re-ask for the information you need to keep, AND add a new specific query to hunt down the missing information mentioned in the feedback. Do not *only* ask about the missing information, or you will lose the context of the rest of the question!

        Example Input 1 (Comparison): Which film was released first, Inception or The Matrix?
        Example Output 1: When was the film Inception released? | When was the film The Matrix released?

        Example Input 2 (Bridge): What is the nationality of the director of the movie "Parasite"?
        Example Output 2: Who is the director of the movie Parasite? | What is the nationality of the director of the movie Parasite?

        Action: Create a REVISED set of 2 search queries that solves the failure reason while maintaining the full scope of the question.
        
        Original Question: {question}
        Failure Reason: {feedback}

        Queries:"""
    else:
        # First attempt
        template = """You are a Search Query Generator.
        Break down the following complex question into 2 simple, atomic search queries.

        CRITICAL RULE: Output ONLY the queries, separated by a pipe character (|). Do not add bullet points, numbers, or introductory text.

        Example Input 1 (Comparison): Which film was released first, Inception or The Matrix?
        Example Output 1: When was the film Inception released? | When was the film The Matrix released?

        Example Input 2 (Bridge): What is the nationality of the director of the movie "Parasite"?
        Example Output 2: Who is the director of the movie Parasite? | What is the nationality of the director of the movie Parasite?

        Question: {question}
        Queries:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    raw_output = chain.invoke({"question": original_question, "feedback": feedback})

    # Parse the output safely
    queries = [q.strip() for q in raw_output.split("|") if q.strip()]
    if not queries:
        queries = [original_question]

    print(f"   (Generated Queries: {queries})")
    return {"search_queries": queries}


# --- Node 2: The Performer (Multi-Query RAG with Budget) ---
def performer_node(state: AgentState):
    print("--- 🏃 PERFORMER: Executing Multi-Query RAG ---")

    queries = state["search_queries"]

    # Dynamic K calculation
    k_per_query = max(1, K // len(queries))
    print(f"   (Budget constraint: {K} total docs -> {k_per_query} docs per query)")

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k_per_query,
            "filter": {"question_id": state["question_id"]}
        }
    )

    all_docs = []
    for query in queries:
        docs = retriever.invoke(query)
        all_docs.extend(docs)

    # Deduplication
    unique_docs = []
    seen_content = set()
    for doc in all_docs:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            unique_docs.append(doc)

    unique_docs = unique_docs[:K]

    context_text = "\n\n".join(doc.page_content for doc in unique_docs)
    titles = [doc.metadata.get("title", "Unknown Title") for doc in unique_docs]
    print(f"   (Retrieved Unique Sources: {titles})")

    # Strict Synthesis Prompt (No "performing" actions)
    template = """You are a strict Information Synthesizer.
    Answer the question based ONLY on the following context:
    
    Context:
    {context}

    Question: {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context_text,
        "question": state["question"]
    })

    current_loop = state.get("loop_count", 0)
    return {
        "answer": answer,
        "context": context_text,
        "loop_count": current_loop + 1,
        "retrieved_titles": titles
    }


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
    print(f"\n--- Processing Question: {question} ---")
    start_time = time.time()

    result = correction_agent.invoke({
        "question": question,
        "question_id": question_id,
        "loop_count": 0
    })

    end_time = time.time()
    total_latency = end_time - start_time
    titles_used = result.get('retrieved_titles', [])

    print(f"\nFinal Answer: {result['answer']}")
    print(f"📚 Sources Used: {titles_used}")
    print(f"⏱️ Total Latency: {total_latency:.2f} seconds")
    print(f"🔄 Total Loops: {result.get('loop_count', 1)}")

    return result['answer'], total_latency, titles_used, result.get('loop_count', 1)


if __name__ == "__main__":
    run_correction_agent("Were Scott Derrickson and Ed Wood of the same nationality?", "5a8b57f25542995d1e6f1371")

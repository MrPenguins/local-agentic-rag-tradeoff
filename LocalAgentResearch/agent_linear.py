import time
from typing import TypedDict
import yaml
from langgraph.graph import StateGraph, START, END

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from hybrid_retriever import hybrid_search_with_score

# --- Configuration ---
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

DB_PATH = config['database']['path']
K = config['database']['k_retrieval']
MODEL_NAME = config['models']['llm_name']
EMBEDDING_NAME = config['models']['embedding_name']
DEVICE = config['models']['device']
LLM_TEMPERATURE = config['models']['llm_temperature']

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
    retrieved_titles: list
    start_time: float
    ttft: float


# --- Node 1: The Planner (Query Generator) ---
def planner_node(state: AgentState):
    print("--- 🧠 PLANNER: Generating Search Queries ---")

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

    raw_output = chain.invoke({"question": state["question"]})

    # Parse the output into a Python list
    queries = [q.strip() for q in raw_output.split("|") if q.strip()]

    # Fallback: If the model failed to format, just use the original question
    if not queries:
        queries = [state["question"]]

    print(f"   (Generated Queries: {queries})")

    return {"search_queries": queries}


# --- Node 2: The Performer (Multi-Query RAG) ---
def performer_node(state: AgentState):
    print("--- 🏃 PERFORMER: Executing Multi-Query RAG ---")

    # 1. Iterative Retrieval with Scores
    all_scored_docs = []
    for query in state["search_queries"]:
        # Returns a list of tuples: (Document, distance_score)
        docs_with_scores = hybrid_search_with_score(
            query, state["question_id"], vectorstore, k=K
        )
        all_scored_docs.extend(docs_with_scores)

    # 2. Deduplication & Score Optimization
    unique_docs_map = {}
    for doc, score in all_scored_docs:
        content = doc.page_content
        # If document is new, OR if we found it again with a BETTER (lower) distance score
        if content not in unique_docs_map or score < unique_docs_map[content][1]:
            unique_docs_map[content] = (doc, score)

    # 3. Global Ranking
    # Sort the dictionary values by the score (index 1), ascending
    ranked_scored_docs = sorted(unique_docs_map.values(), key=lambda x: x[1])

    # 4. Enforce Budget constraint
    top_k_scored = ranked_scored_docs[:K]

    # 5. Extract final documents for generation
    unique_docs = [doc for doc, score in top_k_scored]

    context_text = "\n\n".join(doc.page_content for doc in unique_docs)
    titles = [doc.metadata.get("title", "Unknown Title") for doc in unique_docs]
    print(f"   (Retrieved Unique Sources: {titles})")

    # 3. Generate Answer
    template = """You are a strict Information Synthesizer.
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
        "context": context_text,
        "question": state["question"]
    }):
        if ttft == 0.0:
            # Calculate TTFT based on the global start time passed in the state
            ttft = time.time() - state["start_time"]
        answer += chunk

    return {
        "answer": answer,
        "context": context_text,
        "retrieved_titles": titles,
        "ttft": ttft
    }


# --- Build the Graph ---
def build_linear_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("performer", performer_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "performer")
    workflow.add_edge("performer", END)

    return workflow.compile()


linear_agent = build_linear_agent()


# --- Execution Wrapper ---
def run_linear_agent(question: str, question_id: str):
    """
    Executes the linear agent workflow and records latency.
    """
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

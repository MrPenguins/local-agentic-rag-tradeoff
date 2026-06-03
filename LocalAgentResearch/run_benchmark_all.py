"""
Unified benchmark runner — executes all 5 pipelines per question.

Order: Baseline RAG → Linear V1 → Correction V1 → Linear V2 → Correction V2
"""

import json
import csv
import os

from hybrid_retriever import hybrid_search_with_score
from shared import vectorstore
from rag_baseline import run_standard_rag
from agent_linear import run_linear_agent as run_linear_v1
from agent_correction import run_correction_agent as run_correction_v1
from agent_linear_2 import run_linear_agent as run_linear_v2
from agent_correction_2 import run_correction_agent as run_correction_v2


def extract_gold_titles(supporting_facts):
    """Extract unique Wikipedia titles from the HotpotQA supporting_facts list."""
    return list(set([fact[0] for fact in supporting_facts]))


def run_benchmark(input_json: str, output_csv: str):
    print(f"Loading dataset: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    headers = [
        "question_id", "question", "gold_answer", "gold_titles",
        # Baseline RAG
        "rag_answer", "rag_latency", "rag_ttft", "rag_titles",
        # Linear V1 (query decomposition)
        "linear_v1_answer", "linear_v1_latency", "linear_v1_ttft", "linear_v1_titles",
        # Correction V1 (planner → performer → reviewer loop)
        "correction_v1_answer", "correction_v1_latency", "correction_v1_ttft",
        "correction_v1_titles", "correction_v1_loops",
        # Linear V2 (deep retriever → pointwise filter → generator)
        "linear_v2_answer", "linear_v2_latency", "linear_v2_ttft", "linear_v2_titles",
        # Correction V2 (stateful CRAG)
        "correction_v2_answer", "correction_v2_latency", "correction_v2_ttft",
        "correction_v2_titles", "correction_v2_loops",
    ]

    file_exists = os.path.isfile(output_csv)
    processed_ids = set()
    if file_exists:
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_ids.add(row["question_id"])
        print(f"Found {len(processed_ids)} previously processed questions. Resuming...")

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()

        # --- Silent warm-up (every launch: prime BM25, ChromaDB) ---
        print("Silent warm-up (priming BM25, ChromaDB)...")
        try:
            hybrid_search_with_score("warm up query", None, vectorstore, k=1)
        except Exception:
            pass
        print("Warm-up complete.\n")

        for i, item in enumerate(dataset):
            q_id = item["_id"]

            if q_id in processed_ids:
                continue

            question = item["question"]
            gold_answer = item["answer"]
            gold_titles = extract_gold_titles(item["supporting_facts"])

            print(f"\n{'=' * 50}")
            print(f"Test {i + 1}/{len(dataset)} | ID: {q_id}")
            print(f"Question: {question}")
            print(f"{'=' * 50}")

            # --- 1. Baseline RAG ---
            print("\n>>> [1/5] Baseline RAG...")
            try:
                rag_ans, rag_lat, rag_tit, rag_ttft = run_standard_rag(question, q_id)
            except Exception as e:
                print(f"Error in RAG: {e}")
                rag_ans, rag_lat, rag_tit, rag_ttft = "ERROR", 0.0, [], 0.0

            # --- 2. Linear V1 ---
            print("\n>>> [2/5] Linear V1 (Query Decomposition)...")
            try:
                l1_ans, l1_lat, l1_tit, l1_ttft = run_linear_v1(question, q_id)
            except Exception as e:
                print(f"Error in Linear V1: {e}")
                l1_ans, l1_lat, l1_tit, l1_ttft = "ERROR", 0.0, [], 0.0

            # --- 3. Correction V1 ---
            print("\n>>> [3/5] Correction V1 (Planner → Performer → Reviewer)...")
            try:
                c1_ans, c1_lat, c1_tit, c1_loops, c1_ttft = run_correction_v1(question, q_id)
            except Exception as e:
                print(f"Error in Correction V1: {e}")
                c1_ans, c1_lat, c1_tit, c1_loops, c1_ttft = "ERROR", 0.0, [], 0, 0.0

            # --- 4. Linear V2 ---
            print("\n>>> [4/5] Linear V2 (Deep Retriever → Pointwise Filter)...")
            try:
                l2_ans, l2_lat, l2_tit, l2_ttft = run_linear_v2(question, q_id)
            except Exception as e:
                print(f"Error in Linear V2: {e}")
                l2_ans, l2_lat, l2_tit, l2_ttft = "ERROR", 0.0, [], 0.0

            # --- 5. Correction V2 ---
            print("\n>>> [5/5] Correction V2 (Stateful CRAG)...")
            try:
                c2_ans, c2_lat, c2_tit, c2_loops, c2_ttft = run_correction_v2(question, q_id)
            except Exception as e:
                print(f"Error in Correction V2: {e}")
                c2_ans, c2_lat, c2_tit, c2_loops, c2_ttft = "ERROR", 0.0, [], 0, 0.0

            # --- Save ---
            row = {
                "question_id": q_id,
                "question": question,
                "gold_answer": gold_answer,
                "gold_titles": str(gold_titles),

                "rag_answer": rag_ans,
                "rag_latency": round(rag_lat, 2),
                "rag_ttft": round(rag_ttft, 2),
                "rag_titles": str(rag_tit),

                "linear_v1_answer": l1_ans,
                "linear_v1_latency": round(l1_lat, 2),
                "linear_v1_ttft": round(l1_ttft, 2),
                "linear_v1_titles": str(l1_tit),

                "correction_v1_answer": c1_ans,
                "correction_v1_latency": round(c1_lat, 2),
                "correction_v1_ttft": round(c1_ttft, 2),
                "correction_v1_titles": str(c1_tit),
                "correction_v1_loops": c1_loops,

                "linear_v2_answer": l2_ans,
                "linear_v2_latency": round(l2_lat, 2),
                "linear_v2_ttft": round(l2_ttft, 2),
                "linear_v2_titles": str(l2_tit),

                "correction_v2_answer": c2_ans,
                "correction_v2_latency": round(c2_lat, 2),
                "correction_v2_ttft": round(c2_ttft, 2),
                "correction_v2_titles": str(c2_tit),
                "correction_v2_loops": c2_loops,
            }

            writer.writerow(row)
            f.flush()

            print(f"\n✅ Question {q_id} complete.")

    print(f"\n🎉 Benchmark complete. Results saved to {output_csv}")


if __name__ == "__main__":
    INPUT_FILE_PATH = "./dataset/benchmark_subset_500.json"
    OUTPUT_FILE_PATH = "./output/raw_benchmark_results_500_llama3.1.csv"

    run_benchmark(INPUT_FILE_PATH, OUTPUT_FILE_PATH)

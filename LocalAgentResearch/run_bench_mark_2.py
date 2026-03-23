import json
import csv
import os

from rag_baseline import run_standard_rag

# --- 1. CHANGED IMPORTS ---
from agent_linear_2 import run_linear_agent
from agent_correction_2 import run_correction_agent


def extract_gold_titles(supporting_facts):
    """Extracts unique Wikipedia titles from the HotpotQA supporting_facts list."""
    return list(set([fact[0] for fact in supporting_facts]))


def run_benchmark(input_json: str, output_csv: str):
    print(f"Loading dataset: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Setup CSV Headers
    headers = [
        "question_id", "question", "gold_answer", "gold_titles",
        "rag_answer", "rag_latency", "rag_ttft", "rag_titles",
        "linear_answer", "linear_latency", "linear_ttft", "linear_titles",
        "correction_answer", "correction_latency", "correction_ttft", "correction_titles", "correction_loops"
    ]

    file_exists = os.path.isfile(output_csv)

    # Pre-scan existing CSV to prevent duplicates
    processed_ids = set()
    if file_exists:
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_ids.add(row['question_id'])
        print(f"Found {len(processed_ids)} previously processed questions. Resuming...")

    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()

        # --- 2. CHANGED LOOP TO SLICE FIRST 50 ITEMS ---
        for i, item in enumerate(dataset[:50]):
            q_id = item['_id']

            # Skip if already processed in a previous interrupted run
            if q_id in processed_ids:
                continue

            question = item['question']
            gold_answer = item['answer']
            gold_titles = extract_gold_titles(item['supporting_facts'])

            print(f"\n==================================================")
            # Make sure to print the progress out of 50, not the full dataset length
            print(f"Test {i + 1}/50 | ID: {q_id}")
            print(f"Question: {question}")
            print(f"==================================================")

            # --- 1. Baseline RAG ---
            print("\n>>> Running Baseline RAG...")
            try:
                rag_ans, rag_lat, rag_tit, rag_ttft = run_standard_rag(question, q_id)
            except Exception as e:
                print(f"Error in RAG: {e}")
                rag_ans, rag_lat, rag_tit, rag_ttft = "ERROR", 0.0, [], 0.0

            # --- 2. Linear Agent ---
            print("\n>>> Running Linear Agent...")
            try:
                lin_ans, lin_lat, lin_tit, lin_ttft = run_linear_agent(question, q_id)
            except Exception as e:
                print(f"Error in Linear Agent: {e}")
                lin_ans, lin_lat, lin_tit, lin_ttft = "ERROR", 0.0, [], 0.0

            # --- 3. Correction Agent ---
            print("\n>>> Running Correction Agent...")
            try:
                cor_ans, cor_lat, cor_tit, cor_loops, cor_ttft = run_correction_agent(question, q_id)
            except Exception as e:
                print(f"Error in Correction Agent: {e}")
                cor_ans, cor_lat, cor_tit, cor_loops, cor_ttft = "ERROR", 0.0, [], 0, 0.0

            # --- Save Iteration to CSV ---
            row = {
                "question_id": q_id,
                "question": question,
                "gold_answer": gold_answer,
                "gold_titles": str(gold_titles),

                "rag_answer": rag_ans,
                "rag_latency": round(rag_lat, 2),
                "rag_ttft": round(rag_ttft, 2),
                "rag_titles": str(rag_tit),

                "linear_answer": lin_ans,
                "linear_latency": round(lin_lat, 2),
                "linear_ttft": round(lin_ttft, 2),
                "linear_titles": str(lin_tit),

                "correction_answer": cor_ans,
                "correction_latency": round(cor_lat, 2),
                "correction_ttft": round(cor_ttft, 2),
                "correction_titles": str(cor_tit),
                "correction_loops": cor_loops
            }

            writer.writerow(row)
            f.flush()

            print(f"\n✅ Question {q_id} complete and saved to CSV.")

    print(f"\n🎉 Benchmark complete. Results saved to {output_csv}")


if __name__ == "__main__":
    INPUT_FILE_PATH = "./dataset/benchmark_subset_500.json"

    # --- 3. CHANGED OUTPUT PATH TO AVOID OVERWRITING DATA ---
    OUTPUT_FILE_PATH = "./output/raw_benchmark_results_test_50.csv"

    run_benchmark(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
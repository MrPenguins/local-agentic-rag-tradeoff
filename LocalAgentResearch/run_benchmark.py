import json
import csv
import os

from rag_baseline import run_standard_rag
from agent_linear import run_linear_agent
from agent_correction import run_correction_agent


def extract_gold_titles(supporting_facts):
    """Extracts unique Wikipedia titles from the HotpotQA supporting_facts list."""
    return list(set([fact[0] for fact in supporting_facts]))


def run_benchmark(input_json: str, output_csv: str):
    print(f"Loading dataset: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Setup CSV Headers
    headers = [
        "question_id", "question", "gold_answer", "gold_titles",
        "rag_answer", "rag_latency", "rag_titles",
        "linear_answer", "linear_latency", "linear_titles",
        "correction_answer", "correction_latency", "correction_titles", "correction_loops"
    ]

    # Write headers if file doesn't exist (allows for safe pausing/resuming)
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()

        for i, item in enumerate(dataset):
            q_id = item['_id']
            question = item['question']
            gold_answer = item['answer']
            gold_titles = extract_gold_titles(item['supporting_facts'])

            print(f"\n==================================================")
            print(f"Test {i + 1}/{len(dataset)} | ID: {q_id}")
            print(f"Question: {question}")
            print(f"==================================================")

            # --- 1. Baseline RAG ---
            print("\n>>> Running Baseline RAG...")
            try:
                rag_ans, rag_lat, rag_tit = run_standard_rag(question, q_id)
            except Exception as e:
                print(f"Error in RAG: {e}")
                rag_ans, rag_lat, rag_tit = "ERROR", 0.0, []

            # --- 2. Linear Agent ---
            print("\n>>> Running Linear Agent...")
            try:
                lin_ans, lin_lat, lin_tit = run_linear_agent(question, q_id)
            except Exception as e:
                print(f"Error in Linear Agent: {e}")
                lin_ans, lin_lat, lin_tit = "ERROR", 0.0, []

            # --- 3. Correction Agent ---
            print("\n>>> Running Correction Agent...")
            try:
                cor_ans, cor_lat, cor_tit, cor_loops = run_correction_agent(question, q_id)
            except Exception as e:
                print(f"Error in Correction Agent: {e}")
                cor_ans, cor_lat, cor_tit, cor_loops = "ERROR", 0.0, [], 0

            # --- Save Iteration to CSV ---
            # We save immediately after each question. If your GPU crashes on question 49,
            # you will not lose the data for the first 48 questions.
            row = {
                "question_id": q_id,
                "question": question,
                "gold_answer": gold_answer,
                "gold_titles": str(gold_titles),

                "rag_answer": rag_ans,
                "rag_latency": round(rag_lat, 2),
                "rag_titles": str(rag_tit),

                "linear_answer": lin_ans,
                "linear_latency": round(lin_lat, 2),
                "linear_titles": str(lin_tit),

                "correction_answer": cor_ans,
                "correction_latency": round(cor_lat, 2),
                "correction_titles": str(cor_tit),
                "correction_loops": cor_loops
            }

            writer.writerow(row)
            f.flush()  # Force write to disk immediately

            print(f"\n✅ Question {q_id} complete and saved to CSV.")

    print(f"\n🎉 Benchmark complete. Results saved to {output_csv}")


if __name__ == "__main__":
    # Point this to the 50-question subset you generated in the previous step
    INPUT_FILE_PATH = "./dataset/benchmark_subset_50.json"
    OUTPUT_FILE_PATH = "./output/raw_benchmark_results_50.csv"

    run_benchmark(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
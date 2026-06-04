import csv
import os
import threading
import concurrent.futures
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

csv.field_size_limit(2**31 - 1)


# --- 1. Define Strict Output Schema ---
class Grade(BaseModel):
    score: int = Field(description="Strictly 1 if correct, 0 if incorrect.")
    reasoning: str = Field(description="A concise 1-sentence explanation for the score.")


PIPELINES = [
    ("rag",            "RAG Baseline"),
    ("linear_v1",      "Linear V1 (Query Decomp)"),
    ("correction_v1",  "Correction V1 (P→P→R)"),
    ("linear_v2",      "Linear V2 (Deep Retriever)"),
    ("correction_v2",  "Correction V2 (Stateful CRAG)"),
]


def grade_answer(chain, question: str, gold_answer: str, generated_answer: str) -> Grade:
    """Execute the LLM grader. Returns a Grade object, defaults to 0 on failure."""
    if generated_answer == "ERROR" or not generated_answer:
        return Grade(score=0, reasoning="Pipeline generation failed or returned empty.")

    try:
        result = chain.invoke({
            "question": question,
            "gold_answer": gold_answer,
            "generated_answer": generated_answer
        })
        return result
    except Exception as e:
        print(f"      [API Error] {e}")
        return Grade(score=0, reasoning="API Call Failed.")


def run_llm_judge(input_csv: str, output_csv: str, api_key: str,
                   model: str = "deepseek-v4-flash",
                   max_workers: int = 3):
    print(f"Loading raw benchmark results from: {input_csv}")

    # --- 2. Initialize the Judge ---
    # Temperature = 0 for deterministic grading.
    # DeepSeek API is OpenAI-compatible → ChatOpenAI with base_url override.
    # `extra_body` disables thinking mode (flash behaviour per
    # https://api-docs.deepseek.com/guides/thinking_mode).
    judge_llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
    )
    structured_judge = judge_llm.with_structured_output(Grade)

    # --- 3. The Zero-Shot Rubric ---
    template = """You are an impartial, strict academic grader evaluating an AI system.
    Compare the Generated Answer against the Gold Answer for the given Question.

    Grading Rules:
    - Score 1 (Correct): The Generated Answer contains the core factual truth of the Gold Answer. It is acceptable if it is more verbose, provided it does not introduce contradictory factual hallucinations.
    - Score 0 (Incorrect): The Generated Answer contradicts the Gold Answer, misses the core fact entirely, or answers a fundamentally different question.

    Question: {question}
    Gold Answer: {gold_answer}
    Generated Answer: {generated_answer}
    """
    prompt = ChatPromptTemplate.from_template(template)
    grading_chain = prompt | structured_judge

    # --- 4. Defensive File Handling ---
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    file_exists = os.path.isfile(output_csv)

    processed_ids = set()
    if file_exists:
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_ids.add(row['question_id'])
        print(f"Found {len(processed_ids)} previously graded questions. Resuming...")

    # Load input data
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    if not all_rows:
        print("No rows found in input CSV.")
        return

    # Build output headers: original columns + 5×2 judging columns
    headers = list(all_rows[0].keys()) + [
        f"{pkg}_llm_score" for pkg, _ in PIPELINES
    ] + [
        f"{pkg}_llm_reasoning" for pkg, _ in PIPELINES
    ]

    # --- 5. Threading Setup ---
    csv_lock = threading.Lock()

    rows_to_process = [row for row in all_rows if row['question_id'] not in processed_ids]
    total_to_process = len(rows_to_process)

    print(f"Starting evaluation with {max_workers} workers for {total_to_process} questions...")
    print(f"Model: {model} | API: https://api.deepseek.com\n")

    def process_row(row):
        q_id = row['question_id']
        question = row['question']
        gold = row['gold_answer']

        # Grade all 5 pipelines independently (one API call each)
        for pkg, display in PIPELINES:
            grade = grade_answer(
                grading_chain, question, gold,
                row.get(f'{pkg}_answer', '')
            )
            row[f'{pkg}_llm_score'] = grade.score
            row[f'{pkg}_llm_reasoning'] = grade.reasoning

        # Compact score summary for logging
        scores = " | ".join(
            f"{display.split()[0]}={row[f'{pkg}_llm_score']}"
            for pkg, display in PIPELINES
        )

        with csv_lock:
            with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerow(row)
            print(f"✅ [Saved] ID: {q_id} | {scores}")

    # --- 6. Execution ---
    if not file_exists:
        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, row) for row in rows_to_process]
        concurrent.futures.wait(futures)

        # Check for exceptions
        for future in futures:
            if future.exception():
                print(f"⚠️  Thread error: {future.exception()}")

    print(f"\n🎉 Evaluation complete. Results saved to {output_csv}")


if __name__ == "__main__":
    API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-api-key-here")

    INPUT_CSV = "./output/raw_benchmark_results_500_llama3.1.csv"
    OUTPUT_CSV = "./output/llm_judged_results_500_llama3.1.csv"

    # max_workers=3 is conservative for DeepSeek's free/standard rate limit.
    # Increase if you have a higher-tier API key.
    run_llm_judge(INPUT_CSV, OUTPUT_CSV, API_KEY, max_workers=3)

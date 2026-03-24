import csv
import os
import sys
import threading
import concurrent.futures
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

csv.field_size_limit(sys.maxsize)


# --- 1. Define Strict Output Schema ---
class Grade(BaseModel):
    score: int = Field(description="Strictly 1 if correct, 0 if incorrect.")
    reasoning: str = Field(description="A concise 1-sentence explanation for the score.")


def grade_answer(chain, question: str, gold_answer: str, generated_answer: str):
    """Executes the LLM grader. Returns a Grade object or defaults to 0 on API failure."""
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


def run_llm_judge(input_csv: str, output_csv: str, api_key: str, max_workers: int = 5):
    print(f"Loading raw benchmark results from: {input_csv}")

    # --- 2. Initialize the Judge ---
    # Temperature MUST be 0 for deterministic grading.
    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key
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

    # Prepare output headers
    headers = list(all_rows[0].keys()) + [
        "rag_llm_score", "rag_llm_reasoning",
        "linear_llm_score", "linear_llm_reasoning",
        "correction_llm_score", "correction_llm_reasoning"
    ]

    # --- 5. Threading Setup ---
    # Create a lock to prevent concurrent writing to the CSV
    csv_lock = threading.Lock()

    # Filter out rows that are already processed
    rows_to_process = [row for row in all_rows if row['question_id'] not in processed_ids]
    total_to_process = len(rows_to_process)

    print(f"Starting multi-threaded evaluation with {max_workers} workers for {total_to_process} questions...\n")

    # Define the worker function that processes a single row
    def process_row(row):
        q_id = row['question_id']
        question = row['question']
        gold = row['gold_answer']

        # I/O Bound API Calls (Executing concurrently across threads)
        rag_grade = grade_answer(grading_chain, question, gold, row['rag_answer'])
        lin_grade = grade_answer(grading_chain, question, gold, row['linear_answer'])
        cor_grade = grade_answer(grading_chain, question, gold, row['correction_answer'])

        # Update row dictionary
        row['rag_llm_score'] = rag_grade.score
        row['rag_llm_reasoning'] = rag_grade.reasoning
        row['linear_llm_score'] = lin_grade.score
        row['linear_llm_reasoning'] = lin_grade.reasoning
        row['correction_llm_score'] = cor_grade.score
        row['correction_llm_reasoning'] = cor_grade.reasoning

        # Thread-safe disk writing
        with csv_lock:
            with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerow(row)
            print(f"✅ [Saved] ID: {q_id} | Scores: RAG={rag_grade.score}, LIN={lin_grade.score}, COR={cor_grade.score}")

    # --- 6. Execution ---
    # Write headers if file is new
    if not file_exists:
        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    # Launch the ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # submit maps the function to the filtered rows
        futures = [executor.submit(process_row, row) for row in rows_to_process]

        # Wait for all to complete
        concurrent.futures.wait(futures)

    print(f"\n🎉 Multi-threaded LLM Evaluation complete. Results saved to {output_csv}")


if __name__ == "__main__":
    API_KEY = "your-api-key-here"

    INPUT_CSV = "./output/raw_benchmark_results_500.csv"
    OUTPUT_CSV = "./output/llm_judged_results_500.csv"

    # Set max_workers based on your Google API Tier limits.
    # 5 is generally safe for free/standard tiers.
    run_llm_judge(INPUT_CSV, OUTPUT_CSV, API_KEY, max_workers=10)

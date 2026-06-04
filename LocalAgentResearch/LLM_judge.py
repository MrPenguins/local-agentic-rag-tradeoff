import csv
import json
import os
import threading
import concurrent.futures
from pydantic import BaseModel, Field
from openai import OpenAI

csv.field_size_limit(2**31 - 1)


# --- 1. Define Output Schema ---
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

JUDGE_SYSTEM_PROMPT = """\
You are an impartial, strict academic grader.
Compare the Generated Answer against the Gold Answer for the given Question.

Grading Rules:
- Score 1 (Correct): the Generated Answer contains the core factual truth of the Gold Answer.
- Score 0 (Incorrect): the Generated Answer contradicts the Gold Answer, misses the core fact, or answers a different question.

Output ONLY a JSON object with these exact keys:
{"score": <0 or 1>, "reasoning": "<one short sentence>"}"""


def grade_answer(
    client: OpenAI,
    model: str,
    question: str,
    gold_answer: str,
    generated_answer: str,
) -> Grade:
    """Call DeepSeek API with JSON mode, parse response into a Grade object."""
    if generated_answer == "ERROR" or not generated_answer:
        return Grade(score=0, reasoning="Pipeline generation failed or returned empty.")

    user_message = (
        f"Question: {question}\n"
        f"Gold Answer: {gold_answer}\n"
        f"Generated Answer: {generated_answer}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)

        score = int(data.get("score", 0))
        reasoning = str(data.get("reasoning", "") or "")

        # Clamp to valid range
        if score not in (0, 1):
            score = 0
            reasoning = f"Invalid score ({score}). Raw: {raw[:80]}"

        return Grade(score=score, reasoning=reasoning)

    except json.JSONDecodeError:
        return Grade(score=0, reasoning=f"JSON parse failed. Raw: {raw[:120]}")
    except Exception as e:
        print(f"      [API Error] {e}")
        return Grade(score=0, reasoning="API Call Failed.")


def run_llm_judge(
    input_csv: str,
    output_csv: str,
    api_key: str,
    model: str = "deepseek-chat",
    max_workers: int = 3,
):
    print(f"Loading raw benchmark results from: {input_csv}")

    # --- 2. Initialize OpenAI client (DeepSeek endpoint) ---
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    # --- 3. Defensive File Handling ---
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    file_exists = os.path.isfile(output_csv)

    processed_ids = set()
    if file_exists:
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_ids.add(row["question_id"])
        print(f"Found {len(processed_ids)} previously graded questions. Resuming...")

    # Load input data
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    if not all_rows:
        print("No rows found in input CSV.")
        return

    # Build output headers: original columns + 5×2 judging columns
    headers = (
        list(all_rows[0].keys())
        + [f"{pkg}_llm_score" for pkg, _ in PIPELINES]
        + [f"{pkg}_llm_reasoning" for pkg, _ in PIPELINES]
    )

    # --- 4. Threading Setup ---
    csv_lock = threading.Lock()

    rows_to_process = [
        row for row in all_rows if row["question_id"] not in processed_ids
    ]
    total_to_process = len(rows_to_process)

    print(
        f"Starting evaluation with {max_workers} workers "
        f"for {total_to_process} questions..."
    )
    print(f"Model: {model} | API: https://api.deepseek.com\n")

    def process_row(row):
        q_id = row["question_id"]
        question = row["question"]
        gold = row["gold_answer"]

        # Grade all 5 pipelines independently (one API call each)
        for pkg, display in PIPELINES:
            grade = grade_answer(
                client, model,
                question, gold,
                row.get(f"{pkg}_answer", ""),
            )
            row[f"{pkg}_llm_score"] = grade.score
            row[f"{pkg}_llm_reasoning"] = grade.reasoning

        # Compact score summary for logging
        scores = " | ".join(
            f"{display.split()[0]}={row[f'{pkg}_llm_score']}"
            for pkg, display in PIPELINES
        )

        with csv_lock:
            with open(output_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerow(row)
            print(f"✅ [Saved] ID: {q_id} | {scores}")

    # --- 5. Execution ---
    if not file_exists:
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, row) for row in rows_to_process]
        concurrent.futures.wait(futures)

        for future in futures:
            if future.exception():
                print(f"⚠️  Thread error: {future.exception()}")

    print(f"\n🎉 Evaluation complete. Results saved to {output_csv}")


if __name__ == "__main__":
    API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-api-key-here")

    INPUT_CSV = "./output/raw_benchmark_results_500_llama3.1.csv"
    OUTPUT_CSV = "./output/llm_judged_results_500_llama3.1.csv"

    run_llm_judge(INPUT_CSV, OUTPUT_CSV, API_KEY, max_workers=3)

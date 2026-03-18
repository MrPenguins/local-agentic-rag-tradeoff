import csv
import os
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# --- 1. Define Strict Output Schema ---
# This forces the API to return a predictable JSON object instead of raw text.
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


def run_llm_judge(input_csv: str, output_csv: str, api_key: str):
    print(f"Loading raw benchmark results from: {input_csv}")

    # --- 2. Initialize the Judge ---
    # Temperature MUST be 0 for deterministic grading.
    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Highly recommended for fast, cheap, structured classification
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

    # --- 5. The Execution Loop ---
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()

        for i, row in enumerate(all_rows):
            q_id = row['question_id']
            if q_id in processed_ids:
                continue

            print(f"\nEvaluating {i + 1}/{len(all_rows)} | ID: {q_id}")
            question = row['question']
            gold = row['gold_answer']

            # Grade Baseline
            print("   -> Grading RAG Baseline...")
            rag_grade = grade_answer(grading_chain, question, gold, row['rag_answer'])

            # Grade Linear
            print("   -> Grading Linear Agent...")
            lin_grade = grade_answer(grading_chain, question, gold, row['linear_answer'])

            # Grade Correction
            print("   -> Grading Correction Agent...")
            cor_grade = grade_answer(grading_chain, question, gold, row['correction_answer'])

            # Update row dictionary with new grading data
            row['rag_llm_score'] = rag_grade.score
            row['rag_llm_reasoning'] = rag_grade.reasoning
            row['linear_llm_score'] = lin_grade.score
            row['linear_llm_reasoning'] = lin_grade.reasoning
            row['correction_llm_score'] = cor_grade.score
            row['correction_llm_reasoning'] = cor_grade.reasoning

            writer.writerow(row)
            f.flush()
            print(f"   [Saved] Scores: RAG={rag_grade.score}, LIN={lin_grade.score}, COR={cor_grade.score}")

    print(f"\n🎉 LLM Evaluation complete. Results saved to {output_csv}")


if __name__ == "__main__":
    API_KEY = "your-api-key-here"

    INPUT_CSV = "./output/raw_benchmark_results_500.csv"
    OUTPUT_CSV = "./output/llm_judged_results_500.csv"

    run_llm_judge(INPUT_CSV, OUTPUT_CSV, API_KEY)
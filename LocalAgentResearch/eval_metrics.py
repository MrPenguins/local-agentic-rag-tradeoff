import csv
import ast
import re
import string
import sys
from collections import Counter

csv.field_size_limit(sys.maxsize)


# --- NLP Normalization Functions ---
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def exact_match_score(prediction, ground_truth):
    """Calculates Exact Match."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    """Calculates Token-level F1 Score."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def calculate_p90(data):
    """Calculates the 90th percentile latency."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(0.9 * len(sorted_data))
    return sorted_data[index]


# --- Main Evaluation Logic ---
def evaluate_metrics(csv_filepath: str):
    print(f"Loading results from {csv_filepath}...\n")

    pipelines = ["rag", "linear", "correction"]
    metrics = {
        p: {
            "total_hits": 0,
            "distribution": {0: 0, 1: 0, 2: 0, "2+": 0},
            "total_em": 0.0,
            "total_f1": 0.0,
            "latencies": [],
            "ttfts": [],
            "loops": []
        } for p in pipelines
    }

    total_gold_documents = 0
    total_questions = 0

    with open(csv_filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                gold_titles = set(ast.literal_eval(row['gold_titles']))
                rag_titles = set(ast.literal_eval(row['rag_titles']))
                linear_titles = set(ast.literal_eval(row['linear_titles']))
                correction_titles = set(ast.literal_eval(row['correction_titles']))
            except (ValueError, SyntaxError) as e:
                print(f"⚠️ Skipping row {row.get('question_id', 'Unknown')} due to parsing error.")
                continue

            total_questions += 1
            total_gold_documents += len(gold_titles)
            gold_answer = row['gold_answer']

            # Map pipeline data
            pipeline_data = {
                "rag": {
                    "titles": rag_titles, "answer": row['rag_answer'],
                    "lat": float(row['rag_latency']), "ttft": float(row.get('rag_ttft', 0.0))
                },
                "linear": {
                    "titles": linear_titles, "answer": row['linear_answer'],
                    "lat": float(row['linear_latency']), "ttft": float(row.get('linear_ttft', 0.0))
                },
                "correction": {
                    "titles": correction_titles, "answer": row['correction_answer'],
                    "lat": float(row['correction_latency']), "ttft": float(row.get('correction_ttft', 0.0)),
                    "loops": int(row.get('correction_loops', 1))
                }
            }

            # Calculate metrics for each pipeline
            for p_name, p_data in pipeline_data.items():
                # 1. Retrieval Metrics
                matches = len(gold_titles.intersection(p_data["titles"]))
                metrics[p_name]["total_hits"] += matches

                if matches == 0:
                    metrics[p_name]["distribution"][0] += 1
                elif matches == 1:
                    metrics[p_name]["distribution"][1] += 1
                elif matches == 2:
                    metrics[p_name]["distribution"][2] += 1
                else:
                    metrics[p_name]["distribution"]["2+"] += 1

                # 2. Generation Metrics (EM & F1)
                pred_answer = str(p_data["answer"])
                metrics[p_name]["total_em"] += exact_match_score(pred_answer, gold_answer)
                metrics[p_name]["total_f1"] += f1_score(pred_answer, gold_answer)

                # 3. Latency & Loop Metrics
                metrics[p_name]["latencies"].append(p_data["lat"])
                metrics[p_name]["ttfts"].append(p_data["ttft"])
                if p_name == "correction":
                    metrics[p_name]["loops"].append(p_data["loops"])

    if total_questions == 0:
        print("No valid rows found in CSV.")
        return

    # --- Print the Final Report ---
    print("==================================================")
    print("📊 FULL PIPELINE PERFORMANCE REPORT")
    print("==================================================")
    print(f"Total Questions Evaluated: {total_questions}")
    print(f"Total Gold Documents: {total_gold_documents}\n")

    for p in pipelines:
        hits = metrics[p]["total_hits"]
        dist = metrics[p]["distribution"]
        avg_em = (metrics[p]["total_em"] / total_questions) * 100
        avg_f1 = (metrics[p]["total_f1"] / total_questions) * 100
        recall_rate = (hits / total_gold_documents) * 100 if total_gold_documents > 0 else 0

        avg_lat = sum(metrics[p]["latencies"]) / total_questions
        p90_lat = calculate_p90(metrics[p]["latencies"])

        avg_ttft = sum(metrics[p]["ttfts"]) / total_questions
        p90_ttft = calculate_p90(metrics[p]["ttfts"])

        print(f"--- {p.upper()} PIPELINE ---")
        print(f"Context Recall Rate:         {recall_rate:.2f}% ({hits}/{total_gold_documents})")
        print(f"  • 0 Correct Docs: {dist[0]} | 1 Correct: {dist[1]} | 2 Correct: {dist[2]}")
        print(f"Generation Exact Match (EM): {avg_em:.2f}%")
        print(f"Generation F1 Score:         {avg_f1:.2f}%")
        print(f"Avg TTFT:                    {avg_ttft:.2f}s")
        print(f"P90 TTFT:                    {p90_ttft:.2f}s")
        print(f"Avg Total Latency:           {avg_lat:.2f}s")
        print(f"P90 Total Latency:           {p90_lat:.2f}s")
        if p == "correction":
            avg_loops = sum(metrics[p]["loops"]) / total_questions
            print(f"Avg Reasoning Loops:         {avg_loops:.2f}")
        print()

    # --- Print the Cost of Reasoning Trade-off ---
    print("==================================================")
    print("⚖️ THE COST OF REASONING: TRADE-OFF ANALYSIS")
    print("==================================================")

    rag_f1 = (metrics["rag"]["total_f1"] / total_questions) * 100
    rag_lat = sum(metrics["rag"]["latencies"]) / total_questions
    rag_ttft = sum(metrics["rag"]["ttfts"]) / total_questions

    for agent in ["linear", "correction"]:
        agent_f1 = (metrics[agent]["total_f1"] / total_questions) * 100
        agent_lat = sum(metrics[agent]["latencies"]) / total_questions
        agent_ttft = sum(metrics[agent]["ttfts"]) / total_questions

        f1_gain = agent_f1 - rag_f1
        lat_tax = agent_lat - rag_lat
        ttft_tax = agent_ttft - rag_ttft

        print(f"--- {agent.upper()} vs BASELINE RAG ---")
        print(f"F1 Score Gain:      +{f1_gain:.2f}%")
        print(f"TTFT Tax:           +{ttft_tax:.2f} seconds wait before first word")
        print(f"Total Latency Tax:  +{lat_tax:.2f} seconds wait before final word")

        if lat_tax > 0:
            efficiency = f1_gain / lat_tax
            print(f"Efficiency:         +{efficiency:.2f}% F1 per additional second of Total Latency")
        else:
            print("Efficiency:         Faster than baseline! (Anomalous)")
        print()


if __name__ == "__main__":
    CSV_FILE = "./output/raw_benchmark_results_5.csv"
    evaluate_metrics(CSV_FILE)
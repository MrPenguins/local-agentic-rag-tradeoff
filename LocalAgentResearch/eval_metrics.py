import csv
import ast
import re
import string
from collections import Counter

csv.field_size_limit(2 ** 31 - 1)  # Safe max for C long on all platforms


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


def calculate_percentile(data, percentile):
    """Calculates a given percentile from a list of numbers."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(percentile * len(sorted_data))
    index = min(index, len(sorted_data) - 1)
    return sorted_data[index]


def calculate_median(data):
    """Calculates the median of a list of numbers."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2.0
    return sorted_data[mid]


# --- Pipeline Configuration ---
# Maps internal pipeline name → CSV column prefix and display name
PIPELINE_CONFIG = {
    "rag": {
        "prefix": "rag",
        "display": "BASELINE RAG",
        "has_loops": False,
    },
    "linear_v1": {
        "prefix": "linear_v1",
        "display": "LINEAR V1 (Query Decomposition)",
        "has_loops": False,
    },
    "correction_v1": {
        "prefix": "correction_v1",
        "display": "CORRECTION V1 (Planner → Performer → Reviewer)",
        "has_loops": True,
    },
    "linear_v2": {
        "prefix": "linear_v2",
        "display": "LINEAR V2 (Deep Retriever → Pointwise Filter)",
        "has_loops": False,
    },
    "correction_v2": {
        "prefix": "correction_v2",
        "display": "CORRECTION V2 (Stateful CRAG)",
        "has_loops": True,
    },
}


# --- Main Evaluation Logic ---
def evaluate_metrics(csv_filepath: str):
    print(f"Loading results from {csv_filepath}...\n")

    pipelines = list(PIPELINE_CONFIG.keys())
    metrics = {
        p: {
            "total_hits": 0,
            "distribution": {0: 0, 1: 0, 2: 0},
            "total_em": 0.0,
            "total_f1": 0.0,
            "latencies": [],
            "ttfts": [],
            "loops": [],
            "errors": 0,
        } for p in pipelines
    }

    total_gold_documents = 0
    total_questions = 0

    with open(csv_filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                gold_titles = set(ast.literal_eval(row['gold_titles']))
            except (ValueError, SyntaxError):
                print(f"⚠️ Skipping row {row.get('question_id', 'Unknown')} — gold_titles parse error.")
                continue

            total_questions += 1
            total_gold_documents += len(gold_titles)
            gold_answer = row['gold_answer']

            for p_name, p_cfg in PIPELINE_CONFIG.items():
                prefix = p_cfg["prefix"]

                # Parse titles
                try:
                    p_titles = set(ast.literal_eval(row[f'{prefix}_titles']))
                except (ValueError, SyntaxError, KeyError):
                    p_titles = set()

                p_answer = str(row.get(f'{prefix}_answer', ''))
                p_latency = float(row.get(f'{prefix}_latency', 0.0))
                p_ttft = float(row.get(f'{prefix}_ttft', 0.0))

                # Track errors
                if p_answer == "ERROR":
                    metrics[p_name]["errors"] += 1
                    continue

                # 1. Retrieval Metrics
                matches = len(gold_titles.intersection(p_titles))
                metrics[p_name]["total_hits"] += matches

                if matches == 0:
                    metrics[p_name]["distribution"][0] += 1
                elif matches == 1:
                    metrics[p_name]["distribution"][1] += 1
                elif matches == 2:
                    metrics[p_name]["distribution"][2] += 1

                # 2. Generation Metrics (EM & F1)
                metrics[p_name]["total_em"] += exact_match_score(p_answer, gold_answer)
                metrics[p_name]["total_f1"] += f1_score(p_answer, gold_answer)

                # 3. Latency & Loop Metrics
                metrics[p_name]["latencies"].append(p_latency)
                metrics[p_name]["ttfts"].append(p_ttft)

                if p_cfg["has_loops"]:
                    p_loops = int(row.get(f'{prefix}_loops', 1))
                    metrics[p_name]["loops"].append(p_loops)

    if total_questions == 0:
        print("No valid rows found in CSV.")
        return

    # --- Print the Full Report ---
    print("=" * 60)
    print("📊 FULL PIPELINE PERFORMANCE REPORT (5 Pipelines)")
    print("=" * 60)
    print(f"Total Questions Evaluated: {total_questions}")
    print(f"Total Gold Documents: {total_gold_documents}\n")

    for p in pipelines:
        cfg = PIPELINE_CONFIG[p]
        m = metrics[p]
        valid = total_questions - m["errors"]

        if valid == 0:
            print(f"--- {cfg['display']} --- (ALL ERRORS, skipping)\n")
            continue

        hits = m["total_hits"]
        dist = m["distribution"]
        avg_em = (m["total_em"] / valid) * 100
        avg_f1 = (m["total_f1"] / valid) * 100
        recall_rate = (hits / total_gold_documents) * 100 if total_gold_documents > 0 else 0

        avg_lat = sum(m["latencies"]) / valid
        median_lat = calculate_median(m["latencies"])
        p90_lat = calculate_percentile(m["latencies"], 0.90)

        avg_ttft = sum(m["ttfts"]) / valid
        median_ttft = calculate_median(m["ttfts"])
        p90_ttft = calculate_percentile(m["ttfts"], 0.90)

        print(f"--- {cfg['display']} ---")
        if m["errors"] > 0:
            print(f"⚠️  Errors (excluded):        {m['errors']}/{total_questions}")
        print(f"Context Recall Rate:         {recall_rate:.2f}% ({hits}/{total_gold_documents})")
        print(f"  • 0 Correct Docs: {dist[0]} | 1 Correct: {dist[1]} | 2 Correct: {dist[2]}")
        print(f"Generation Exact Match (EM): {avg_em:.2f}%")
        print(f"Generation F1 Score:         {avg_f1:.2f}%")
        print(f"Avg TTFT:                    {avg_ttft:.2f}s  (Median: {median_ttft:.2f}s, P90: {p90_ttft:.2f}s)")
        print(f"Avg Total Latency:           {avg_lat:.2f}s  (Median: {median_lat:.2f}s, P90: {p90_lat:.2f}s)")
        if cfg["has_loops"]:
            avg_loops = sum(m["loops"]) / valid
            print(f"Avg Reasoning Loops:         {avg_loops:.2f}")
        print()

    # --- Trade-off Analysis ---
    print("=" * 60)
    print("⚖️ THE COST OF REASONING: TRADE-OFF ANALYSIS (vs Baseline RAG)")
    print("=" * 60)

    rag_m = metrics["rag"]
    rag_valid = total_questions - rag_m["errors"]
    if rag_valid == 0:
        print("Baseline RAG had no valid results. Cannot compute trade-off.")
        return

    rag_f1 = (rag_m["total_f1"] / rag_valid) * 100
    rag_em = (rag_m["total_em"] / rag_valid) * 100
    rag_lat = sum(rag_m["latencies"]) / rag_valid
    rag_ttft = sum(rag_m["ttfts"]) / rag_valid
    rag_median_lat = calculate_median(rag_m["latencies"])

    for agent in ["linear_v1", "correction_v1", "linear_v2", "correction_v2"]:
        cfg = PIPELINE_CONFIG[agent]
        am = metrics[agent]
        a_valid = total_questions - am["errors"]
        if a_valid == 0:
            print(f"\n--- {cfg['display']} vs BASELINE RAG --- (ALL ERRORS, skipping)")
            continue

        agent_f1 = (am["total_f1"] / a_valid) * 100
        agent_em = (am["total_em"] / a_valid) * 100
        agent_lat = sum(am["latencies"]) / a_valid
        agent_ttft = sum(am["ttfts"]) / a_valid
        agent_median_lat = calculate_median(am["latencies"])

        f1_gain = agent_f1 - rag_f1
        em_gain = agent_em - rag_em
        lat_tax = agent_lat - rag_lat
        ttft_tax = agent_ttft - rag_ttft
        median_lat_tax = agent_median_lat - rag_median_lat

        print(f"\n--- {cfg['display']} vs BASELINE RAG ---")
        print(f"EM Gain:                 {em_gain:+.2f}%")
        print(f"F1 Score Gain:           {f1_gain:+.2f}%")
        print(f"TTFT Tax:                {ttft_tax:+.2f}s (wait before first word)")
        print(f"Total Latency Tax (Avg): {lat_tax:+.2f}s")
        print(f"Total Latency Tax (Med): {median_lat_tax:+.2f}s")

        if lat_tax > 0:
            efficiency = f1_gain / lat_tax
            print(f"Efficiency:              {efficiency:+.4f}% F1 per additional second")
        elif lat_tax < 0:
            print(f"Efficiency:              Faster than baseline! ({f1_gain:+.2f}% F1 with {lat_tax:.2f}s less)")
        else:
            print(f"Efficiency:              Same latency as baseline")

    print()


if __name__ == "__main__":
    CSV_FILE = "./output/raw_benchmark_results_500_llama3.1.csv"
    evaluate_metrics(CSV_FILE)

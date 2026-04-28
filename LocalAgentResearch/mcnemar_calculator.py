import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import os

# 1. Define the input file path
input_csv = "./output/llm_judged_results_500_merged.csv"


def run_mcnemar_tests(csv_filepath):
    print(f"Loading data from {csv_filepath}...\n")
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(
            f"Cannot find {csv_filepath}. Please ensure the previous merge script completed successfully.")

    df = pd.read_csv(csv_filepath)

    # 2. Define baseline
    baseline_col = "rag_llm_score"
    if baseline_col not in df.columns:
        raise ValueError(f"Baseline column '{baseline_col}' not found in the dataset.")

    # 3. Find all other columns representing LLM scores
    target_cols = [col for col in df.columns if 'llm_score' in col and col != baseline_col]

    if not target_cols:
        print("No other 'llm_score' columns found to compare against.")
        return

    print("============================================================")
    print("📊 MCNEMAR'S TEST RESULTS (Baseline: rag_llm_score)")
    print("============================================================\n")

    # 4. Calculate McNemar's test for each target column
    for target in target_cols:
        # Drop rows where either the baseline or the target score is missing (NaN)
        # This ensures we are only doing paired comparisons
        paired_df = df[[baseline_col, target]].dropna().copy()

        # Ensure values are treated as integers (Assuming 1 = Correct, 0 = Incorrect)
        b = paired_df[baseline_col].astype(int)
        t = paired_df[target].astype(int)

        # Build the 2x2 contingency table:
        # [ [Both Correct (1,1),                   Baseline Correct & Target Incorrect (1,0)]
        #   [Baseline Incorrect & Target Correct (0,1), Both Incorrect (0,0)] ]
        b1_t1 = len(paired_df[(b == 1) & (t == 1)])
        b1_t0 = len(paired_df[(b == 1) & (t == 0)])
        b0_t1 = len(paired_df[(b == 0) & (t == 1)])
        b0_t0 = len(paired_df[(b == 0) & (t == 0)])

        table = [[b1_t1, b1_t0],
                 [b0_t1, b0_t0]]

        # The test relies on discordant pairs (cases where the two models disagreed)
        discordant_pairs = b1_t0 + b0_t1

        # Standard practice: use the exact binomial test for small sample sizes (<25 discordant pairs)
        # and the Chi-square approximation with continuity correction for larger samples.
        use_exact = True if discordant_pairs < 25 else False

        # Calculate p-value
        result = mcnemar(table, exact=use_exact, correction=True)

        # Determine significance (alpha = 0.05)
        significance = "Significant (Reject Null Hypothesis)" if result.pvalue < 0.05 else "Not Significant"

        print(f"Comparison: {baseline_col} vs {target}")
        print(f"  Valid Paired Samples: {len(paired_df)}")
        print(f"  Discordant Pairs:     {discordant_pairs}")
        print(f"    - {baseline_col} correct, {target} incorrect: {b1_t0}")
        print(f"    - {baseline_col} incorrect, {target} correct: {b0_t1}")
        print(f"  p-value:              {result.pvalue:.5f} ({significance})")
        print("-" * 60)


if __name__ == "__main__":
    run_mcnemar_tests(input_csv)
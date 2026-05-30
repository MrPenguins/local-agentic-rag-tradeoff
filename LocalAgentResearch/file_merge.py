import pandas as pd
import os

# 1. Define file paths
file1_path = "./output/llm_judged_results_500.csv"
file2_path = "./output/llm_judged_results_500_2.csv"
output_path = "./output/llm_judged_results_500_merged.csv"

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 2. Read the CSV files
print(f"Loading {file1_path}...")
df1 = pd.read_csv(file1_path)
print(f"Loading {file2_path}...")
df2 = pd.read_csv(file2_path)

# 3. Check that 'question_id' exists in both
if 'question_id' not in df1.columns or 'question_id' not in df2.columns:
    raise ValueError("Column 'question_id' must exist in both CSV files to perform the merge.")

# 4. Report any mismatches in question_id
ids_df1 = set(df1['question_id'].astype(str).str.strip())
ids_df2 = set(df2['question_id'].astype(str).str.strip())

missing_in_df2 = ids_df1 - ids_df2
missing_in_df1 = ids_df2 - ids_df1

if not missing_in_df2 and not missing_in_df1:
    print("✅ 100% Match: All question_ids match perfectly between the two files.")
else:
    print("⚠️ MISMATCH DETECTED!")
    if missing_in_df2:
        print(f" - {len(missing_in_df2)} IDs found in File 1 but missing in File 2.")
        print(f"   Examples: {list(missing_in_df2)[:5]}")
    if missing_in_df1:
        print(f" - {len(missing_in_df1)} IDs found in File 2 but missing in File 1.")
        print(f"   Examples: {list(missing_in_df1)[:5]}")

# 5. Locate the "linear_answer" column in the second file
if "linear_answer" not in df2.columns:
    raise ValueError("Column 'linear_answer' not found in the second CSV.")

start_col_idx = df2.columns.get_loc("linear_answer")

# 6. Extract the target columns from the second file
cols_to_add = df2.columns[start_col_idx:]

# Keep 'question_id' for merging, alongside the target columns
df2_subset = df2[['question_id'] + list(cols_to_add)].copy()

# 7. Append '_v2' ONLY to the selected column names (excluding 'question_id')
rename_dict = {col: f"{col}_v2" for col in cols_to_add}
df2_subset.rename(columns=rename_dict, inplace=True)

# 8. Merge the dataframes on 'question_id'
print("Merging files on 'question_id'...")
merged_df = pd.merge(df1, df2_subset, on='question_id', how='outer')

# 9. Drop specific unwanted columns from the final merged dataframe
cols_to_drop = ['rag_llm_score_v2', 'rag_llm_reasoning_v2']
existing_cols_to_drop = [col for col in cols_to_drop if col in merged_df.columns]

if existing_cols_to_drop:
    merged_df.drop(columns=existing_cols_to_drop, inplace=True)
    print(f"Dropped columns: {existing_cols_to_drop}")

# 10. Save to a new file
merged_df.to_csv(output_path, index=False)
print(f"Successfully saved merged results to {output_path}")
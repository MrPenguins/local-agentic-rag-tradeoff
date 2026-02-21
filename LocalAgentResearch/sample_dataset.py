import json
import random


def create_benchmark_subset(input_path: str, output_path: str, n_samples: int):
    print(f"Loading clean dataset from '{input_path}'...")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_questions = len(data)
    print(f"Found {total_questions} total questions.")

    # Randomly sample N distinct questions
    print(f"Randomly selecting {n_samples} questions...")
    sampled_data = random.sample(data, n_samples)

    # Save the new subset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=2)

    print(f"✅ Successfully saved {n_samples} questions to '{output_path}'.")


if __name__ == "__main__":
    INPUT_FILE_PATH = "./dataset/clean_benchmark.json"
    N = 5
    OUTPUT_FILE_PATH = f"./dataset/benchmark_subset_{N}.json"

    # Run the function
    create_benchmark_subset(INPUT_FILE_PATH, OUTPUT_FILE_PATH, N)
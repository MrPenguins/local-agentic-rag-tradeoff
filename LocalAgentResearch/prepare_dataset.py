import os
import requests

# --- Configuration ---
DATASET_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
RAW_FILE = "./dataset/hotpot_dev_distractor_v1.json"
OUTPUT_DIR = "./data"


def download_dataset():
    directory = os.path.dirname(RAW_FILE)

    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if not os.path.exists(RAW_FILE):
        print(f"Downloading {RAW_FILE}...")
        response = requests.get(DATASET_URL)
        response.raise_for_status()

        with open(RAW_FILE, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print(f"Found {RAW_FILE} locally.")


if __name__ == "__main__":
    download_dataset()

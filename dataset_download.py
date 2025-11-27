import os
from datasets import load_dataset

RAW_DIR = "data/raw"

def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)

def download_codexglue_defect_dataset():
    print("Loading CodeXGLUE Defect Detection dataset (C language only)...")

    dataset = load_dataset("google/code_x_glue_cc_defect_detection")

    print("Saving dataset into data/raw/...")

    for split in ["train", "validation", "test"]:
        path = os.path.join(RAW_DIR, f"{split}.jsonl")
        dataset[split].to_json(path, orient="records", lines=True)
        print(f"Saved {split} → {path}")

    print("\nDataset download complete.")

if __name__ == "__main__":
    ensure_dirs()
    download_codexglue_defect_dataset()

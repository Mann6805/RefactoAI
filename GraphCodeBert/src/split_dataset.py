# src/split_dataset.py

import json
import os

# Paths
INPUT_FILE = "data/raw/code_defect_dataset.jsonl"
TRAIN_FILE = "data/processed/train.jsonl"
TEST_FILE = "data/processed/test.jsonl"

# Ensure output directory exists
os.makedirs(os.path.dirname(TRAIN_FILE), exist_ok=True)
os.makedirs(os.path.dirname(TEST_FILE), exist_ok=True)

# Open input file and prepare lists
train_data = []
test_data = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        sample = json.loads(line)
        split = sample.get("split", "train").lower()
        if split == "train":
            train_data.append(sample)
        elif split == "test":
            test_data.append(sample)
        else:
            print(f"Warning: Unknown split '{split}' in sample {sample.get('id', 'unknown')}")

# Write train file
with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    for sample in train_data:
        f.write(json.dumps(sample) + "\n")

# Write test file
with open(TEST_FILE, "w", encoding="utf-8") as f:
    for sample in test_data:
        f.write(json.dumps(sample) + "\n")

print(f"Done! {len(train_data)} training samples and {len(test_data)} testing samples saved.")
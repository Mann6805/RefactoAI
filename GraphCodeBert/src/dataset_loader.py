# src/dataset_loader.py

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# ----------------------------
# Configuration
# ----------------------------
TOKENIZER_NAME = "microsoft/graphcodebert-base"
MAX_LEN = 256  # max tokens for code snippet
LANGUAGES = ["Python", "Java"]  # Languages to include
DEFECT_TYPES = [
    "resource_leak",
    "null_pointer_dereference",
    "concurrency_issue",
    "security_vulnerability",
    "code_complexity"
]

# ----------------------------
# PyTorch Dataset Class
# ----------------------------
class CodeDefectDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len=MAX_LEN):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        code = item["code"]
        # Encode labels as multi-hot vector
        labels = [1 if item["defect_type"] == dt else 0 for dt in DEFECT_TYPES]

        # Tokenize code
        encoding = self.tokenizer(
            code,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float)
        }

# ----------------------------
# Load JSONL and prepare dataset
# ----------------------------
def load_jsonl(file_path, languages=LANGUAGES):
    """
    Load a .jsonl file and filter by language
    """
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample.get("language") in languages:
                data_list.append(sample)
    return data_list

# ----------------------------
# Prepare PyTorch Dataset
# ----------------------------
def get_datasets(train_file, test_file, tokenizer_name=TOKENIZER_NAME, max_len=MAX_LEN):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_data = load_jsonl(train_file)
    test_data = load_jsonl(test_file)

    train_dataset = CodeDefectDataset(train_data, tokenizer, max_len)
    test_dataset = CodeDefectDataset(test_data, tokenizer, max_len)

    return train_dataset, test_dataset

if __name__ == "__main__":
    train_file = "data/processed/train.jsonl"
    test_file = "data/processed/test.jsonl"

    train_dataset, test_dataset = get_datasets(train_file, test_file)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    # Inspect first sample
    sample = train_dataset[0]
    print("Input IDs shape:", sample["input_ids"].shape)
    print("Attention mask shape:", sample["attention_mask"].shape)
    print("Labels:", sample["labels"])
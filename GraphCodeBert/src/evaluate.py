# src/evaluate.py

import torch
from transformers import AutoModelForSequenceClassification
from dataset_loader import get_datasets, DEFECT_TYPES
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
import numpy as np

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = "models/graphcodebert_finetuned/"
TEST_FILE = "data/processed/test.jsonl"
MAX_LEN = 256
BATCH_SIZE = 8

# ----------------------------
# Custom collate function
# ----------------------------
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# ----------------------------
# Load model and dataset
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

_, test_dataset = get_datasets(train_file="data/processed/train.jsonl",
                               test_file=TEST_FILE,
                               max_len=MAX_LEN)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ----------------------------
# Evaluate
# ----------------------------
all_labels = []
all_preds = []

sigmoid = torch.nn.Sigmoid()

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].cpu().numpy()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        all_labels.append(labels)
        all_preds.append(preds)

# Concatenate all batches
all_labels = np.vstack(all_labels)
all_preds = np.vstack(all_preds)

# Compute metrics
precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

print("Evaluation on test set:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

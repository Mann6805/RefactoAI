import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from ..utils.config import config

class DefectDataset(Dataset):
    def __init__(self, path, max_length=512, max_samples=None):
        self.data = [json.loads(line) for line in open(path, "r")]
        if max_samples:
            self.data = self.data[:max_samples]
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = self.tokenizer(
            item["code"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long)
        }

def create_dataloader(path, batch_size, max_samples=None):
    ds = DefectDataset(path, max_length=config.MAX_SEQ_LENGTH, max_samples=max_samples)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model.graphcoderbert_lora import GraphCodeBERT_LoRA
from ..dataset.data_loader import create_dataloader
from ..utils.config import config
from ..utils.metrics import compute_metrics

def evaluate_on_test(model_path):
    # Load model
    model = GraphCodeBERT_LoRA(num_labels=2, r=config.LORA_R, alpha=config.LORA_ALPHA, dropout=config.LORA_DROPOUT)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    # Create test dataloader
    test_loader = create_dataloader(
        os.path.join(config.PROCESSED_DATA_PATH, "test.jsonl"),
        batch_size=config.BATCH_SIZE
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            labels = batch["label"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    metrics = compute_metrics(all_preds, all_labels)
    print("Test set evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

if __name__ == "__main__":
    model_path = os.path.join(config.MODEL_SAVE_DIR, "graphcodebert_lora.pt")
    evaluate_on_test(model_path)
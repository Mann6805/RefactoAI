import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from .model.graphcoderbert_lora import GraphCodeBERT_LoRA
from .dataset.data_loader import create_dataloader
from .utils.config import config
from .utils.metrics import compute_metrics

def train():
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Dataloaders
    train_loader = create_dataloader(
        os.path.join(config.PROCESSED_DATA_PATH, "train.jsonl"),
        batch_size=config.BATCH_SIZE,
        max_samples=1000
    )
    val_loader = create_dataloader(
        os.path.join(config.PROCESSED_DATA_PATH, "validation.jsonl"),
        batch_size=config.BATCH_SIZE,
        max_samples=200
    )

    # print("Running pre-training dataset check...")

    # label_counts = {}
    # total_samples = 0

    # for batch in train_loader:
    #     labels = batch["label"]
    #     if isinstance(labels, torch.Tensor):
    #         labels = labels.cpu().numpy()
    #     for l in labels:
    #         label_counts[l] = label_counts.get(l, 0) + 1
    #         total_samples += 1

    # print(f"Total training samples: {total_samples}")
    # print("Label distribution:")
    # for lbl, count in label_counts.items():
    #     print(f"  Label {lbl}: {count} samples")

    # unique_labels = list(label_counts.keys())
    # if not all(l in [0, 1] for l in unique_labels):
    #     raise ValueError(f"Unexpected labels found: {unique_labels}")

    # print("Dataset check passed")


    model = GraphCodeBERT_LoRA(
        num_labels=2,
        r=config.LORA_R,
        alpha=config.LORA_ALPHA,
        dropout=config.LORA_DROPOUT
    ).to(config.DEVICE)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            labels = batch["label"].to(config.DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})

        print(f"\nEpoch {epoch+1} completed. Avg training loss: {total_loss/len(train_loader):.4f}")
        evaluate(model, val_loader)

    save_path = os.path.join(config.MODEL_SAVE_DIR, "graphcodebert_lora.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")


def evaluate(model, dataloader):
    model.eval()
    preds, labels_list = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            labels = batch["label"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).cpu().numpy()

            preds.extend(pred)
            labels_list.extend(labels)

    metrics = compute_metrics(preds, labels_list)
    print(f"Validation → Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    model.train()


if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    train()
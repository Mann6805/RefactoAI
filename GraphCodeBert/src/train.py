# src/train.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from dataset_loader import get_datasets, DEFECT_TYPES
from sklearn.metrics import f1_score, precision_score, recall_score

# ----------------------------
# Configuration
# ----------------------------
MODEL_NAME = "microsoft/graphcodebert-base"
TRAIN_FILE = "data/processed/train.jsonl"
TEST_FILE = "data/processed/test.jsonl"
OUTPUT_DIR = "models/graphcodebert_finetuned/"
MAX_LEN = 256

# ----------------------------
# Metrics for multi-label classification
# ----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.tensor(logits))
    y_pred = (probs >= 0.5).int().numpy()
    y_true = labels
    return {
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

# ----------------------------
# Load Datasets
# ----------------------------
train_dataset, test_dataset = get_datasets(TRAIN_FILE, TEST_FILE, max_len=MAX_LEN)
print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

# ----------------------------
# Custom collate function
# ----------------------------
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# ----------------------------
# Hyperparameter search function
# ----------------------------
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(DEFECT_TYPES),
        problem_type="multi_label_classification"
    )

def hyperparameter_search():
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="eval_f1",
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )

    # Define search space for Optuna
    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 7),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1)
        }

    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        n_trials=5,  # adjust based on your time/GPU
        hp_space=hp_space,
        compute_objective=lambda metrics: metrics["eval_f1"]
    )

    print("Best hyperparameters found:", best_run)

    # Update training args with best hyperparameters
    training_args.learning_rate = best_run.hyperparameters["learning_rate"]
    training_args.per_device_train_batch_size = best_run.hyperparameters["per_device_train_batch_size"]
    training_args.num_train_epochs = best_run.hyperparameters["num_train_epochs"]
    training_args.weight_decay = best_run.hyperparameters["weight_decay"]

    # Final trainer with best hyperparameters
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )

    # Train final model
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Fine-tuned model saved to {OUTPUT_DIR}")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    hyperparameter_search()

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    precision = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

import json
import os
from tqdm import tqdm
from ..utils.config import config

def preprocess_dataset():
    os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)

    for split in ["train", "validation", "test"]:
        raw_path = os.path.join(config.RAW_DATA_PATH, f"{split}.jsonl")
        processed_path = os.path.join(config.PROCESSED_DATA_PATH, f"{split}.jsonl")

        with open(raw_path, "r") as f:
            raw_data = [json.loads(line) for line in f]

        processed = [{"code": item["func"], "label": item["target"]} for item in tqdm(raw_data, desc=f"Preprocessing {split}")]

        with open(processed_path, "w") as f:
            for ex in processed:
                f.write(json.dumps(ex) + "\n")

        print(f"✔ Saved processed {split} → {processed_path}")

if __name__ == "__main__":
    preprocess_dataset()
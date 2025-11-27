import torch
import os

class Config:
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Seed for reproducibility
    SEED = 42

    # Dataset paths
    RAW_DATA_PATH = "./data/raw"
    PROCESSED_DATA_PATH = "./data/processed"

    # Training hyperparameters
    NUM_EPOCHS = 20           # Full training epochs
    BATCH_SIZE = 16          # Adjust based on GPU memory
    LEARNING_RATE = 1e-4     # Fine-tuning learning rate
    WEIGHT_DECAY = 0.01

    # LoRA hyperparameters
    LORA_R = 16               # Rank of LoRA matrices
    LORA_ALPHA = 32          # Scaling factor
    LORA_DROPOUT = 0.1       # Dropout probability

    # Model saving / logging
    MODEL_SAVE_DIR = "./saved_models"
    LOG_DIR = "./logs"

    # Tokenization / sequence
    MAX_SEQ_LENGTH = 256     # Full code sequence length

# Instantiate a single config object
config = Config()
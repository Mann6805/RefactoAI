import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaConfig
from peft import get_peft_model, LoraConfig, TaskType

class GraphCodeBERT_LoRA(nn.Module):
    def __init__(self, num_labels=2, r=8, alpha=16, dropout=0.1):
        super().__init__()

        # Load GraphCodeBERT for sequence classification
        self.model = RobertaForSequenceClassification.from_pretrained(
            "microsoft/graphcodebert-base",
            num_labels=num_labels
        )

        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["query", "value"]
        )
        self.model = get_peft_model(self.model, lora_config)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
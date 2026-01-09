# config.py
import torch

class Config:
    MODEL_ID = "xlm-roberta-large"
    MAX_LENGTH = 128
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    TARGET_MODULES = ["query", "key", "value", "dense"]
    
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 8
    EPOCHS = 5
    OUTPUT_DIR = "./results"
    
    DATA_PATH = "data/synthetic_data.json"
    LABELS = [
        "Multi-source synthesis", 
        "Procedure in context", 
        "Definition + qualification", 
        "Explainer with trade-offs", 
        "Aggregated facts"
    ]

label2id = {label: i for i, label in enumerate(Config.LABELS)}
id2label = {i: label for i, label in enumerate(Config.LABELS)}
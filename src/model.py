# src/model.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

def get_model_and_tokenizer(cfg, num_labels, label2id, id2label, is_train=True):
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.MODEL_ID, 
        num_labels=num_labels, 
        label2id=label2id, 
        id2label=id2label,
        # 修正 Warning: 使用 dtype 取代 torch_dtype
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if is_train:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=cfg.LORA_R, 
            lora_alpha=cfg.LORA_ALPHA, 
            lora_dropout=cfg.LORA_DROPOUT,
            target_modules=cfg.TARGET_MODULES,
            bias="none"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer
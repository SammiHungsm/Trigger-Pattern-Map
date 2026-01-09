# src/model.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

def get_model_and_tokenizer(model_id, num_labels, label2id, id2label, is_train=True):
    # 1. 載入 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 2. 載入模型 (加入 torch_dtype 慳 VRAM)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=num_labels, 
        label2id=label2id, 
        id2label=id2label,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if is_train:
        # LoRA 配置優化
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.1,
            # 對於 RoBERTa 架構，呢幾個 modules 通常係效能最好嘅組合
            target_modules=["query", "key", "value", "dense"], 
            bias="none"
        )
        model = get_peft_model(model, peft_config)
        
        # 打印出嚟睇下有幾多參數係真正訓練緊 (Optional)
        model.print_trainable_parameters()
    
    return model, tokenizer
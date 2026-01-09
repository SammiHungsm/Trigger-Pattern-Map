# src/model.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

def get_model_and_tokenizer(model_id, num_labels, label2id, id2label, is_train=True):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
    )
    
    if is_train:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["query", "value"]
        )
        model = get_peft_model(model, peft_config)
    
    return model, tokenizer
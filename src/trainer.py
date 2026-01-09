# src/trainer.py
import numpy as np
import evaluate
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

# 載入評估指標（準確率）
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """計算準確率嘅 Helper function"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def run_training(model, tokenizer, train_dataset, val_dataset, output_dir="./results"):
    """
    封裝 Trainer 邏輯，減少重複代碼
    """
    
    # 1. 定義訓練參數
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-4,            # LoRA 建議較高嘅 LR
        per_device_train_batch_size=8,  # 視乎 GPU VRAM 調整
        per_device_eval_batch_size=8,
        num_train_epochs=5,             # 合成數據建議行多幾 epoch
        weight_decay=0.01,
        evaluation_strategy="epoch",    # 每個 epoch 做一次評估
        save_strategy="epoch",          # 每個 epoch 儲存一次權重
        load_best_model_at_end=True,    # 訓練完自動車返最好嗰個 version
        logging_steps=10,
        remove_unused_columns=False,    # 重要：LoRA 需要保留一啲 column
        fp16=True,                      # 如果有 NVIDIA GPU 就開，快好多
    )

    # 2. Data Collator (自動幫你做 padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 4. 開始訓練
    print("🚀 訓練開始...")
    trainer.train()
    
    # 5. 儲存最終模型 (LoRA weights)
    trainer.save_model(f"{output_dir}/final_model")
    print(f"✅ 模型已儲存至 {output_dir}/final_model")
    
    return trainer
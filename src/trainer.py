# src/trainer.py
import numpy as np
import evaluate
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def run_training(model, tokenizer, train_ds, val_ds, id2label, label2id, cfg):
    # 修正：將 evaluation_strategy 改為 eval_strategy
    training_args = TrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        learning_rate=cfg.LEARNING_RATE,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE,
        num_train_epochs=cfg.EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",    # 最新版 transformers 參數名
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        remove_unused_columns=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    # 將標籤映射存入模型配置，方便推論
    model.config.id2label = id2label
    model.config.label2id = label2id
    trainer.save_model(f"{cfg.OUTPUT_DIR}/final_model")
    print(f"✅ 訓練完成！模型已儲存至 {cfg.OUTPUT_DIR}/final_model")
import numpy as np
import evaluate
import torch
from transformers import (
    Trainer, 
    TrainingArguments, 
    DataCollatorForTokenClassification,
    TrainerCallback
)

# 1. 載入 NER 專用評估指標 (seqeval)
# 佢會幫你計 Precision, Recall, F1，而唔係單純嘅 Accuracy
metric = evaluate.load("seqeval")

def compute_metrics(p, id2label):
    """NER 專用指標計算函數"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 移除 -100 (padding/special tokens) 並轉返做 Label 名稱
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 2. 自定義 Debug Callback：將訓練過程寫入 TensorBoard 同埋印出樣本
class NERDebugCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\n🔍 [Debug] Step {state.global_step} 評估結果: F1={metrics.get('eval_f1', 0):.4f}")

def run_training(model, tokenizer, train_dataset, val_dataset, id2label, output_dir="./results"):
    """
    優化版 NER 訓練器
    """
    
    # 1. 定義訓練參數 (加入 TensorBoard 同埋 VRAM 優化)
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-4,
        per_device_train_batch_size=4,   # Large 模型建議由 4 開始，防止 OOM
        gradient_accumulation_steps=2,  # 累積梯度維持有效 batch size 為 8
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="steps",          # 每隔一段步數就評估，唔使等全個 epoch
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        logging_steps=10,               # 每 10 步就 Log 一次
        remove_unused_columns=True,     # 必設為 True 以避免 "str" ValueError
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(), # 有 GPU 就開 FP16
        # 🔥 TensorBoard 配置
        report_to="tensorboard",
        logging_dir=f"{output_dir}/logs",
        # 🔥 VRAM 優化
        gradient_checkpointing=True
    )

    # 2. Data Collator (NER 必須用 ForTokenClassification)
    # 佢會幫你自動處理 Label 嘅 Padding 設為 -100
    data_collator = DataCollatorForTokenClassification(
        tokenizer, 
        pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # 3. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # 傳入自定義嘅 compute_metrics (需要 id2label)
        compute_metrics=lambda p: compute_metrics(p, id2label),
        callbacks=[NERDebugCallback()]
    )

    # 4. 開始訓練
    print("🚀 訓練啟動中... 你可以喺新 Terminal 輸入 'tensorboard --logdir=./results/logs' 睇圖表")
    trainer.train()
    
    # 5. 儲存
    trainer.save_model(f"{output_dir}/final_model")
    return trainer
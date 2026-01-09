# src/data_loader.py
import json
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from config import label2id

def load_synthetic_data(json_path, tokenizer, max_length=128):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # 清洗：確保標籤完全匹配
    df['label'] = df['label'].astype(str).str.strip()
    df['mapped_label'] = df['label'].map(label2id)
    
    if df['mapped_label'].isnull().any():
        invalid = df[df['mapped_label'].isnull()]['label'].unique()
        print(f"⚠️ 跳過無法辨識的標籤: {invalid}")
        df = df.dropna(subset=['mapped_label'])
    
    df['label'] = df['mapped_label'].astype(int)
    df = df.drop(columns=['mapped_label'])
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
    
    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

    # 移除 text 避免 Tensor 報錯
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    train_ds.set_format("torch")
    val_ds.set_format("torch")
    
    return train_ds, val_ds
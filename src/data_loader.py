# src/data_loader.py
import json
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

def load_synthetic_data(json_path, tokenizer, max_length=128):
    """
    讀取合成數據並轉換為 Hugging Face Dataset 格式
    """
    # 1. 讀取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # 2. 類別轉 ID (必須與 model.py 中的 label2id 一致)
    label2id = {
        "Multi-source": 0, 
        "Procedure": 1, 
        "Definition": 2, 
        "Explainer": 3, 
        "Aggregated Facts": 4
    }
    df['label'] = df['label'].map(label2id)
    
    # 3. 拆分訓練集與驗證集 (8:2)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 4. 轉換為 Dataset
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    
    # 5. Tokenization 處理 (減少 loop 代碼)
    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)
    
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    
    return train_ds, val_ds
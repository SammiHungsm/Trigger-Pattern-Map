# main.py
from src.model import get_model_and_tokenizer
from src.data_loader import load_synthetic_data
from src.trainer import run_training

def main():
    # 1. 設定
    MODEL_ID = "xlm-roberta-large"
    LABELS = ["Multi-source", "Procedure", "Definition", "Explainer", "Aggregated Facts"]
    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label = {i: l for i, l in enumerate(LABELS)}

    # 2. 攞模型同 Tokenizer (引用 src/model.py)
    model, tokenizer = get_model_and_tokenizer(
        MODEL_ID, len(LABELS), label2id, id2label, is_train=True
    )

    # 3. 準備數據 (引用 src/data_loader.py)
    # 記得先放好 data/synthetic_data.json
    train_ds, val_ds = load_synthetic_data("data/synthetic_data.json", tokenizer)
    train_ds = train_ds.remove_columns(["text", "tokens"]) 
    val_ds = val_ds.remove_columns(["text", "tokens"])

    run_training(model, tokenizer, train_ds, val_ds)

if __name__ == "__main__":
    main()
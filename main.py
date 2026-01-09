# main.py
from config import Config, label2id, id2label
from src.model import get_model_and_tokenizer
from src.data_loader import load_synthetic_data
from src.trainer import run_training

def main():
    # 1. 初始化
    model, tokenizer = get_model_and_tokenizer(
        Config, len(Config.LABELS), label2id, id2label, is_train=True
    )

    # 2. 準備數據
    train_ds, val_ds = load_synthetic_data(Config.DATA_PATH, tokenizer, Config.MAX_LENGTH)

    # 3. 執行訓練
    run_training(model, tokenizer, train_ds, val_ds, id2label, label2id, Config)

if __name__ == "__main__":
    main()
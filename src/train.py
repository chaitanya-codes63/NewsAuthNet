# src/step2_train.py
import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.preprocessing import TextTokenizer
from src.models_tf import build_keras_model, save_keras_pkl

SEED = 42
tf.keras.utils.set_random_seed(SEED)

PROC_DIR = Path("data/processed")
MODELS_DIR = Path("models")

class EpochLogger(tf.keras.callbacks.Callback):
    """Prints a clear summary after every epoch so you can track val_loss & accuracy."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(
            f"\n>>> [Epoch {epoch + 1:02d}] "
            f"train_loss: {logs.get('loss', 0):.4f} | "
            f"train_acc: {logs.get('accuracy', 0)*100:.2f}% | "
            f"val_loss: {logs.get('val_loss', 0):.4f} | "
            f"val_acc: {logs.get('val_accuracy', 0)*100:.2f}% | "
            f"val_auc: {logs.get('val_auc', 0):.4f}"
        )

def train_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame):
    print("\n==========================================")
    print("Training Classical Baseline (TF-IDF + LR)")
    print("==========================================")
    t0 = time.time()
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X_tr = vec.fit_transform(train_df["text"].fillna(""))
    X_va = vec.transform(val_df["text"].fillna(""))
    y_tr = train_df["label"].values
    y_va = val_df["label"].values

    best_c, best_f1, best_clf = 1.0, -1.0, None
    for c in [0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=c, max_iter=1000, random_state=SEED)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_va)
        f1 = f1_score(y_va, preds, average="macro")
        print(f"  C={c:<4} -> Val Macro-F1: {f1:.4f}")
        if f1 > best_f1:
            best_c, best_f1, best_clf = c, f1, clf

    out_dir = MODELS_DIR / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(
            {
                "vectorizer": vec,
                "model": best_clf,
                "best_C": best_c,
                "train_seconds": round(time.time() - t0, 2),
            },
            f,
        )
    print(f"Saved Baseline (C={best_c}) -> {out_dir / 'model.pkl'}")

def train_neural(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int = 6,
    batch_size: int = 64,
):
    print(f"\n==========================================")
    print(f"Training TensorFlow Model: {model_name.upper()}")
    print(f"==========================================")
    t0 = time.time()
    out_dir = MODELS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 30000
    max_len = 300

    print("Fitting word-level tokenizer on training split only...")
    tok = TextTokenizer(vocab_size=vocab_size, max_len=max_len)
    tok.fit(train_df["text"].fillna("").tolist())
    tok.save(out_dir / "tokenizer.pkl")

    X_tr = tok.encode_batch(train_df["text"].fillna("").tolist())
    y_tr = train_df["label"].values.astype("float32")
    X_va = tok.encode_batch(val_df["text"].fillna("").tolist())
    y_va = val_df["label"].values.astype("float32")

    cfg = {
        "name": model_name,
        "vocab_size": vocab_size,
        "max_len": max_len,
        "embed_dim": 128,
        "hidden_dim": 96,
        "filters": 128,
        "kernel_sizes": (3, 4, 5),
        "dropout": 0.3,
        "lr": 1e-3,
    }

    model = build_keras_model(**cfg)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[EpochLogger(), early_stop],
        verbose=1,
    )

    # Save weights + architecture config as .pkl
    save_keras_pkl(model, cfg, out_dir / "model.pkl")

    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_dict["train_seconds"] = round(time.time() - t0, 2)
    with open(out_dir / "history.json", "w") as f:
        json.dump(hist_dict, f, indent=2)

    print(f"Saved {model_name.upper()} weights -> {out_dir / 'model.pkl'}")
    print(f"Saved training history -> {out_dir / 'history.json'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["baseline", "cnn", "lstm", "gru"],
        choices=["baseline", "cnn", "lstm", "gru"],
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    train_df = pd.read_csv(PROC_DIR / "train.csv")
    val_df = pd.read_csv(PROC_DIR / "val.csv")

    for m in args.models:
        if m == "baseline":
            train_baseline(train_df, val_df)
        else:
            train_neural(m, train_df, val_df, epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
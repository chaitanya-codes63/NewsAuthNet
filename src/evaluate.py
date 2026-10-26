# src/step3_evaluate.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from src.predict import Predictor, installed_models

PROC_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"

def plot_cm(cm, model_name: str):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Fake (0)", "Real (1)"],
        yticklabels=["Fake (0)", "Real (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"Confusion Matrix - {model_name.upper()}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"cm_{model_name}.png", dpi=150)
    plt.close(fig)

def plot_learning_curves(model_name: str):
    hist_path = MODELS_DIR / model_name / "history.json"
    if not hist_path.exists():
        return
    with open(hist_path, "r") as f:
        h = json.load(f)
    epochs = range(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
    axes[0].plot(epochs, h["loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, h["val_loss"], marker="s", label="Val Loss")
    axes[0].set_title(f"{model_name.upper()} - Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, h["accuracy"], marker="o", label="Train Acc")
    axes[1].plot(epochs, h["val_accuracy"], marker="s", label="Val Acc")
    axes[1].set_title(f"{model_name.upper()} - Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"curves_{model_name}.png", dpi=150)
    plt.close(fig)

def evaluate_all():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    test_df = pd.read_csv(PROC_DIR / "test.csv")
    y_true = test_df["label"].values
    texts = test_df["text"].fillna("").tolist()

    all_metrics = {}
    rows = []

    for name in installed_models():
        print(f"\nEvaluating {name.upper()} on {len(test_df)} test articles...")
        pred_engine = Predictor(name)
        y_scores = np.array(pred_engine.predict_batch(texts))
        y_pred = (y_scores >= 0.5).astype(int)

        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(np.mean(f1))
        auc = float(roc_auc_score(y_true, y_scores))

        all_metrics[name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "roc_auc": auc,
            "per_class": {
                "Fake": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f1[0]), "support": int(s[0])},
                "Real": {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f1[1]), "support": int(s[1])},
            },
            "confusion_matrix": cm,
        }

        rows.append({
            "model": name.upper(),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "roc_auc": auc,
            "fake_recall": float(r[0]),
            "real_recall": float(r[1]),
        })

        plot_cm(cm, name)
        plot_learning_curves(name)

    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    comp_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    comp_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    print("\n=== HELD-OUT 15% TEST SPLIT COMPARISON ===")
    print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    evaluate_all()
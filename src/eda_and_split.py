
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from src.preprocessing import clean_text, _SOURCE_TAG, _SOURCE_WORDS

SEED = 42
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"

def has_publisher_tag(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(_SOURCE_TAG.search(text) or _SOURCE_WORDS.search(text))

def run_eda_and_split():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Loading raw datasets...")
    fake_df = pd.read_csv(RAW_DIR / "Fake.csv")
    true_df = pd.read_csv(RAW_DIR / "True.csv")

    fake_df["label"] = 0  # 0 = Fake
    true_df["label"] = 1  # 1 = Real

    raw_fake_count = len(fake_df)
    raw_real_count = len(true_df)
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Combine title + text
    df["raw_full"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["has_tag"] = df["raw_full"].apply(has_publisher_tag)

    tag_fake = int(df[df["label"] == 0]["has_tag"].sum())
    tag_real = int(df[df["label"] == 1]["has_tag"].sum())

    print("2. Cleaning text and stripping publisher shortcuts (Reuters, etc.)...")
    df["clean_text"] = df["raw_full"].apply(lambda x: clean_text(x, strip_sources=True))
    df["char_len"] = df["clean_text"].str.len()
    df["word_count"] = df["clean_text"].str.split().str.len()

    # Filter out empty/short rows (< 20 chars) and exact duplicates
    df = df[df["char_len"] >= 20].copy()
    
    # Remove rows where identical clean_text has conflicting labels
    label_counts = df.groupby("clean_text")["label"].nunique()
    conflicting_texts = label_counts[label_counts > 1].index
    df = df[~df["clean_text"].isin(conflicting_texts)].copy()

    # Drop exact duplicates
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

    clean_fake_count = int((df["label"] == 0).sum())
    clean_real_count = int((df["label"] == 1).sum())

    print("3. Creating Stratified 70% Train / 15% Val / 15% Test Splits...")
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
    )

    train_df[["title", "clean_text", "label"]].rename(columns={"clean_text": "text"}).to_csv(PROC_DIR / "train.csv", index=False)
    val_df[["title", "clean_text", "label"]].rename(columns={"clean_text": "text"}).to_csv(PROC_DIR / "val.csv", index=False)
    test_df[["title", "clean_text", "label"]].rename(columns={"clean_text": "text"}).to_csv(PROC_DIR / "test.csv", index=False)

    # Save audit summary for Table 4.1
    audit = {
        "raw": {"fake": raw_fake_count, "real": raw_real_count, "total": raw_fake_count + raw_real_count},
        "publisher_tags_stripped": {"fake": tag_fake, "real": tag_real, "total": tag_fake + tag_real},
        "after_audit_dedup": {"fake": clean_fake_count, "real": clean_real_count, "total": len(df)},
        "splits": {
            "train": {"fake": int((train_df["label"] == 0).sum()), "real": int((train_df["label"] == 1).sum()), "total": len(train_df)},
            "val": {"fake": int((val_df["label"] == 0).sum()), "real": int((val_df["label"] == 1).sum()), "total": len(val_df)},
            "test": {"fake": int((test_df["label"] == 0).sum()), "real": int((test_df["label"] == 1).sum()), "total": len(test_df)},
        },
        "avg_word_count": {
            "fake": round(float(df[df["label"] == 0]["word_count"].mean()), 1),
            "real": round(float(df[df["label"] == 1]["word_count"].mean()), 1),
        }
    }

    with open(REPORTS_DIR / "data_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    # Generate EDA Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.countplot(data=df, x="label", ax=axes[0], palette=["#e11d48", "#059669"])
    axes[0].set_xticklabels(["Fake (0)", "Real (1)"])
    axes[0].set_title("Class Distribution After De-duplication")
    axes[0].set_ylabel("Article Count")

    sns.histplot(
        data=df[df["word_count"] <= 1000],
        x="word_count",
        hue="label",
        bins=40,
        ax=axes[1],
        palette=["#e11d48", "#059669"],
        element="step",
    )
    axes[1].set_title("Article Word Count Distribution (<= 1000 words)")
    axes[1].set_xlabel("Word Count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_distribution.png", dpi=150)
    plt.close(fig)

    print("\n=== DATA AUDIT SUMMARY (reports/data_audit.json) ===")
    print(json.dumps(audit, indent=2))
    print(f"EDA chart saved to: {FIG_DIR / 'eda_distribution.png'}")

if __name__ == "__main__":
    run_eda_and_split()
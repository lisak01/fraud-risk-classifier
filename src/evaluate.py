"""
evaluate.py
------------
Loads the fine-tuned model and evaluates it on the test set.
Outputs per-class precision, recall, F1 and saves a confusion matrix plot.

Two ways to run this script:
 
  1. python src/evaluate.py
     Uses default model path: models/finbert-fraud-risk
     Quickest way to evaluate after training.
 
  2. python src/evaluate.py --model_dir models/finbert-fraud-risk
     Explicitly specifies which model to evaluate.
     Useful when you have multiple trained models and want to compare:
     - python src/evaluate.py --model_dir models/finbert-ep5
     - python src/evaluate.py --model_dir models/distilbert-fraud

"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from utils import get_logger

logger = get_logger(__name__)

LABEL_MAP = {
    0: "FRAUD",
    1: "AML_RISK",
    2: "CREDIT_RISK",
    3: "OPERATIONAL_RISK",
    4: "LOW_RISK",
}
LABEL_NAMES = list(LABEL_MAP.values())


def load_model_and_tokenizer(model_dir: str):
    logger.info(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model


def predict_batch(texts, classifier_pipeline, batch_size: int = 32):
    """Run inference in batches and return predicted label IDs."""
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = classifier_pipeline(batch, truncation=True, max_length=128)
        for r in results:
            label_name = r["label"]
            # map label name back to ID
            label_id = {v: k for k, v in LABEL_MAP.items()}[label_name]
            all_preds.append(label_id)
    return np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, output_path: str = "reports/confusion_matrix.png"):
    """Save a styled confusion matrix heatmap."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (normalized)", fontsize=14, pad=15)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")
    return output_path


def evaluate(model_dir: str = "models/finbert-fraud-risk", data_dir: str = "data/processed"):
    tokenizer, model = load_model_and_tokenizer(model_dir)

    # HuggingFace pipeline for easy batch inference
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # CPU; set to 0 for GPU
    )

    # Load test data
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    texts  = test_df["sentence"].tolist()
    y_true = test_df["label"].values

    logger.info(f"Running inference on {len(texts)} test samples...")
    y_pred = predict_batch(texts, clf)

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=LABEL_NAMES,
        digits=4,
        output_dict=True,
    )
    report_str = classification_report(
        y_true, y_pred,
        target_names=LABEL_NAMES,
        digits=4,
    )
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report_str)

    # Confusion matrix
    cm_path = plot_confusion_matrix(y_true, y_pred)

    # Log to MLflow
    mlflow.set_experiment("fraud-risk-classification")
    with mlflow.start_run(run_name="evaluation"):
        for label in LABEL_NAMES:
            mlflow.log_metrics({
                f"{label}_precision": report[label]["precision"],
                f"{label}_recall":    report[label]["recall"],
                f"{label}_f1":        report[label]["f1-score"],
            })
        mlflow.log_metric("weighted_f1", report["weighted avg"]["f1-score"])
        mlflow.log_artifact(cm_path)
        logger.info("Metrics logged to MLflow.")

    return report


if __name__ == "__main__":
    # Entry point: parse command-line arguments and run evaluation
    # Usage: python src/evaluate.py
    #        python src/evaluate.py --model_dir models/finbert-fraud-risk
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/finbert-fraud-risk")
    parser.add_argument("--data_dir",  default="data/processed")
    args = parser.parse_args()
    evaluate(model_dir=args.model_dir, data_dir=args.data_dir)

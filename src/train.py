"""
train.py
---------
Fine-tunes a HuggingFace transformer model (default: FinBERT) on the
fraud/risk classification task. Tracks experiments with MLflow.

  Two ways to run this script:
 
  1. python src/train.py
     Uses all default values:
     - model: ProsusAI/finbert
     - epochs: 3
     - batch_size: 16
     - learning_rate: 2e-05
 
  2. python src/train.py --model_name ProsusAI/finbert --epochs 5
     Overrides specific values from the command line.
     Useful for experimenting without changing the code.
     Example variations:
     python src/train.py --model_name ProsusAI/finbert --epochs 5 --batch_size 8 --lr 3e-05
      --epochs 5              (train longer)
      --batch_size 8          (less memory)
      --lr 3e-05              (different learning rate)
"""

import argparse
import os
import numpy as np
import mlflow
import mlflow.pytorch
from datasets import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch import nn
from utils import get_logger

logger = get_logger(__name__)

# Label definitions (must match data_preparation.py)
LABEL_MAP = {
    0: "FRAUD",
    1: "AML_RISK",
    2: "CREDIT_RISK",
    3: "OPERATIONAL_RISK",
    4: "LOW_RISK",
}
ID2LABEL = LABEL_MAP
LABEL2ID = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)


# ---------------------------------------------------------------------------
# Custom Trainer with class-weighted loss (handles label imbalance)
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = torch.tensor(self.class_weights, dtype=torch.float).to(logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def tokenize_dataset(df: pd.DataFrame, tokenizer, max_length: int = 128) -> Dataset:
    """Convert DataFrame to HuggingFace Dataset and tokenize."""
    ds = Dataset.from_pandas(df[["sentence", "label"]].rename(columns={"sentence": "text"}))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1":        f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall":    recall_score(labels, preds, average="weighted", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    model_name: str = "ProsusAI/finbert",
    data_dir: str = "data/processed",
    output_dir: str = "models/finbert-fraud-risk",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
):
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Load data
    logger.info("Loading processed data...")
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    val_df   = pd.read_csv(f"{data_dir}/val.csv")

    # Compute class weights for imbalanced labels
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_LABELS),
        y=train_df["label"].values,
    )
    logger.info(f"Class weights: {dict(zip(LABEL_MAP.values(), class_weights.round(3)))}")

    # Tokenize
    logger.info("Tokenizing datasets...")
    train_ds = tokenize_dataset(train_df, tokenizer, max_length)
    val_ds   = tokenize_dataset(val_df, tokenizer, max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),  # use mixed precision if GPU available
        report_to="none",                # we handle MLflow manually
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # MLflow experiment tracking — speichert immer im Projektroot
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlflow.set_tracking_uri(f"file://{project_root}/mlruns")
    mlflow.set_experiment("fraud-risk-classification")
    with mlflow.start_run(run_name=f"{model_name.split('/')[-1]}-ep{epochs}"):
        mlflow.log_params({
            "model_name":    model_name,
            "epochs":        epochs,
            "batch_size":    batch_size,
            "learning_rate": learning_rate,
            "max_length":    max_length,
            "num_labels":    NUM_LABELS,
        })

        logger.info("Starting training...")
        trainer.train()

        # Log final validation metrics
        val_results = trainer.evaluate()
        mlflow.log_metrics({
            "val_f1":        val_results["eval_f1"],
            "val_precision": val_results["eval_precision"],
            "val_recall":    val_results["eval_recall"],
        })
        logger.info(f"Validation results: {val_results}")

    # Save final model + tokenizer
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT for fraud/risk classification")
    parser.add_argument("--model_name",  default="ProsusAI/finbert")
    parser.add_argument("--data_dir",    default="data/processed")
    parser.add_argument("--output_dir",  default="models/finbert-fraud-risk")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--max_length",  type=int,   default=128)
    args = parser.parse_args()

    train(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )

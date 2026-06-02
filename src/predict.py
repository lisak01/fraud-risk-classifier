"""
predict.py
-----------
Load the fine-tuned model and classify a single text or a CSV file.

Usage:
    # Single text
    python src/predict.py --text "Customer transfers large amounts to shell companies abroad."

    # Batch from CSV
    python src/predict.py --input_file valid_data/test.csv --output_file predictions.csv
"""

import argparse
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from utils import get_logger

logger = get_logger(__name__)

LABEL_MAP = {
    "FRAUD":            "Fraud indicator detected",
    "AML_RISK":         "Anti-money laundering risk",
    "CREDIT_RISK":      "Credit risk signal",
    "OPERATIONAL_RISK": "Operational risk indicator",
    "LOW_RISK":         "Low risk — normal communication",
}


def load_pipeline(model_dir: str = "models/finbert-fraud-risk"):
    """
    Loads the fine-tuned tokenizer and model from disk.
    Returns a HuggingFace text-classification pipeline ready for inference.
    device=-1 means CPU; set to 0 to use GPU.
    """
    logger.info(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        truncation=True,
        max_length=128,
    )


def classify_text(text: str, clf) -> dict:
    """Classify a single text and return label + confidence."""
    result = clf(text)[0]
    label = result["label"]
    score = result["score"]
    return {
        "text":        text,
        "label":       label,
        "description": LABEL_MAP.get(label, label),
        "confidence":  round(score, 4),
    }


def classify_file(input_path: str, output_path: str, clf, text_column: str = "sentence"):
    """
    Runs batch inference on all texts in a CSV file.
    Merges predictions with the original dataframe and saves to output_path.
    Raises ValueError if the expected text column is not found.
    """
    df = pd.read_csv(input_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available: {list(df.columns)}")

    logger.info(f"Classifying {len(df)} texts from {input_path}...")
    results = [classify_text(t, clf) for t in df[text_column].tolist()]
    results_df = pd.DataFrame(results)

    # Merge with original df
    output_df = pd.concat([df.reset_index(drop=True), results_df[["label", "confidence", "description"]]], axis=1)
    output_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    return output_df


def pretty_print(result: dict):
    """
    Prints a single classification result in a readable format.
    Truncates long input texts to 80 characters for clean display.
    """
    print("\n" + "─" * 50)
    print(f"  Input:       {result['text'][:80]}{'...' if len(result['text']) > 80 else ''}")
    print(f"  Label:       {result['label']}")
    print(f"  Assessment:  {result['description']}")
    print(f"  Confidence:  {result['confidence']:.1%}")
    print("─" * 50 + "\n")


if __name__ == "__main__":
    # Entry point: supports three modes:
    # 1. Single text:  python src/predict.py --text "..."
    # 2. Batch CSV:    python src/predict.py --input_file data/test.csv
    # 3. Demo mode:    python src/predict.py  (no arguments)
    parser = argparse.ArgumentParser(description="Fraud/Risk text classifier")
    parser.add_argument("--model_dir",   default="models/finbert-fraud-risk")
    parser.add_argument("--text",        type=str, default=None, help="Single text to classify")
    parser.add_argument("--input_file",  type=str, default=None, help="CSV file for batch prediction")
    parser.add_argument("--output_file", type=str, default="predictions.csv")
    parser.add_argument("--text_column", type=str, default="sentence")
    args = parser.parse_args()

    clf = load_pipeline(args.model_dir)

    if args.text:
        result = classify_text(args.text, clf)
        pretty_print(result)
    elif args.input_file:
        classify_file(args.input_file, args.output_file, clf, args.text_column)
    else:
        # Demo mode — show a few example predictions
        examples = [
            "The customer has been transferring funds in amounts just below the reporting threshold repeatedly.",
            "Invoice payment processed successfully. Amount matches purchase order.",
            "Account holder reported unauthorized access and multiple failed login attempts from foreign IPs.",
            "Client is showing signs of inability to service debt obligations for the third consecutive month.",
            "Routine quarterly interest payment received on schedule.",
        ]
        print("\nFraud & Risk Text Classifier — Demo\n")
        for text in examples:
            result = classify_text(text, clf)
            pretty_print(result)

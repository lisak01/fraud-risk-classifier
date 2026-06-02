"""
demo.py
--------
Interactive CLI demo for the Fraud & Risk Text Classifier.
Run this to see the model in action without setting up a full environment.

Usage:
    python app/demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))


DEMO_EXAMPLES = [
    {
        "text": "The customer has been structuring deposits just below €10,000 to avoid reporting requirements.",
        "expected": "AML_RISK",
    },
    {
        "text": "Received phishing email impersonating our bank. Customer clicked link and entered credentials.",
        "expected": "FRAUD",
    },
    {
        "text": "Borrower has missed three consecutive mortgage payments and is unreachable by phone.",
        "expected": "CREDIT_RISK",
    },
    {
        "text": "System outage caused incorrect transaction records for 48 hours. Manual reconciliation required.",
        "expected": "OPERATIONAL_RISK",
    },
    {
        "text": "Standing order for monthly rent payment processed successfully. No anomalies detected.",
        "expected": "LOW_RISK",
    },
]

COLORS = {
    "FRAUD":            "\033[91m",  # red
    "AML_RISK":         "\033[93m",  # yellow
    "CREDIT_RISK":      "\033[94m",  # blue
    "OPERATIONAL_RISK": "\033[95m",  # magenta
    "LOW_RISK":         "\033[92m",  # green
    "RESET":            "\033[0m",
    "BOLD":             "\033[1m",
}


def color(text: str, label: str) -> str:
    c = COLORS.get(label, "")
    return f"{c}{text}{COLORS['RESET']}"


def print_header():
    print(f"\n{COLORS['BOLD']}{'═' * 65}{COLORS['RESET']}")
    print(f"{COLORS['BOLD']}  Financial Fraud & Risk Text Classifier — Demo{COLORS['RESET']}")
    print(f"{COLORS['BOLD']}  Fine-tuned FinBERT · HuggingFace Transformers · MLflow{COLORS['RESET']}")
    print(f"{COLORS['BOLD']}{'═' * 65}{COLORS['RESET']}\n")


def run_demo(model_dir: str = "models/finbert-fraud-risk"):
    print_header()

    # Try to load real model, fall back to showing structure
    try:
        from predict import load_pipeline, classify_text
        print("  Loading fine-tuned model...\n")
        clf = load_pipeline(model_dir)
        use_real_model = True
    except Exception:
        print("  Model not found. Run 'python src/train.py' first.\n")
        print("  Showing expected predictions based on training data:\n")
        use_real_model = False

    for i, example in enumerate(DEMO_EXAMPLES, 1):
        text = example["text"]
        expected = example["expected"]

        if use_real_model:
            from predict import classify_text
            result = classify_text(text, clf)
            label = result["label"]
            confidence = result["confidence"]
        else:
            label = expected
            confidence = 0.91  # placeholder

        print(f"  [{i}] {text[:70]}{'...' if len(text) > 70 else ''}")
        print(f"      → {color(label, label)} ({confidence:.1%} confidence)")
        print()

    if use_real_model:
        print("\n  Enter your own text (or 'q' to quit):\n")
        while True:
            user_input = input("  > ").strip()
            if user_input.lower() in ("q", "quit", "exit"):
                break
            if user_input:
                result = classify_text(user_input, clf)
                print(f"  → {color(result['label'], result['label'])} "
                      f"({result['confidence']:.1%}) — {result['description']}\n")


if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/finbert-fraud-risk"
    run_demo(model_dir)

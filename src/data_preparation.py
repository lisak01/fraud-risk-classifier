"""
data_preparation.py
--------------------
Generated a synthetic banking dataset with realistic financial texts for 5 risk categories. 
No external data source required — runs fully offline.
"""

import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import get_logger

logger = get_logger(__name__)

LABEL_MAP = {
    0: "FRAUD",
    1: "AML_RISK",
    2: "CREDIT_RISK",
    3: "OPERATIONAL_RISK",
    4: "LOW_RISK",
}
ID2LABEL = LABEL_MAP
LABEL2ID = {v: k for k, v in LABEL_MAP.items()}

# ---------------------------------------------------------------------------
# Synthetic texts per category with realistic banking vocabulary.​​​​​​​​​​​​​​​​
# ---------------------------------------------------------------------------

#{} These are placeholders from our data generator 
#in data_preparation.py. The _fill_template() function automatically replaces them with random values before the data is used for training.
TEMPLATES = {
    0: [  # FRAUD
        "Unauthorized wire transfer of {amount} EUR detected from account {acc}.", 
        "Customer reported phishing email impersonating the bank requesting login credentials.",
        "Multiple failed login attempts followed by successful access from an unknown IP address.",
        "Credit card used in {city} minutes after being used in {city2} — likely cloned.",
        "Customer received fake invoice from spoofed company email requesting urgent payment.",
        "Identity theft suspected: new account opened with stolen documents in branch {branch}.",
        "Suspicious chargeback request for transaction the customer claims not to have authorized.",
        "Account credentials were found in a data breach; customer has not been notified yet.",
        "Duplicate invoices submitted for reimbursement by the same vendor within 48 hours.",
        "Employee accessed {num} customer accounts outside of normal working hours without business reason.",
        "Transaction reversed immediately after funds were withdrawn at ATM in foreign country.",
        "Customer denies initiating a {amount} EUR international wire transfer made this morning.",
        "Phishing link clicked by corporate user; credentials may have been compromised.",
        "Fraudulent account application submitted with altered identity documents.",
        "Card-not-present fraud detected: {num} online purchases in rapid succession.",
    ],
    1: [  # AML_RISK
        "Customer deposits {amount} EUR in cash daily, always just below the reporting threshold.",
        "Series of small transfers to {num} different accounts in high-risk jurisdictions.",
        "Shell company with no apparent business activity receives large regular wire transfers.",
        "Funds rapidly moved through {num} accounts before being withdrawn as cash — layering pattern.",
        "Customer unable to provide source of funds for {amount} EUR deposit.",
        "Politically exposed person opened account without disclosure; enhanced due diligence required.",
        "Transaction pattern inconsistent with stated business activity of the account holder.",
        "Large cash deposits followed by immediate international wire transfers to offshore accounts.",
        "Customer structuring deposits to avoid the {amount} EUR mandatory reporting threshold.",
        "Business account receives funds from {num} unrelated third parties with no clear invoices.",
        "High-value real estate purchased with cash — no mortgage, no financing documentation.",
        "Frequent currency exchange transactions with no documented business justification.",
        "Wire transfers to sanctioned country detected; account flagged for immediate review.",
        "Customer recently added {num} new beneficiaries in high-risk countries simultaneously.",
        "Round-number transactions of exactly {amount} EUR repeated weekly with no variation.",
    ],
    2: [  # CREDIT_RISK
        "Borrower has missed {num} consecutive mortgage payments and is unreachable.",
        "Debt-to-income ratio exceeds {num}% — loan application does not meet credit policy.",
        "Customer declared personal insolvency; outstanding loan balance of {amount} EUR at risk.",
        "Corporate borrower's revenue declined {num}% — covenant breach likely in next quarter.",
        "Credit score dropped from {num} to {num2} following multiple missed payments.",
        "Overdraft limit exceeded for {num} consecutive months with no repayment plan agreed.",
        "Customer requested restructuring of {amount} EUR loan due to financial hardship.",
        "SME client unable to service debt after loss of primary contract worth {amount} EUR.",
        "Non-performing loan provisioning required for account outstanding over 90 days.",
        "Guarantor for corporate loan has also filed for insolvency — collateral now insufficient.",
        "Rising interest rates increased debt service costs beyond borrower's repayment capacity.",
        "Credit facility of {amount} EUR fully drawn with no indication of near-term repayment.",
        "Customer's employer entered administration — risk of income loss and loan default.",
        "Collateral valuation for mortgage fell {num}% below outstanding loan balance.",
        "Early warning indicator: {num} returned direct debits in the past 30 days.",
    ],
    3: [  # OPERATIONAL_RISK
        "System outage caused {num} transactions to be processed twice — manual reversal required.",
        "Incorrect exchange rate applied to {num} FX transactions due to feed failure.",
        "Regulatory report submitted with incorrect data; resubmission required within 48 hours.",
        "Branch staff processed a {amount} EUR transaction without required dual authorization.",
        "Customer data exported to wrong recipient due to email configuration error.",
        "Core banking system unavailable for {num} hours during peak trading — SLA breached.",
        "Settlement failure on {num} bond trades due to incorrect IBAN in payment instruction.",
        "New product launched without completing mandatory risk assessment and sign-off process.",
        "Backup procedure not executed — data recovery for yesterday's transactions not possible.",
        "Staff member bypassed four-eyes principle for a {amount} EUR payment approval.",
        "Audit trail missing for {num} transactions processed during system migration window.",
        "Third-party vendor experienced outage affecting {num} customer-facing services.",
        "Error in interest calculation affected {num} savings accounts — correction in progress.",
        "Change management process not followed for production deployment — rollback initiated.",
        "Sensitive customer data stored in unencrypted format on local workstation.",
    ],
    4: [  # LOW_RISK
        "Monthly salary payment of {amount} EUR received as expected from employer.",
        "Standing order for rent of {amount} EUR processed successfully.",
        "Customer transferred {amount} EUR to own savings account at another bank.",
        "Quarterly dividend payment received from investment portfolio.",
        "Routine account statement generated and sent to customer address on file.",
        "Direct debit for utility bill of {amount} EUR processed without issues.",
        "Customer upgraded to premium account following review of relationship value.",
        "Mortgage payment of {amount} EUR received on schedule — no arrears.",
        "Annual fee of {amount} EUR debited from account as per agreed terms.",
        "Customer contacted support to update address — identity verification successful.",
        "Interest payment of {amount} EUR credited to savings account at month end.",
        "ATM withdrawal of {amount} EUR in home country — no anomalies detected.",
        "Business account turnover consistent with prior months — no unusual activity.",
        "New fixed deposit opened for {amount} EUR with {num}-month term.",
        "Card replacement requested by customer after expiry — standard process completed.",
    ],
}


def _fill_template(text: str) -> str:
    """Replace placeholders with random realistic values."""
    cities = ["Berlin", "Munich", "Frankfurt", "Hamburg", "Vienna", "Zurich", "Amsterdam"]
    branches = ["B-042", "B-117", "B-203", "B-089"]
    text = text.replace("{amount}", str(random.choice([500, 1200, 2500, 4800, 9500, 15000, 48000, 95000])))
    text = text.replace("{amount2}", str(random.choice([1000, 5000, 20000])))
    text = text.replace("{num}", str(random.randint(2, 15)))
    text = text.replace("{num2}", str(random.randint(300, 600)))
    text = text.replace("{city}", random.choice(cities))
    text = text.replace("{city2}", random.choice(cities))
    text = text.replace("{branch}", random.choice(branches))
    text = text.replace("{acc}", f"DE{random.randint(10000000, 99999999)}")
    return text


def generate_dataset(samples_per_class: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic banking risk dataset."""
    random.seed(seed)
    np.random.seed(seed)

    rows = []
    for label_id, templates in TEMPLATES.items():
        for _ in range(samples_per_class):
            template = random.choice(templates)
            sentence = _fill_template(template)
            rows.append({"sentence": sentence, "label": label_id, "label_name": LABEL_MAP[label_id]})

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(f"Generated {len(df)} samples across {len(LABEL_MAP)} classes")
    logger.info(f"Label distribution:\n{df['label_name'].value_counts()}")
    return df


def prepare_splits(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15):
    """
    Splits the dataset into train, validation, and test sets.
    Stratified split ensures equal label distribution across all three sets.
    Default: 70% train / 15% val / 15% test.
    """
    logger.info("Splitting into train / val / test...")
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=42
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size / (1 - test_size),
        stratify=train_df["label"],
        random_state=42,
    )
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir: str = "data/processed"):
    """
    Saves train, validation, and test splits as CSV files.
    Also saves a labels.csv with the id-to-label mapping for reference.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    label_df = pd.DataFrame([{"id": k, "label": v} for k, v in LABEL_MAP.items()])
    label_df.to_csv(f"{output_dir}/labels.csv", index=False)
    logger.info(f"Splits saved to {output_dir}/")


if __name__ == "__main__":
    # Entry point: generate dataset, split, and save to disk
    df = generate_dataset(samples_per_class=300)
    train_df, val_df, test_df = prepare_splits(df)
    save_splits(train_df, val_df, test_df)
    logger.info("Data preparation complete.")

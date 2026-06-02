# Financial Fraud & Risk Text Classifier

Ein NLP-Klassifikationsmodell für Banking-Risikotexte, gebaut mit HuggingFace Transformers und FinBERT. Das Modell klassifiziert Finanztexte automatisch in Risikokategorien — relevant für Compliance, AML und Fraud Detection in Banken.

## Tech Stack

- **Model:** FinBERT (ProsusAI/finbert) — auf Finanztexten vortrainiertes BERT
- **Training:** HuggingFace Transformers, Trainer API
- **Tracking:** MLflow
- **Data:** HuggingFace Datasets
- **Testing:** pytest

## Features

## Fraud & Risk Klassifikation

Texte werden automatisch in fünf Risikokategorien eingestuft:

```
FRAUD            → Phishing, Identitätsdiebstahl, Betrugsindikator
AML_RISK         → Geldwäscheverdacht, Structuring, Layering
CREDIT_RISK      → Zahlungsausfall, Insolvenzindikator
OPERATIONAL_RISK → Systemfehler, Prozessversagen, Policy-Verstoß
LOW_RISK         → Normale, konforme Finanzkommunikation
```

## FinBERT als Basismodell

Statt generischem BERT wird FinBERT verwendet — vortrainiert auf SEC-Filings, Earnings Calls und Finanznachrichten. Das gibt dem Modell ein starkes Domänenwissen für Banking-Sprache von Anfang an.

## Class-Weighted Loss

Weil Fraud-Texte in echten Datensätzen seltener vorkommen als Low-Risk-Texte, wird der Trainingsverlust automatisch gewichtet. Das verhindert, dass das Modell einfach immer "LOW_RISK" vorhersagt.

## MLflow Experiment Tracking

Jeder Trainingslauf wird automatisch geloggt — Hyperparameter, F1 pro Klasse, Precision, Recall und Confusion Matrix. Vergleichbar mit dem networksecurity-Projekt.

## Projektstruktur

```
fraud-risk-classifier/
├── src/
│   ├── data_preparation.py     # Dataset laden, Labels mappen, Splits erstellen
│   ├── train.py                # Fine-Tuning mit HuggingFace Trainer
│   ├── evaluate.py             # Metriken + Confusion Matrix
│   ├── predict.py              # Inference auf einzelne Texte oder CSV-Dateien
│   └── utils.py                # Logger, Hilfsfunktionen
├── app/
│   └── demo.py                 # Interaktives CLI-Demo
├── tests/
│   └── test_predict.py         # Unit Tests für die Inference-Pipeline
├── requirements.txt
└── setup.py
```

## Setup

### 1. Repository klonen

    git clone https://github.com/lisak01/fraud-risk-classifier.git
    cd fraud-risk-classifier

### 2. Abhängigkeiten installieren

    pip install -r requirements.txt
    oder
    python -m pip install -r requirements.txt

### 3. Daten vorbereiten

    python src/data_preparation.py


### 4. Modell trainieren

#### Variante A — Default

    python src/train.py

#### Variante B — eigene Parameter

    python src/train.py --model_name ProsusAI/finbert --epochs 5 --batch_size 8 --lr 3e-05


### 5. Modell evaluieren

#### Variante A — Default

    python src/evaluate.py

#### Variante B — expliziter Pfad

    python src/evaluate.py --model_dir models/finbert-fraud-risk


Terminal:

![python src:evaluate-py.png](python%20src:evaluate-py.png)


### 6. Unit Tests

    pytest tests/test_predict.py -v

### 7. Einzeltext klassifizieren

    python src/predict.py --text "Customer deposits 9900 EUR in cash every week to avoid the reporting threshold."

Ausgabe:

```
Label:      AML_RISK
Confidence: 94.7%
Assessment: Anti-money laundering risk
```

### 8. Batch — ganze CSV klassifizieren
    
    python src/predict.py --input_file data/processed/test.csv --output_file predictions.csv

### 9. Demo — keine Argumente, zeigt Beispieltexte
    
    python app/demo.py

Terminal:

![python app:demo-py.png](python%20app:demo-py.png)

### 10. MLflow UI
    
    mlflow ui

#### Browser: http://127.0.0.1:5000




mlflow Finbert-ep3 
Metrics and parameters

![ml flow finbert-ep3 metrics and parameters.png](ml%20flow%20finbert-ep3%20metrics%20and%20parameters.png)


mlflow Finbert-ep3 
Model metrics

![mlflow finbert-ep3 model metrics.png](mlflow%20finbert-ep3%20model%20metrics.png)



mlflow evaluation
Metrics

![ml flows evaluation metrics.png](ml%20flows%20evaluation%20metrics.png)


mlflow evaluation
Model metrics

![ml flow model metrics.png](ml%20flow%20model%20metrics.png)


## Skripte im Überblick

- `data_preparation.py` — Rohdaten laden, bereinigen, in Train/Val/Test aufteilen
- `train.py` — Fine-Tuning starten, Modell speichern
- `evaluate.py` — Testset auswerten, Confusion Matrix exportieren
- `predict.py` — Einzeltext oder CSV klassifizieren

## Relevanz für Banken

FinBERT und alle verwendeten Modelle sind Open Source — das Modell kann vollständig on-premise betrieben werden, ohne dass Daten das Bankennetzwerk verlassen. Besonders relevant vor dem Hintergrund von DSGVO und den Datenschutzanforderungen im deutschen Bankensektor.


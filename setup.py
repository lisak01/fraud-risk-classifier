from setuptools import setup, find_packages

setup(
    name="fraud-risk-classifier",
    version="1.0.0",
    author="Elizaveta",
    description="Financial fraud & risk text classification using fine-tuned FinBERT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "transformers>=4.40.0",
        "datasets>=2.19.0",
        "torch>=2.2.0",
        "scikit-learn>=1.4.0",
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "mlflow>=2.12.0",
    ],
)

# Credit Default Risk Intelligence

End-to-end machine-learning project for estimating next-month credit-card default risk using the UCI **Default of Credit Card Clients** dataset.

## Project highlights
- 30,000 historical client records
- Leakage-safe preprocessing pipeline
- Stratified train/test split
- 5-fold cross-validation
- Logistic Regression, Random Forest, HistGradientBoosting, XGBoost
- CV-based model selection
- Validation-based threshold tuning
- ROC-AUC, PR-AUC, precision, recall, F1
- Permutation feature importance
- Basic subgroup/fairness audit
- Streamlit demo
- Reproducible GitHub Actions training workflow

## Measured results
The build workflow trains the models and writes the executed metrics to `outputs/metrics.json`.

The current executed build selected **HistGradientBoosting** with approximately:
- Test ROC-AUC: **0.7800**
- Test PR-AUC: **0.5525**
- Test Recall: **0.5818**
- Test F1: **0.5456**
- Tuned operating threshold: **0.265**

These are measured pipeline outputs, not placeholder values.

## Run locally
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Dataset
UCI Machine Learning Repository — *Default of Credit Card Clients*  
I-Cheng Yeh · DOI: 10.24432/C55S3H · CC BY 4.0

## Responsible use
This repository is an educational portfolio project, not a production underwriting system. The data is historical (Taiwan, 2005) and is not representative of every modern lending population.

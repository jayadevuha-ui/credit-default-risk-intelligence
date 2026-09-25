# Project Documentation

## Objective
Estimate next-month credit-card default risk while demonstrating reproducible model development.

## Method
1. Load the public UCI dataset.
2. Normalize category codes.
3. Stratified 80/20 train-test split.
4. Fit preprocessing inside sklearn pipelines.
5. Compare multiple classifiers with 5-fold stratified CV.
6. Select by mean CV ROC-AUC.
7. Tune the operating threshold on a validation split from training data.
8. Refit on full training data and evaluate on untouched test data.
9. Persist the model, metrics, comparison table and interpretability outputs.

## Primary metrics
ROC-AUC and PR-AUC are emphasized because the positive class is a minority. Precision, recall, F1 and confusion matrix are also reported.

## Responsible-use note
This is a portfolio artifact. Real lending systems require current representative data, calibration, regulatory review, governance, monitoring and human oversight.

# Model Card

## Intended use
Educational demonstration of a complete binary-classification workflow.

## Out-of-scope use
Do not use this model to approve, deny, or price real credit.

## Training data
UCI Default of Credit Card Clients, historical Taiwan data from 2005.

## Model selection
Candidate models are compared by 5-fold stratified cross-validation ROC-AUC on the training split.

## Evaluation
The chosen model is evaluated on a held-out 20% test split. Threshold selection uses a validation subset of the training data.

## Limitations
Historical population, limited variables, possible subgroup disparities, and no claim of contemporary calibration or production suitability.

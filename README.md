Telco Customer Churn Classifier

Stacking Ensemble | Precision: 0.57 | Recall: 0.70 | F1: 0.63


Project Overview

A machine learning pipeline that predicts whether a telecom customer will churn (cancel their service). Built as a stacking ensemble — four base models whose predictions are combined by a meta-model — with threshold tuning optimised for recall, because missing a churner is more costly than a false alarm.


Business Problem

Acquiring a new customer costs 5–7× more than retaining an existing one. If a company can predict which customers are about to leave, it can intervene — with a discount, a call, or a better plan — before they cancel.

The challenge: only 27% of customers in this dataset churned. A naive model that predicts "stays" for everyone gets 73% accuracy while being completely useless. This project evaluates models on precision and recall, not accuracy.


Dataset

Telco Customer Churn — IBM Watson / Kaggle

7,043 customers × 21 columns

Column TypeExamplesDemographicsgender, SeniorCitizen, Partner, DependentsAccount infotenure, Contract, PaperlessBilling, PaymentMethodServicesPhoneService, InternetService, OnlineSecurity, StreamingTVBillingMonthlyCharges, TotalChargesTargetChurn (Yes/No)

Class distribution: 73% stayed, 27% churned — imbalanced dataset.


Pipeline Architecture

Raw Data (7,043 customers)
        ↓
Feature Engineering & Preprocessing
        ↓
    Train set (5,625)     Test set (1,407)
        ↓
┌─────────────────────────────────┐
│         Base Models             │
│  LightGBM   XGBoost             │
│  CatBoost   Random Forest       │
│  (OOF predictions, 5-fold CV)   │
└─────────────────────────────────┘
        ↓
Out-of-Fold Prediction Matrix (5625 × 4)
        ↓
Meta-Model (Logistic Regression)
        ↓
Threshold Tuning (0.5 → 0.318)
        ↓
Final Prediction


Feature Engineering

Preprocessing steps:


Dropped customerID — unique identifier, zero predictive signal
Fixed TotalCharges — stored as string with 11 hidden nulls; converted to numeric, dropped nulls
Binary encoding — Yes/No columns → 1/0 (Partner, Dependents, PhoneService, etc.)
One-hot encoding — multi-class columns → binary columns (InternetService, Contract, PaymentMethod)
Standard scaling — numerical columns normalised to mean=0, std=1 (tenure, MonthlyCharges, TotalCharges)


Final feature matrix: 7,032 customers × 23 features


Modelling Approach

Why Stacking?

Each model learns differently:


Random Forest — builds many decorrelated decision trees, robust to noise
XGBoost — gradient boosting, corrects errors sequentially, strong on tabular data
LightGBM — faster gradient boosting, leaf-wise growth, handles large datasets efficiently
CatBoost — gradient boosting with native categorical handling, reduces overfitting


No single model sees the full picture. The meta-model (Logistic Regression) learns to weight each base model's predictions — trusting some more than others depending on the situation.

Out-of-Fold (OOF) Predictions

Base model predictions were generated using 5-fold cross-validation (cross_val_predict). Each customer's prediction came from a model that never saw that customer during training — preventing data leakage into the meta-model.


Threshold Tuning

Default classification threshold is 0.5. For churn, this is the wrong choice.

By lowering the threshold to 0.318, the model flags a customer as churning at a lower confidence level — catching more churners at the cost of some false alarms.

MetricDefault (0.50)Tuned (0.318)Precision0.620.57Recall0.470.70F10.540.63

Recall improved from 0.47 to 0.70 — the model now catches 70% of churners instead of 47%. In a business context, this means 85 more at-risk customers identified per 1,000, each one a potential retention opportunity.


Final Results

Evaluated once on the held-out test set (1,407 customers):

              precision    recall  f1-score   support

   No churn       0.88      0.81      0.85      1033
      Churn       0.57      0.70      0.63       374

    accuracy                           0.78      1407

113 false negatives — churners the model missed.

158 false positives — loyal customers incorrectly flagged.


Error Analysis — A Case the Model Got Wrong

Customer #3721 — predicted to stay, actually churned.

FeatureValueInterpretationTenureVery lowNew customerMonthlyChargesVery lowMinimal spendInternetServiceNoneNo internet planContractMonth-to-monthNo commitmentPaymentMethodMailed checkLow digital engagementChurn probability0.11Model 89% confident they'd stay

Why the model failed: Every signal pointed to a low-risk customer — new, paying little, minimal services. The model never learned that low engagement + zero commitment + easy cancellation is itself a churn pattern. This customer wasn't invested enough to stay.

Business implication: These silent, low-engagement churners are the hardest to retain precisely because they're invisible to the model. A separate rule-based flag for "month-to-month + no internet + low tenure" could catch this segment.


Key Learnings

Accuracy is misleading on imbalanced data. A model predicting "no churn" for everyone scores 73% accuracy. Our model scores 78% — only marginally better by accuracy, but catches 70% of actual churners vs 0%.

Threshold tuning is not optional. The default 0.5 threshold left 53% of churners undetected. A data-driven threshold choice nearly halved that miss rate.

Stacking adds value when base models disagree. The meta-model learned which base models to trust in different regions of the feature space — something no single model could do alone.


Tech Stack


Python 3, Google Colab
pandas, numpy, scikit-learn
XGBoost, LightGBM, CatBoost
matplotlib



How to Run


Clone this repository
Download dataset from Kaggle
Place WA_Fn-UseC_-Telco-Customer-Churn.csv in the project root
Open churn_classifier.ipynb in Google Colab
Run all cells in order

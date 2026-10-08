# Wellspring: Heart Disease Screening System

A machine learning pipeline that predicts the presence of heart disease from routine clinical measurements, built on the **Cleveland cohort** of the UCI Heart Disease dataset. The project covers data validation, imputation, outlier treatment, encoding, scaling, model comparison with stratified k-fold cross-validation, and final evaluation on a held-out test set.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features Used](#features-used)
- [Pipeline](#pipeline)
- [Models](#models)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Installation](#Installation)
- [Usage](#usage)
- [Saved Artifacts](#saved-artifacts)

---

## Overview

Heart disease is a leading cause of death worldwide, and early screening can improve outcomes. Wellspring builds and compares classifiers that flag patients likely to have heart disease using a small, clinically relevant set of six features.

**Goals**

- Clean and validate clinical data so impossible values do not distort the models
- Compare linear and non-linear classifiers fairly with stratified k-fold cross-validation
- Report performance on a held-out test set, with attention to recall (missed cases are costly in screening)
- Identify which features drive the predictions

---

## Dataset

- **Source:** UCI Heart Disease dataset (`heart_disease_uci.csv`), filtered to the **Cleveland** cohort
- **Size:** 304 patient records, 16 original columns
- **Target:** `num` (0–4 severity) converted to a binary `target`: `0` = no heart disease, `1` = heart disease (`num > 0`)
- **Class balance:** about 54.3% no disease / 45.7% disease

---

## Features Used

Six clinical features were selected for the primary prediction task:

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Biological sex (one-hot encoded) |
| `cp` | Chest pain type: typical angina, atypical angina, non-anginal, asymptomatic (one-hot encoded) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `thalch` | Maximum heart rate achieved |

After one-hot encoding, the model input is **10 features**: `age`, `trestbps`, `chol`, `thalch`, `sex_Female`, `sex_Male`, `cp_asymptomatic`, `cp_atypical angina`, `cp_non-anginal`, `cp_typical angina`.

---

## Pipeline

1. **Load and filter:** read the full UCI CSV and keep only the Cleveland cohort.
2. **Physiological validation:** `trestbps = 0` and `chol = 0` are impossible, so they are set to `NaN` rather than dropping rows. Duplicate records are removed.
3. **Imputation:** `IterativeImputer` estimates missing values from the other features. Categorical columns are temporarily ordinal-encoded for imputation, then mapped back.
4. **Outlier treatment:** IQR-based capping and flooring (winsorization) on `age`, `trestbps`, `chol`, and `thalch`. Boxplots are used to inspect outliers first.
5. **Encoding:** binary target from `num`, ordinal encoding for exploration, and one-hot encoding of `sex` and `cp` for modeling.
6. **Correlation analysis:** heatmap and correlation-with-target ranking.
7. **Scaling:** `StandardScaler` (z-score standardization).
8. **Split:** 80/20 stratified train/test split (`random_state=107`): 243 training rows, 61 test rows.
9. **Cross-validation:** 5-fold stratified CV (`shuffle=True`, `random_state=42`) on the training set to compare models.
10. **Final training and evaluation:** models are fit on the training set and evaluated once on the held-out test set.
11. **Interpretability:** Random Forest feature importances.

---

## Models

| Model | Key settings |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | `max_depth=4` in CV; default depth in the held-out run |
| Random Forest | `n_estimators=200`; `max_depth=4` in CV; `class_weight='balanced'` and default depth in the held-out run |

A default **0.5 probability threshold** is used to convert probabilities into predictions.

---

## Results

### 5-fold stratified cross-validation (training set, accuracy)

| Model | Mean CV Accuracy | Std |
|---|---|---|
| Logistic Regression | 0.7737 | ± 0.0220 |
| Decision Tree | 0.7041 | ± 0.0437 |
| Random Forest | 0.7493 | ± 0.0419 |

### Training accuracy (for overfitting check)

| Model | Training Accuracy |
|---|---|
| Logistic Regression | 0.7819 |
| Decision Tree | 0.8354 |
| Random Forest | 0.8519 |

### Held-out test set (61 patients)

| Model | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.836 | 0.821 | 0.821 | 0.821 | 28 | 5 | 5 | 23 |
| Decision Tree | 0.721 | 0.690 | 0.714 | 0.702 | 24 | 9 | 8 | 20 |
| **Random Forest** | **0.852** | 0.806 | **0.893** | **0.847** | 27 | 6 | 3 | 25 |

**Takeaway:** Random Forest achieves the best test accuracy, recall, and F1, and misses the fewest true cases (3 false negatives), which matters most in a screening setting. Logistic Regression is a close second and is the most stable model in cross-validation.

---

## Limitations

- **Small dataset:** 304 rows means the 61-row test set gives noisy estimates; one or two patients shift the metrics noticeably.
- **Single cohort:** results come from the Cleveland data only and may not generalize to other populations.

---

## Future Work

- Use identical model settings for CV and final evaluation, and add hyperparameter search (e.g. `GridSearchCV`)
- Include the other UCI cohorts (Hungarian, Switzerland, Long Beach VA) and additional clinical features
- Explore more advanced feature engineering and models

## Installation

git clone https://github.com/Yunnaan/Wellspring-Heart-Disease.git
cd < Wellspring-Heart-Disease

python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter


## Usage

1. Download `heart_disease_uci.csv` from the UCI Heart Disease dataset and place it where the notebook expects it. 
3. Run the cells in order: imports → EDA → cleaning → imputation → outliers → encoding → scaling → split → cross-validation → training → evaluation.

---

## Saved Artifacts

The notebook saves reusable preprocessing objects with `joblib`:

- `standard_scaler.joblib`: the fitted `StandardScaler`
- `outlier_fences.joblib`: IQR lower and upper fences for each numeric feature

# hospital-readmission-ml

End-to-end machine learning project for predicting whether a patient with diabetes will be readmitted to a hospital within 30 days of discharge.

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008). It covers data validation, exploratory analysis, leakage-safe patient-level splitting, baseline modeling, CatBoost training, threshold selection, and evaluation on a held-out test set.

> **Important:** this is a research/educational project. The model is not validated for clinical use and must not be used as a medical decision-making system.

## Problem statement

The target is binary:

- `1` — the patient was readmitted in less than 30 days (`readmitted == "<30"`);
- `0` — no readmission within 30 days.

The dataset is imbalanced: only **11.16%** of the original encounters belong to the positive class. For that reason, the project focuses on **Average Precision (PR-AUC)** and recall in addition to ROC-AUC and accuracy.

## Dataset

| Property | Value |
|---|---:|
| Raw encounters | 101,766 |
| Raw columns | 50 |
| Unique patients | 71,518 |
| Positive cases (`<30`) | 11,357 |
| Positive rate | 11.16% |
| Exact duplicate rows | 0 |
| Duplicate encounter IDs | 0 |

Important data-quality findings:

- `weight` is missing in 96.86% of rows and is removed;
- `max_glu_serum` and `A1Cresult` are highly sparse but retained for modeling;
- `examide` and `citoglipton` are constant and excluded;
- encounters ending in death, hospice, or related discharge dispositions are excluded from the modeling cohort;
- patient and encounter identifiers are not used as predictors.

After cohort preparation, the modeling dataset contains **99,343 encounters from 69,990 patients**.

## Leakage-safe validation

The same patient can appear in multiple encounters. A random row-level split would therefore leak patient information across datasets. This project uses `patient_nbr` only as a grouping key:

| Split | Encounters | Patients | Positive rate |
|---|---:|---:|---:|
| Train | 69,519 | 48,993 | 11.39% |
| Validation | 14,911 | 10,498 | 11.67% |
| Test | 14,913 | 10,499 | 11.12% |

The train, validation, and test patient sets are mutually disjoint. Five-fold model comparison also uses `StratifiedGroupKFold`.

## Modeling

Two main approaches are compared:

1. **Logistic Regression baseline**
   - median imputation and standard scaling for numeric features;
   - constant-value imputation and one-hot encoding for categorical features;
   - decision threshold selected on the validation set.
2. **CatBoost**
   - native handling of categorical features;
   - 1,000 trees, depth 6, learning rate 0.05;
   - early stopping on the validation set;
   - balanced and screening decision thresholds.

### Cross-validation results

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.649 ± 0.007 | 0.208 ± 0.008 |
| CatBoost | **0.677 ± 0.006** | **0.239 ± 0.008** |

CatBoost is selected as the main model because it improves both ranking metrics while remaining stable across patient-grouped folds.

## Test results

The final CatBoost model is evaluated under two operating scenarios. Both thresholds are selected using the validation set and then applied unchanged to the held-out test set.

| Scenario | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 | Alert rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced F1 | 0.154 | 0.683 | 0.239 | 0.228 | 0.395 | **0.289** | 19.3% |
| Screening (recall ≥ 60% on validation) | 0.112 | 0.683 | 0.239 | 0.179 | **0.590** | 0.275 | 36.6% |

The balanced scenario flags about one in five encounters and detects roughly 40% of 30-day readmissions. The screening scenario detects about 59%, but more than one in three encounters requires follow-up. The final operating threshold should therefore be chosen from the real cost of missed readmissions and the available clinical review capacity.

## Repository structure

```text
hospital-readmission-ml/
├── data/
│   ├── raw/                    # downloaded UCI data (not committed)
│   ├── interim/                # prepared modeling dataset (not committed)
│   └── processed/
├── docs/
│   └── uci_variables.csv       # UCI data dictionary
├── models/                     # serialized model artifacts (not committed)
├── notebooks/
│   ├── 01_missingness_analysis.ipynb
│   ├── 02_modeling_baseline.ipynb
│   └── 03_catboost.ipynb
├── reports/
│   ├── data_quality_summary.csv
│   ├── missingness.csv
│   └── metrics/
│       └── logistic_baseline.json
├── src/readmission/data/
│   ├── download_data.py
│   └── validate_data.py
├── tests/
├── requirements-lock.txt
└── requrements.txt
```

## Installation

Python 3.11+ is recommended.

```bash
git clone https://github.com/Nestorbi219/hospital-readmission-ml.git
cd hospital-readmission-ml

python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requrements.txt
```

The filename `requrements.txt` reflects the current repository. Renaming it to `requirements.txt` is recommended.

## Reproducing the analysis

### 1. Download the UCI dataset

```bash
python src/readmission/data/download_data.py
```

This creates:

- `data/raw/diabetic_data.csv`;
- `docs/uci_variables.csv`.

### 2. Validate the raw data

```bash
python src/readmission/data/validate_data.py
```

The validation report is written to `reports/`.

### 3. Run the notebooks in order

```bash
cd notebooks
jupyter lab
```

Run:

1. `01_missingness_analysis.ipynb` — EDA, cohort preparation, grouped train/validation/test split, and creation of `data/interim/modeling_dataset.parquet`;
2. `02_modeling_baseline.ipynb` — dummy and logistic-regression baselines, threshold tuning, metrics, and `models/logistic_baseline.joblib`;
3. `03_catboost.ipynb` — grouped cross-validation, CatBoost training, threshold scenarios, and final test evaluation.

## Current limitations

- The CatBoost notebook contains executed results but does not currently persist the final CatBoost model or its metrics to the repository.
- The final test cell expects a `selected_thresholds` dictionary that is not defined in the committed notebook source. To reproduce the displayed scenarios, define it before the test evaluation:

  ```python
  selected_thresholds = {
      "balanced_f1": float(best_threshold),
      "screening_recall_60": float(
          threshold_scenarios.loc[
              threshold_scenarios["target_recall"].eq(0.60),
              "threshold",
          ].iloc[0]
      ),
  }
  ```

- The repository is currently notebook-oriented; automated training, inference/API serving, Docker configuration, database integration, and substantive tests have not yet been implemented.
- Dataset features are historical (1999–2008), so model performance and calibration should not be assumed to transfer to present-day hospitals or other patient populations.
- Model outputs indicate statistical risk, not causality, and subgroup fairness has not yet been evaluated.

## Next steps

- move preprocessing and training logic from notebooks into reusable modules under `src/readmission/`;
- save the final CatBoost artifact, thresholds, feature schema, and test metrics;
- add unit and integration tests for download, validation, preprocessing, and inference;
- add probability calibration and subgroup performance checks;
- add a command-line training and prediction interface;
- introduce CI and reproducible environment checks.

## License

Project code is distributed under the [MIT License](LICENSE).

The UCI dataset is distributed separately under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and should be cited as:

> Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999–2008*. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J

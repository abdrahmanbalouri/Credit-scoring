# Credit Scoring

A machine-learning project for predicting whether a loan applicant is likely to default. The project uses the Home Credit Default Risk dataset, performs feature engineering and preprocessing, and trains a class-balanced random forest evaluated with ROC AUC.

**Username:** `ismailhajji`

## Project structure

```text
.
├── data/
│   ├── application_train.csv
│   ├── application_test.csv
│   └── processed/
├── feature_engineering/
│   └── EDA.ipynb
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── results/
│   ├── clients_outputs/
│   ├── model/
│   └── prediction.csv
├── README.md
├── requirements.txt
└── username.txt
```

## Pipeline

The preprocessing script:

- replaces the anomalous `DAYS_EMPLOYED` value (`365243`) with a missing value;
- creates age, employment, income, credit, annuity, and external-score features;
- removes features with more than 65% missing values in the training split;
- imputes numeric and categorical missing values using training-split statistics; and
- one-hot encodes categorical variables using training-split categories, while aligning all columns.

Preprocessing recreates the same split as training before learning these statistics, keeping validation rows out of preprocessing decisions. Undefined ratios caused by zero denominators are treated as missing values.

The training script uses a stratified 80/20 split and fits a random forest with balanced class weights, adding up to 110 trees. Trees are limited to depth 10, with at least 500 samples per leaf and 1,000 samples before splitting. These settings were selected by comparing three candidates on an internal split of the training partition. It stops early after 10 consecutive trees without validation ROC AUC improvement and saves the model with the best validation AUC plus a learning-curve chart under `results/model/`.

Reproduce the regularization comparison with `python scripts/regularization.py`. Its internal selection scores are distinct from the outer validation AUC reported by `predict.py`. The search results, training/validation curve values, and historical baseline are saved under `results/model/`. The model report compares the measured gaps; a small gap is evidence of reduced overfitting on this split, not a guarantee of future performance.

## Getting started

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```powershell
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Add the data

Download the Home Credit Default Risk data and place at least these files in `data/`:

```text
data/application_train.csv
data/application_test.csv
```

The raw and processed datasets are intentionally ignored by Git because of their size.

### 4. Preprocess the data

Run commands from the repository root:

```bash
python scripts/preprocess.py
```

This creates:

```text
data/processed/train_clean.csv
data/processed/test_clean.csv
```

### 5. Train the model

```bash
python scripts/train.py
```

The script prints the reference ROC AUC (`0.66`) and calculated validation ROC AUC, updates the model report with the learning-curve stopping decision, and creates:

```text
results/model/my_own_model.pkl
results/model/learning_curve.png
results/model/model_report.txt
results/model/training_history.csv
```

## Exploratory analysis

Start Jupyter and open `feature_engineering/EDA.ipynb`:

```bash
jupyter notebook
```

## Existing model artifacts

The `results/model/` directory contains the serialized random-forest model, its learning curve, a global feature-importance plot, and the methodology report. Serialized models should only be loaded from trusted sources and should use a compatible scikit-learn version.

## Current limitations

The prediction/reporting script creates `results/prediction.csv`, the global feature-importance plot, and the three client PDF reports. The project does not currently include the optional Dash dashboard.

Run `python scripts/predict.py` after preprocessing and training to generate these outputs. The CSV's `TARGET` column and the client reports contain the predicted probability of default (`TARGET=1`), from 0 to 1. Class predictions are used only to select correctly and incorrectly classified training examples for the reports.

The prediction script also calculates validation ROC AUC and updates `results/model/model_report.txt`. It recreates the training script's 20% stratified holdout with `random_state=42`; use the same processed training data and row order used to train the saved model.

Each client PDF has an HTML companion in `results/clients_outputs/`. Open the HTML files in a browser for offline Plotly charts of key client/loan variables and comparisons with other clients. The PDFs and HTML reports include the probability, SHAP force plot, strongest contributions, and an explanation of whether the prediction matches the known outcome. Test outcomes are unknown.

Both EDA notebooks have an appended feature experiment log. Reproduce the measured comparisons with `python scripts/experiments.py`; results are saved to `results/feature_engineering/experiments.csv`. The exploratory comparison uses fixed samples and identical forest settings with and without engineered features. Its scores are distinct from the saved model's full validation AUC. Existing EDA cells are retained.

## Dataset

This project is designed around Kaggle's **Home Credit Default Risk** dataset. Follow the dataset's original terms when downloading or redistributing its files.

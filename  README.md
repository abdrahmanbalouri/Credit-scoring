# Credit Scoring

A machine-learning project for predicting whether a loan applicant is likely to default. The project uses the Home Credit Default Risk dataset, performs feature engineering and preprocessing, and trains a class-balanced random forest evaluated with ROC AUC.

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
│   ├── dashboard/
│   └── model/
└── requirements.txt
```

## Pipeline

The preprocessing script:

- replaces the anomalous `DAYS_EMPLOYED` value (`365243`) with a missing value;
- creates age, employment, income, credit, annuity, and external-score features;
- removes features with more than 65% missing training values;
- imputes numeric and categorical missing values using training-data statistics; and
- one-hot encodes categorical variables while aligning the train and test columns.

The training script uses a stratified 80/20 split and fits a random forest with balanced class weights. It saves the trained model and a learning-curve chart under `results/model/`.

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

The script prints the validation ROC AUC and creates:

```text
results/model/random_forest.pkl
results/model/learning_curve.png
```

## Exploratory analysis

Start Jupyter and open `feature_engineering/EDA.ipynb`:

```bash
jupyter notebook
```

## Existing model artifacts

The `results/model/` directory contains serialized random-forest and XGBoost models as well as learning curves and feature-importance plots from previous experiments. Serialized models should only be loaded from trusted sources and may require the same library versions used during training.

## Current limitations

The experimental prediction/reporting script and Dash dashboard are not part of the reproducible workflow above. They currently reference legacy preprocessing functions and a model filename that are not produced by the current training pipeline. They also require optional packages that are not listed in `requirements.txt` (including SHAP, Dash, Plotly, Jinja2, and xhtml2pdf).

## Dataset

This project is designed around Kaggle's **Home Credit Default Risk** dataset. Follow the dataset's original terms when downloading or redistributing its files.

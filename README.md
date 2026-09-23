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
results/model/my_own_model.pkl
results/model/learning_curve.png
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

## Dataset

This project is designed around Kaggle's **Home Credit Default Risk** dataset. Follow the dataset's original terms when downloading or redistributing its files.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
output_dir = data_dir / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(data_dir / "application_train.csv")
test = pd.read_csv(data_dir / "application_test.csv")

target = train.pop("TARGET")
train_ids = train.pop("SK_ID_CURR")
test_ids = test.pop("SK_ID_CURR")

# 365243 is a placeholder, not a real number of employment days.
for data in [train, test]:
    # data["DAYS_EMPLOYED_ANOMALOUS"] = (data["DAYS_EMPLOYED"] == 365243).astype("int8")
    data["DAYS_EMPLOYED"] = data["DAYS_EMPLOYED"].replace(365243, np.nan)

    data["AGE_YEARS"] = -data["DAYS_BIRTH"] / 365.25
    data["EMPLOYMENT_YEARS"] = -data["DAYS_EMPLOYED"] / 365.25
    # Undefined ratios become missing values and are imputed below.
    data["CREDIT_INCOME_RATIO"] = data["AMT_CREDIT"] / data["AMT_INCOME_TOTAL"].replace(0, np.nan)
    data["ANNUITY_INCOME_RATIO"] = data["AMT_ANNUITY"] / data["AMT_INCOME_TOTAL"].replace(0, np.nan)
    data["CREDIT_ANNUITY_RATIO"] = data["AMT_CREDIT"] / data["AMT_ANNUITY"].replace(0, np.nan)
    data["INCOME_PER_PERSON"] = data["AMT_INCOME_TOTAL"] / data["CNT_FAM_MEMBERS"].replace(0, np.nan)
    data["EXT_SOURCE_MEAN"] = data[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)

columns_to_drop = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_BIRTH",
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
]
train = train.drop(columns=columns_to_drop)
test = test.drop(columns=columns_to_drop)

# Use exactly the training rows selected in train.py. Validation rows must not
# influence missing-value thresholds, imputation statistics, or the vocabulary.
fit_indices, _ = train_test_split(
    train.index, test_size=0.2, random_state=42, stratify=target
)

# Remove columns with more than 65% missing values in the training data.
high_missing = train.columns[train.loc[fit_indices].isna().mean() > 0.65].tolist()
train = train.drop(columns=high_missing)
test = test.drop(columns=high_missing)

categorical_columns = train.select_dtypes(include="object").columns.tolist()
numeric_columns = train.select_dtypes(include="number").columns

# Learn replacements from train only, then apply them to both datasets.
medians = train.loc[fit_indices, numeric_columns].median().fillna(0)
train[numeric_columns] = train[numeric_columns].fillna(medians)
test[numeric_columns] = test[numeric_columns].fillna(medians)
train[categorical_columns] = train[categorical_columns].fillna("Missing")
test[categorical_columns] = test[categorical_columns].fillna("Missing")

feature_columns = pd.get_dummies(
    train.loc[fit_indices], columns=categorical_columns, dtype="int8"
).columns
train = pd.get_dummies(train, columns=categorical_columns, dtype="int8")
test = pd.get_dummies(test, columns=categorical_columns, dtype="int8")
train = train.reindex(columns=feature_columns, fill_value=0)
test = test.reindex(columns=feature_columns, fill_value=0)

train.insert(0, "SK_ID_CURR", train_ids)
train.insert(1, "TARGET", target)
test.insert(0, "SK_ID_CURR", test_ids)

train.to_csv(output_dir / "train_clean.csv", index=False)
test.to_csv(output_dir / "test_clean.csv", index=False)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Files saved in:", output_dir)

from pathlib import Path

import numpy as np
import pandas as pd


project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
output_dir = data_dir / "processed"

train = pd.read_csv(data_dir / "application_train.csv")
test = pd.read_csv(data_dir / "application_test.csv")

target = train.pop("TARGET")
train_ids = train.pop("SK_ID_CURR")
test_ids = test.pop("SK_ID_CURR")

# 365243 is a placeholder, not a real number of employment days.
for data in [train, test]:
    data["DAYS_EMPLOYED_ANOMALOUS"] = (data["DAYS_EMPLOYED"] == 365243).astype("int8")
    data["DAYS_EMPLOYED"] = data["DAYS_EMPLOYED"].replace(365243, np.nan)

    data["AGE_YEARS"] = -data["DAYS_BIRTH"] / 365.25
    data["EMPLOYMENT_YEARS"] = -data["DAYS_EMPLOYED"] / 365.25
    data["CREDIT_INCOME_RATIO"] = data["AMT_CREDIT"] / data["AMT_INCOME_TOTAL"]
    data["ANNUITY_INCOME_RATIO"] = data["AMT_ANNUITY"] / data["AMT_INCOME_TOTAL"]
    data["CREDIT_ANNUITY_RATIO"] = data["AMT_CREDIT"] / data["AMT_ANNUITY"]
    data["INCOME_PER_PERSON"] = data["AMT_INCOME_TOTAL"] / data["CNT_FAM_MEMBERS"]
    data["EXT_SOURCE_MEAN"] = data[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)

# Remove columns with more than 65% missing values in the training data.
high_missing = train.columns[train.isna().mean() > 0.65].tolist()
train = train.drop(columns=high_missing)
test = test.drop(columns=high_missing)

categorical_columns = train.select_dtypes(include="object").columns.tolist()
numeric_columns = train.select_dtypes(include="number").columns

# Learn replacements from train only, then apply them to both datasets.
medians = train[numeric_columns].median()
train[numeric_columns] = train[numeric_columns].fillna(medians)
test[numeric_columns] = test[numeric_columns].fillna(medians)
train[categorical_columns] = train[categorical_columns].fillna("Missing")
test[categorical_columns] = test[categorical_columns].fillna("Missing")

train = pd.get_dummies(train, columns=categorical_columns, dtype="int8")
test = pd.get_dummies(test, columns=categorical_columns, dtype="int8")
train, test = train.align(test, join="left", axis=1, fill_value=0)

train.insert(0, "SK_ID_CURR", train_ids)
train.insert(1, "TARGET", target)
test.insert(0, "SK_ID_CURR", test_ids)

train.to_csv(output_dir / "train_clean.csv", index=False)
test.to_csv(output_dir / "test_clean.csv", index=False)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Files saved in:", output_dir)

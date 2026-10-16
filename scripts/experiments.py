"""Reproducible feature ablation for the EDA appendix; does not replace the model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
ENGINEERED_FEATURES = [
    "AGE_YEARS", "EMPLOYMENT_YEARS", "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO", "CREDIT_ANNUITY_RATIO", "INCOME_PER_PERSON",
    "EXT_SOURCE_MEAN",
]


def run_experiments():
    """Compare features on the same sampled holdout, then audit the saved model."""
    data = pd.read_csv(PROJECT_DIR / "data/processed/train_clean.csv")
    X = data.drop(columns=["SK_ID_CURR", "TARGET"])
    y = data["TARGET"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Fixed stratified samples make this small comparison repeatable and keep
    # the full validation result distinct from the exploratory experiment.
    train_indices = y_train.index
    val_indices = y_val.index
    if len(train_indices) > 30000:
        train_indices, _ = train_test_split(
            train_indices, train_size=30000, random_state=42, stratify=y_train)
    if len(val_indices) > 10000:
        val_indices, _ = train_test_split(
            val_indices, train_size=10000, random_state=42, stratify=y_val)

    rows = []
    for name, use_engineered in [
        ("Prior-only baseline", None),
        ("Forest without engineered features", False),
        ("Forest with engineered features", True),
    ]:
        columns = X.columns if use_engineered is not False else X.columns.difference(ENGINEERED_FEATURES)
        model = DummyClassifier(strategy="prior") if use_engineered is None else RandomForestClassifier(
            n_estimators=30, max_depth=12, min_samples_split=100,
            min_samples_leaf=40, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        )
        model.fit(X.loc[train_indices, columns], y.loc[train_indices])
        auc = roc_auc_score(y.loc[val_indices], model.predict_proba(X.loc[val_indices, columns])[:, 1])
        rows.append(dict(experiment=name, validation_auc=auc,
                         training_rows=len(train_indices), validation_rows=len(val_indices),
                         features=len(columns), evaluation="Fixed exploratory sample"))

    saved = joblib.load(PROJECT_DIR / "results/model/my_own_model.pkl")
    saved.set_params(verbose=0)
    rows.append(dict(
        experiment=f"Saved scoring model ({saved.n_estimators} trees)",
        validation_auc=roc_auc_score(y_val, saved.predict_proba(X_val)[:, 1]),
        training_rows=len(y_train), validation_rows=len(y_val), features=X.shape[1],
        evaluation="Full validation split; different training size from experiments",
    ))
    results = pd.DataFrame(rows)
    results["delta_vs_prior_baseline"] = results["validation_auc"] - rows[0]["validation_auc"]
    output = PROJECT_DIR / "results/feature_engineering/experiments.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    return results


if __name__ == "__main__":
    print(run_experiments().to_string(index=False))

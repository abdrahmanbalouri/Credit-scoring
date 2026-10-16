"""Compare tree regularization within the training partition only."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]


def compare_regularization():
    data = pd.read_csv(PROJECT_DIR / "data/processed/train_clean.csv")
    X = data.drop(columns=["TARGET", "SK_ID_CURR"])
    y = data["TARGET"]
    # Leave the subject's outer validation rows out of parameter selection.
    X_outer, _, y_outer, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_fit, X_select, y_fit, y_select = train_test_split(
        X_outer, y_outer, test_size=0.2, random_state=17, stratify=y_outer)
    rows = []
    for depth, leaf in [(8, 300), (10, 500), (12, 1000)]:
        model = RandomForestClassifier(
            n_estimators=60, max_depth=depth, min_samples_split=2 * leaf,
            min_samples_leaf=leaf, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1)
        model.fit(X_fit, y_fit)
        train_auc = roc_auc_score(y_fit, model.predict_proba(X_fit)[:, 1])
        selection_auc = roc_auc_score(y_select, model.predict_proba(X_select)[:, 1])
        rows.append(dict(
            max_depth=depth, min_samples_leaf=leaf, min_samples_split=2 * leaf,
            n_estimators=60, training_auc=train_auc, selection_auc=selection_auc,
            gap=train_auc - selection_auc))
    results = pd.DataFrame(rows)
    # A 0.03 gap is a project diagnostic, not a guarantee against overfitting.
    eligible = results.loc[results["gap"] <= 0.03]
    if eligible.empty:
        raise ValueError("No candidate met the diagnostic gap limit; inspect the results.")
    chosen = eligible["selection_auc"].idxmax()
    results["selected"] = results.index == chosen
    output = PROJECT_DIR / "results/model/regularization_search.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    return results


if __name__ == "__main__":
    print(compare_regularization().to_string(index=False))

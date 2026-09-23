import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

from scripts.preprocess import load_and_preprocess_data

def main():
    os.makedirs("results/model", exist_ok=True)

    data_path = "data/application_train.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    X, y = load_and_preprocess_data(data_path)

    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    val_preds = model.predict_proba(X_val)[:, 1]
    final_auc = roc_auc_score(y_val, val_preds)
    print(f"Validation AUC Score: {final_auc:.4f}")

    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
   

    print(f"Validation AUC in Learning Curve: {val_mean[-1]:.4f}")

    plt.figure(figsize=(10, 6))
    plt.title("Model Learning Curve (Random Forest)", fontsize=14)
    plt.xlabel("Training Set Size", fontsize=12)
    plt.ylabel("AUC Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

 

    plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training AUC")
    plt.plot(train_sizes, val_mean, 's-', color="orange", label="Validation AUC")

    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/model/learning_curve.png", dpi=300)
    plt.close()

    importances = model.feature_importances_
    feature_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=True)

    plt.figure(figsize=(10, 8))
    feature_imp.tail(15).plot(kind='barh', color='teal')
    plt.title('Global Feature Importance (Top 15)')
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig("results/model/feature_importance.png", dpi=300)
    plt.close()

    model_path = "results/model/my_own_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model successfully saved at: {model_path}")

if __name__ == "__main__":
    main()
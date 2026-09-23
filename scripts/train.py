import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
output_dir = data_dir / "processed"

def plot_learning_curve(train_aucs, val_aucs, output_path):
    trees = range(1, len(train_aucs) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(trees, train_aucs, label="Training AUC")
    plt.plot(trees, val_aucs, label="Validation AUC")

    plt.xlabel("Number of Trees")
    plt.ylabel("ROC AUC")
    plt.title("Random Forest Learning Curve")
    
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    os.makedirs("results/model", exist_ok=True)

    train_pd = pd.read_csv(output_dir / "train_clean.csv")
    
    X = train_pd.drop(columns=['TARGET', "SK_ID_CURR"])
    y = train_pd['TARGET']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=1,
        max_depth=20,
        min_samples_split=100,
        min_samples_leaf=40,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
        warm_start=True
    )

    train_aucs = []
    val_aucs = []
    total_trees = 110

    print(f"Training Random Forest up to {total_trees} trees...")
    
    for i in range(1, total_trees + 1):
        model.set_params(n_estimators=i)
        model.fit(X_train, y_train)
        
        train_preds = model.predict_proba(X_train)[:, 1]
        val_preds = model.predict_proba(X_val)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_preds)
        val_auc = roc_auc_score(y_val, val_preds)
        
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        
    
    final_auc = val_aucs[-1]
    print(final_auc)

    plot_learning_curve(train_aucs, val_aucs, "results/model/learning_curve.png")

    model_path = "results/model/random_forest.pkl"
    joblib.dump(model, model_path)

if __name__ == "__main__":
    main()
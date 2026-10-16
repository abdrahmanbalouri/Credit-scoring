import os
import joblib
import pandas as pd
from copy import deepcopy
import matplotlib
matplotlib.use("Agg")
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
# Selected on an internal split of the training partition; see the EDA notebook.
MODEL_PARAMS = dict(max_depth=10, min_samples_split=1000, min_samples_leaf=500)

def plot_learning_curve(train_aucs, val_aucs, output_path, best_iteration=None):
    trees = range(1, len(train_aucs) + 1)
  
    plt.figure(figsize=(10, 6))
    plt.plot(trees, train_aucs, label="Training AUC")
    plt.plot(trees, val_aucs, label="Validation AUC")
    if best_iteration is not None:
        plt.axvline(best_iteration, color="gray", linestyle="--",
                    label=f"Saved model: {best_iteration} trees")

    plt.xlabel("Number of Trees")
    plt.ylabel("ROC AUC")
    plt.title("Random Forest Learning Curve")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def write_training_report(train_aucs, val_aucs, best_iteration, patience, validation_samples, report_path="./results/model/model_report.txt"):
    report_path = Path(report_path)
    report = report_path.read_text() if report_path.exists() else ""
    report = report.split("\n4. VALIDATION RESULTS\n", 1)[0].rstrip()
    best_index = best_iteration - 1
    stopped_early = len(val_aucs) - best_iteration >= patience
    if stopped_early:
        decision = (
            f"Stopped after {len(val_aucs)} trees: the validation learning curve had no "
            f"improvement for {patience} consecutive trees after its best value."
        )
    else:
        decision = (
            f"Stopped at the configured maximum of {len(val_aucs)} trees; "
            f"the {patience}-tree early-stopping patience was not exhausted."
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        report + "\n\n4. VALIDATION RESULTS\n" + "-" * 70 + "\n"
        + "Reference validation ROC AUC (hard-coded): 0.66\n"
        + f"Calculated validation ROC AUC: {val_aucs[best_index]:.4f}\n"
        + f"Validation samples: {validation_samples}\n"
        + "Evaluation: 20% stratified holdout, random_state=42.\n"
        + "Score: predicted probability of default (TARGET=1).\n"
        + "\n5. LEARNING CURVE AND STOPPING DECISION\n" + "-" * 70 + "\n"
        + "Learning curve: results/model/learning_curve.png (ROC AUC vs. number of trees).\n"
        + decision + "\n"
        + f"Saved the best model at {best_iteration} trees, where validation AUC "
        + f"was highest ({val_aucs[best_index]:.4f}).\n"
        + f"Training AUC at that point: {train_aucs[best_index]:.4f}; "
        + f"train-validation gap: {train_aucs[best_index] - val_aucs[best_index]:.4f}.\n"
        + f"Last evaluated point: training AUC {train_aucs[-1]:.4f}, "
        + f"validation AUC {val_aucs[-1]:.4f}.\n"
        + "The gap measures the difference in training and validation performance; "
        + "it does not by itself prove the absence of overfitting.\n"
        + f"Overfitting controls: max_depth={MODEL_PARAMS['max_depth']}, "
        + f"min_samples_split={MODEL_PARAMS['min_samples_split']}, "
        + f"min_samples_leaf={MODEL_PARAMS['min_samples_leaf']}, "
        + "and validation-based early stopping.\n"
        + "Regularization was compared on an internal 80/20 split (random_state=17) "
        + "of the outer training partition. Preprocessing was fitted on the outer training "
        + "partition; the outer validation rows were excluded from the comparison.\n"
        + "Of three 60-tree candidates, choose the highest internal selection AUC "
        + "among candidates with a training-selection gap <= 0.03. This diagnostic "
        + "limit is not a statistical guarantee. See the comparisons in the EDA notebook.\n"
    )


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
        **MODEL_PARAMS,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
        warm_start=True,
        verbose=0
    )

    train_aucs = []
    val_aucs = []
    total_trees = 110
    patience = 10
    best_auc = float("-inf")
    best_iteration = 0
    best_model = None
    rounds_without_improvement = 0

    
    for i in range(1, total_trees + 1):
        model.set_params(n_estimators=i)
        model.fit(X_train, y_train)
        
        train_preds = model.predict_proba(X_train)[:, 1]
        val_preds = model.predict_proba(X_val)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_preds)
        val_auc = roc_auc_score(y_val, val_preds)
        
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_iteration = i
            # warm_start mutates the forest, so retain an independent snapshot.
            best_model = deepcopy(model)
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

        if rounds_without_improvement >= patience:
            break
        
    
    

    plot_learning_curve(train_aucs, val_aucs, "results/model/learning_curve.png", best_iteration)
    pd.DataFrame({
        "trees": range(1, len(val_aucs) + 1),
        "training_auc": train_aucs, "validation_auc": val_aucs,
        "selected": [i == best_iteration for i in range(1, len(val_aucs) + 1)],
    }).to_csv("results/model/training_history.csv", index=False)



    model_path = "results/model/my_own_model.pkl"
    joblib.dump(best_model, model_path)
    write_training_report(train_aucs, val_aucs, best_iteration, patience, len(y_val))
    print(f"Calculated validation ROC AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()

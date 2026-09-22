from scripts import preprocess
import joblib
import shap 
import pandas as pd

import matplotlib.pyplot as plt

def predict():


    X_test , id = preprocess.predect_processing()

    model = joblib.load("./results/model/my_own_model.pkl")
    preds = model.predict(X_test)

    out = pd.DataFrame({"SK_ID_CURR": id, "TARGET": preds})
    out.to_csv("./results/prediction.csv", index=False)

    print(f"✅ Predictions saved: submission.csv")
    print(f"   Total predictions: {len(preds)}")
    future_import(model , X_test)

def future_import(model, X_test) :
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values,X_test , show=False)
    plt.savefig("./results/model/feature_importance.png", bbox_inches="tight")
    plt.clf()
predict()

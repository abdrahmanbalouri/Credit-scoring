from scripts import preprocess
import joblib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def predict():
    model = joblib.load("./results/model/my_own_model.pkl")


    X_test , id = preprocess.predect_processing()

    preds = model.predict(X_test)

    out = pd.DataFrame({"SK_ID_CURR": id, "TARGET": preds})
    out.to_csv("./results/prediction.csv", index=False)

    print(f"✅ Predictions saved: submission.csv")
    print(f"   Total predictions: {len(preds)}")

predict()

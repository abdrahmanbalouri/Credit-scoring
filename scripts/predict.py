import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xhtml2pdf import pisa
import pdfkit
import jinja2
import base64
from io import BytesIO
import joblib
def select_target_clients(model, X_train, y_train, train_ids, test_ids):

    train_preds = model.predict(X_train)
    
    correct_mask = (train_preds == y_train)
    wrong_mask = (train_preds != y_train)

    correct_train_id = train_ids[correct_mask].iloc[0]
    wrong_train_id = train_ids[wrong_mask].iloc[0]
    test_id = test_ids[0]

    return [
        {"id": correct_train_id, "dataset_type": "train_correct", "pdf_name":"client1_correct_train.pdf"},
        {"id": wrong_train_id, "dataset_type": "train_wrong","pdf_name":"client2_wrong_train.pdf"},
        {"id": test_id, "dataset_type": "test","pdf_name":"lient_test.pdf"}
    ]
def predict():
    df_test_with_id = pd.read_csv("./data/processed/test_clean.csv")
    test_ids = df_test_with_id["SK_ID_CURR"]
    X_test = df_test_with_id.drop(columns=["SK_ID_CURR"])

    model = joblib.load("./results/model/my_own_model.pkl")
    preds = model.predict(X_test)

    out = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": preds})
    out.to_csv("./results/prediction.csv", index=False)
    print(f"✅ Predictions saved: ./results/prediction.csv")

    future_import(model, X_test)

  
    df_train_with_id = pd.read_csv("./data/processed/train_clean.csv")
    train_ids = df_train_with_id["SK_ID_CURR"]
    y_train = df_train_with_id["TARGET"]
    X_train = df_train_with_id.drop(columns=["SK_ID_CURR", "TARGET"])

    feature_cols = [col for col in X_test.columns if col != "SK_ID_CURR"]

    target_clients = select_target_clients(model, X_train, y_train, train_ids, test_ids)

    for client_info in target_clients:
        cid = client_info["id"]
        ctype = client_info["dataset_type"]
        
        df_source = df_test_with_id if ctype == "test" else df_train_with_id

        output_filename = f"./results/clients_outputs/{client_info["pdf_name"]}"
        generate_local_interpretation_pdf(
                model=model,
                customer_id=cid,
                df=df_source,
                id_column="SK_ID_CURR",
                feature_columns=feature_cols,
                output_pdf_path=output_filename
            )
       


def future_import(model, X_test) :
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values,X_test , show=False)
    plt.savefig("./results/model/feature_importance.png", bbox_inches="tight")
    plt.clf()



def generate_local_interpretation_pdf(
    model, 
    customer_id, 
    df, 
    id_column, 
    feature_columns, 
    output_pdf_path="report.pdf"
):
    client_data = df[df[id_column] == customer_id]
    if client_data.empty:
        raise ValueError(f"Customer ID {customer_id} not found in dataset.")
    
    X_client = client_data[feature_columns]
    X_all = df[feature_columns]

    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_client)[0][1]
    else:
        score = model.predict(X_client)[0]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_client)

    plt.figure(figsize=(10, 3))
  # 1. Extract base value safely for binary classification (class 1)
    if isinstance(explainer.expected_value, (np.ndarray, list)):
        base_val = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
    else:
        base_val = explainer.expected_value

    # 2. Extract SHAP values for the single client (row 0)
    vals = shap_values.values if hasattr(shap_values, "values") else shap_values

    # Handle multi-class / 3D output (samples, features, classes) -> take sample 0, class 1
    if vals.ndim == 3:
        sample_shap = vals[0, :, 1]
    elif vals.ndim == 2:
        sample_shap = vals[0]
    else:
        sample_shap = vals

    # Ensure strictly 1D
    sample_shap = sample_shap.ravel()

    # 3. Render force plot
    shap.plots.force(
        base_val,
        sample_shap,
        X_client.iloc[0],
        matplotlib=True,
        show=False
    )
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    shap_base64 = base64.b64encode(image_png).decode('utf-8')

    client_vals = X_client.iloc[0]
    avg_vals = X_all.mean()
    
    comparison_df = pd.DataFrame({
        'Feature': feature_columns,
        'Client Value': client_vals.values,
        'Dataset Average': avg_vals.values
    })

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2C3E50; }
            .score-box { background-color: #ECF0F1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { border: 1px solid #BDC3C7; padding: 8px; text-align: left; }
            th { background-color: #34495E; color: white; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>Local Interpretability Report</h1>
        <div class="score-box">
            <h3>Customer ID: {{ customer_id }}</h3>
            <h2>Prediction Score: {{ "%.4f"|format(score) }}</h2>
        </div>

        <h2>SHAP Force Plot</h2>
        <img src="data:image/png;base64,{{ shap_plot }}" />

        <h2>Client Data vs. Population Average</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Client Value</th>
                <th>Dataset Average</th>
            </tr>
            {% for row in comparison %}
            <tr>
                <td>{{ row['Feature'] }}</td>
                <td>{{ "%.2f"|format(row['Client Value']) if row['Client Value'] is number else row['Client Value'] }}</td>
                <td>{{ "%.2f"|format(row['Dataset Average']) }}</td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """

    template = jinja2.Template(html_template)
    rendered_html = template.render(
        customer_id=customer_id,
        score=score,
        shap_plot=shap_base64,
        comparison=comparison_df.to_dict(orient='records')
    )

    with open(output_pdf_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(rendered_html, dest=pdf_file)

    if pisa_status.err:
        print(f" Error generating PDF: {output_pdf_path}")
    else:
        print(f" Report generated successfully: {output_pdf_path}")
predict()

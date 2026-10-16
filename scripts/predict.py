
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xhtml2pdf import pisa
import jinja2
import base64
from io import BytesIO
import joblib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def evaluate_validation_auc(model, X, y, report_path="./results/model/model_report.txt"):
    # Recreate the held-out rows used by scripts/train.py.
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    probabilities = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probabilities)

    report_path = Path(report_path)
    marker = "\n4. VALIDATION RESULTS\n"
    report = report_path.read_text() if report_path.exists() else ""
    print(report)
    stopping_marker = "\n5. LEARNING CURVE AND STOPPING DECISION\n"
    _, separator, stopping_details = report.partition(stopping_marker)
    stopping_section = separator + stopping_details
    report = report.split(marker, 1)[0].rstrip()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        report + "\n" + marker
        + "-" * 70 + "\n"
        + "Reference validation ROC AUC (hard-coded): 0.66\n"
        + f"Calculated validation ROC AUC: {auc:.4f}\n"
        + f"Validation samples: {len(y_val)}\n"
        + "Evaluation: 20% stratified holdout, random_state=42 (same as train.py).\n"
        + "Score: predicted probability of default (TARGET=1).\n"
        + stopping_section
    )
    return auc


def select_target_clients(model, X_train, y_train, train_ids, test_ids):

    train_preds = model.predict(X_train)
    
    correct_mask = (train_preds == y_train)
    wrong_mask = (train_preds != y_train)

    if not correct_mask.any() or not wrong_mask.any():
        raise ValueError("Client reports require both a correct and an incorrect training prediction.")
    if len(test_ids) == 0:
        raise ValueError("No test clients are available for reporting.")
    correct_train_id = train_ids[correct_mask].iloc[0]
    wrong_train_id = train_ids[wrong_mask].iloc[0]
    test_id = test_ids.iloc[min(1, len(test_ids) - 1)]

    return [
        {"id": correct_train_id, "dataset_type": "train_correct", "pdf_name":"client1_correct_train.pdf"},
        {"id": wrong_train_id, "dataset_type": "train_wrong","pdf_name":"client2_wrong_train.pdf"},
        {"id": test_id, "dataset_type": "test","pdf_name":"client_test.pdf"}
    ]
def predict():
    df_test_with_id = pd.read_csv("./data/processed/test_clean.csv")
    test_ids = df_test_with_id["SK_ID_CURR"]
    X_test = df_test_with_id.drop(columns=["SK_ID_CURR"])

    model = joblib.load("./results/model/my_own_model.pkl")
    model.set_params(verbose=0)
    preds = model.predict_proba(X_test)[:, 1]

    out = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": preds})
    out.to_csv("./results/prediction.csv", index=False)

    feature_importance(model, X_test)


    df_train_with_id = pd.read_csv("./data/processed/train_clean.csv")
    train_ids = df_train_with_id["SK_ID_CURR"]
    y_train = df_train_with_id["TARGET"]
    X_train = df_train_with_id.drop(columns=["SK_ID_CURR", "TARGET"])

    auc = evaluate_validation_auc(model, X_train, y_train)
    print(f"Calculated AUC on validation set: {auc:.4f}")

    target_clients = select_target_clients(model, X_train, y_train, train_ids, test_ids)

    # Retain original financial amounts in charts even when the model uses ratios.
    raw_columns = ["SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]
    chart_sources = {
        "train": pd.read_csv("./data/application_train.csv", usecols=raw_columns),
        "test": pd.read_csv("./data/application_test.csv", usecols=raw_columns),
    }

    for client_info in target_clients:
        cid = client_info["id"]
        ctype = client_info["dataset_type"]
        
        df_source = df_test_with_id if ctype == "test" else df_train_with_id

        output_filename = f"./results/clients_outputs/{client_info['pdf_name']}"
        generate_local_interpretation_pdf(
                model=model,
                customer_id=cid,
                df=df_source,
                id_column="SK_ID_CURR",
                feature_columns=X_train.columns,
                chart_data=chart_sources["test" if ctype == "test" else "train"],
                output_pdf_path=output_filename
            )
       




def feature_importance(model, X_test):
  
    importances = model.feature_importances_
   
    importance_df = (
        pd.DataFrame(
            {"Feature": X_test.columns, "Importance": importances}
        )
        .sort_values(by="Importance", ascending=False)
        .head(20)
    )  

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        palette="viridis",
        hue="Feature",
        legend=False,
    )
    plt.title("Native Feature Importance")
    plt.savefig("./results/model/feature_importance.png", bbox_inches="tight")
    plt.close()


def create_client_charts(client, peers):
    """Show client values and distributions with separate axes for different units."""
    wanted = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AGE_YEARS",
              "CREDIT_INCOME_RATIO", "EMPLOYMENT_YEARS"]
    features = [name for name in wanted if name in peers.columns]
    if not features:
        raise ValueError("No key client features are available for plotting.")
    titles = [name.replace("_", " ").title() for name in features]
    rows = (len(features) + 2) // 3
    key_chart = make_subplots(rows=rows, cols=3, subplot_titles=titles)
    comparison = make_subplots(rows=rows, cols=3, subplot_titles=titles)
    # Bound HTML size; each chart states how many other clients were sampled.
    sample = peers.sample(min(2000, len(peers)), random_state=42)
    for index, feature in enumerate(features):
        row, col = index // 3 + 1, index % 3 + 1
        value = float(client[feature])
        key_chart.add_trace(go.Bar(x=["Client"], y=[value],
                                  text=[f"{value:,.2f}"], textposition="auto",
                                  marker_color="#2563eb", showlegend=False), row=row, col=col)
        comparison.add_trace(go.Box(y=sample[feature].dropna(), name="Other clients",
                                    boxpoints=False, marker_color="#94a3b8",
                                    showlegend=False), row=row, col=col)
        comparison.add_trace(go.Scatter(x=["Other clients"], y=[value], mode="markers",
                                        name="Client", marker=dict(color="#dc2626", size=12),
                                        showlegend=index == 0), row=row, col=col)
    key_chart.update_layout(title="Key client and loan variables", height=rows * 310,
                            template="plotly_white")
    comparison.update_layout(
        title=f"Client (red) vs. {len(sample):,} other clients (distribution)",
        height=rows * 310, template="plotly_white")
    return key_chart, comparison


def explain_prediction(score, predicted_class, actual, base_value, contributions, values):
    """Describe the model's evidence, distinguishing explanations from causes."""
    ranked = pd.Series(contributions, index=values.index).abs().nlargest(8).index
    drivers = [dict(feature=name, value=float(values[name]),
                    contribution=float(contributions[values.index.get_loc(name)]))
               for name in ranked]
    decision = "default" if predicted_class == 1 else "no default"
    explanation = (
        f"The model predicts {decision} with default probability {score:.2%}. "
        f"The class decision uses a 0.50 probability threshold (ties go to class 0). "
        f"The SHAP baseline default probability is {base_value:.2%}; "
        "positive contributions increase default risk and negative contributions reduce it."
    )
    signed = pd.Series(contributions, index=values.index)
    for label, factors in [
        ("Strongest increases in risk", signed[signed > 0].nlargest(3)),
        ("Strongest reductions in risk", signed[signed < 0].nsmallest(3)),
    ]:
        if not factors.empty:
            explanation += " " + label + ": " + "; ".join(
                f"{name} ({contribution * 100:+.2f} percentage points)"
                for name, contribution in factors.items()
            ) + "."
    if actual is not None:
        outcome = "default" if actual == 1 else "no default"
        if predicted_class != actual:
            error = "false positive" if predicted_class == 1 else "false negative"
            explanation += (
                f" This is a {error}: the observed label is {actual} ({outcome}), "
                f"but the predicted class is {predicted_class}. "
                "The contributions below explain the model's incorrect decision. "
                "They do not establish why this client actually repaid or defaulted; "
                "similar feature patterns can have different outcomes."
            )
        else:
            explanation += f" This agrees with the observed label {actual} ({outcome})."
    else:
        explanation += " The test client's outcome is unknown, so correctness cannot be assessed."
    return explanation, drivers


def generate_local_interpretation_pdf(
    model, 
    customer_id, 
    df, 
    id_column, 
    feature_columns, 
    output_pdf_path="report.pdf",
    chart_data=None
):
    client_data = df[df[id_column] == customer_id]
    if len(client_data) != 1:
        raise ValueError(f"Expected one customer with ID {customer_id}; found {len(client_data)}.")
    output_pdf_path = Path(output_pdf_path)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
   
    
    X_client = client_data[feature_columns]
    X_all = df[feature_columns]

    
    score = model.predict_proba(X_client)[0, 1]
    predicted_class = int(model.predict(X_client)[0])
    actual = int(client_data["TARGET"].iloc[0]) if "TARGET" in client_data else None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_client)

    plt.figure(figsize=(10, 3))
    base_val = explainer.expected_value[1]
   

    vals = shap_values.values 

    sample_shap = vals[0, :, 1]
    sample_shap = sample_shap.ravel()
    if not np.isclose(base_val + sample_shap.sum(), score, atol=1e-5):
        raise ValueError("SHAP contributions do not reconstruct the default probability.")
    explanation, drivers = explain_prediction(
        score, predicted_class, actual, base_val, sample_shap, X_client.iloc[0])

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
    chart_frame = df.copy()
    if chart_data is not None:
        raw = chart_data.set_index(id_column)
        for column in raw.columns:
            chart_frame[column] = df[id_column].map(raw[column])
    chart_client = chart_frame.loc[df[id_column] == customer_id].iloc[0]
    peers = chart_frame.loc[df[id_column] != customer_id]
    key_chart, comparison_chart = create_client_charts(chart_client, peers)

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
            <h2>Default Probability (TARGET=1): {{ "%.4f"|format(score) }}</h2>
        </div>

        <h2>Decision explanation</h2>
        <p>{{ explanation }}</p>
        <h3>Largest SHAP contributions to default probability</h3>
        <table>
            <tr><th>Feature</th><th>Client value</th><th>Contribution (percentage points)</th></tr>
            {% for driver in drivers %}
            <tr><td>{{ driver.feature }}</td><td>{{ "%.4f"|format(driver.value) }}</td>
            <td>{{ "%+.2f"|format(driver.contribution * 100) }}</td></tr>
            {% endfor %}
        </table>
        <p>Interactive Plotly client and population charts: {{ html_filename }}</p>

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

    template = jinja2.Environment(autoescape=True).from_string(html_template)
    rendered_html = template.render(
        customer_id=customer_id,
        score=score,
        explanation=explanation,
        drivers=drivers,
        html_filename=output_pdf_path.with_suffix(".html").name,
        shap_plot=shap_base64,
        comparison=comparison_df.to_dict(orient='records')
    )

    # PDFs retain the explanation and force plot; the companion HTML adds
    # interactive Plotly charts with an embedded JS bundle for offline viewing.
 
    
    with open(output_pdf_path, "wb") as pdf_file:
        result = pisa.CreatePDF(rendered_html, dest=pdf_file)
    if result.err:
        raise RuntimeError(f"PDF generation failed for customer {customer_id}.")

    
if __name__ == "__main__":
    predict()

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import shap
import joblib
from scripts import preprocess
model = joblib.load("./results/model/my_own_model.pkl")

X_test, test_ids = preprocess.predect_processing()
X_train, y_train, train_ids = preprocess.preprocess()

df_train = X_train.copy()
df_train["SK_ID_CURR"] = train_ids

df_test = X_test.copy()
df_test["SK_ID_CURR"] = test_ids

feature_cols = [col for col in X_test.columns if col != "SK_ID_CURR"]

explainer = shap.TreeExplainer(model)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Customer Credit Risk Dashboard", className="text-center my-4"), width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Client Search"),
                dbc.CardBody([
                    dbc.Label("Dataset Source:"),
                    dbc.RadioItems(
                        id="dataset-source",
                        options=[
                            {"label": "Test Set", "value": "test"},
                            {"label": "Train Set", "value": "train"}
                        ],
                        value="test",
                        inline=True,
                        className="mb-3"
                    ),
                    dbc.Label("Enter Customer ID (SK_ID_CURR):"),
                    dbc.Input(id="customer-id-input", type="number", placeholder="e.g. 100001", className="mb-3"),
                    dbc.Button("Analyze Client", id="search-btn", color="primary", className="w-100")
                ])
            ], className="shadow-sm")
        ], width=3),

        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Prediction Score / Probability", className="card-title text-muted"),
                            html.H2(id="score-display", children="--", className="text-primary")
                        ])
                    ], className="shadow-sm mb-4")
                ], width=12)
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Local Interpretability (SHAP Force Plot)"),
                        dbc.CardBody([
                            dcc.Graph(id="shap-force-plot")
                        ])
                    ], className="shadow-sm mb-4")
                ], width=12)
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Client Features vs Dataset Average"),
                        dbc.CardBody([
                            html.Div(id="comparison-table-container")
                        ])
                    ], className="shadow-sm mb-4")
                ], width=12)
            ])
        ], width=9)
    ])
], fluid=True)


@app.callback(
    [
        Output("score-display", "children"),
        Output("shap-force-plot", "figure"),
        Output("comparison-table-container", "children")
    ],
    [Input("search-btn", "n_clicks")],
    [
        State("customer-id-input", "value"),
        State("dataset-source", "value")
    ]
)
def update_dashboard(n_clicks, customer_id, dataset_source):
    if not n_clicks or customer_id is None:
        return "--", go.Figure(), html.Div("Enter a Customer ID and click 'Analyze Client'.")

    df_source = df_test if dataset_source == "test" else df_train
    client_data = df_source[df_source["SK_ID_CURR"] == customer_id]

    if client_data.empty:
        return "ID Not Found", go.Figure(), html.Div(f"❌ Customer ID {customer_id} not found in {dataset_source} dataset.", style={"color": "red"})

    X_client = client_data[feature_cols]
    X_all = df_source[feature_cols]

    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_client)[0][1]
    else:
        score = model.predict(X_client)[0]

    score_text = f"{score:.4f}"

    shap_vals = explainer(X_client)
    
    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (np.ndarray, list)) else explainer.expected_value
    values = shap_vals.values[0] if len(shap_vals.values.shape) == 3 else shap_vals.values[0]
    
    top_indices = np.argsort(np.abs(values))[-10:]
    top_features = [feature_cols[i] for i in top_indices]
    top_shap_values = values[top_indices]

    colors = ['#ff0051' if v > 0 else '#008bfb' for v in top_shap_values]

    fig_shap = go.Figure(go.Bar(
        x=top_shap_values,
        y=top_features,
        orientation='h',
        marker_color=colors
    ))
    fig_shap.update_layout(
        title="Top 10 Feature Contributions to Prediction Score",
        xaxis_title="SHAP Value (Impact on prediction)",
        yaxis_title="Feature",
        margin=dict(l=20, r=20, t=40, b=20),
        height=350
    )

    client_vals = X_client.iloc[0]
    avg_vals = X_all.mean()

    comp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Client Value": client_vals.values,
        "Dataset Average": avg_vals.values
    }).round(2)

    table = dash_table.DataTable(
        data=comp_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in comp_df.columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={'backgroundColor': '#34495E', 'color': 'white', 'fontWeight': 'bold'}
    )

    return score_text, fig_shap, table


if __name__ == "__main__":
    app.run(debug=True, port=8050)
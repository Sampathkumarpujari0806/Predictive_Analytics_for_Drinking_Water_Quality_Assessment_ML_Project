"""
Drinking Water Quality & Potability Predictor - Streamlit Front End

Pipeline (same order as your training notebook):
    user inputs -> missing-value median fallback (ph, sulfate, trihalomethanes)
                -> MinMax scaling (9 numeric water-quality columns)
                -> KNeighborsClassifier (or saved model) -> 0 / 1 potability label

Run:
    pip install -r requirements.txt
    streamlit run app.py

Keep water_potability_model.pkl (or model_prod.pkl) in the same folder as this file.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_CANDIDATES = [
    Path(__file__).parent / "model_product.pkl",
]

FEATURE_COLS = [
    "ph",
    "hardness",
    "solids",
    "chloramines",
    "sulfate",
    "conductivity",
    "organic_carbon",
    "trihalomethanes",
    "turbidity",
]

# Fallback medians from water_potability.csv if not saved inside the pickle
DEFAULT_MEDIANS = {
    "ph": 7.036752,
    "sulfate": 333.073546,
    "trihalomethanes": 66.622485,
}

# Fallback MinMax (min, max) ranges from water_potability.csv if scaler is not saved
SCALE_RANGES = {
    "ph": (0.0, 14.0),
    "hardness": (47.432, 323.124),
    "solids": (320.942611, 61227.196008),
    "chloramines": (0.352, 13.127),
    "sulfate": (129.0, 481.030642),
    "conductivity": (181.483754, 753.34262),
    "organic_carbon": (2.2, 28.3),
    "trihalomethanes": (0.738, 124.0),
    "turbidity": (1.45, 6.739),
}

DECODE = {
    0: "Not Potable",
    1: "Potable",
}

CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #f0f9ff 0%, #ffffff 45%);
    }
    .main .block-container {
        max-width: 780px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .app-header {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #ffffff;
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 24px rgba(2, 132, 199, 0.16);
    }
    .app-header h1 {
        color: #ffffff !important;
        font-size: 1.75rem !important;
        margin: 0 0 0.3rem 0 !important;
    }
    .app-header p {
        margin: 0;
        color: #e0f2fe;
        font-size: 0.96rem;
    }
    div[data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.85rem 1rem;
    }
    .stButton > button[kind="primary"] {
        width: 100%;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-weight: 600;
        background-color: #0284c7;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #0369a1;
    }
</style>
"""


def resolve_model_path() -> Path | None:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


@st.cache_resource
def load_artifacts(path_str: str):
    loaded = joblib.load(Path(path_str))
    if isinstance(loaded, dict):
        return (
            loaded.get("model"),
            loaded.get("scaler"),
            loaded.get("medians", DEFAULT_MEDIANS),
        )
    return loaded, None, DEFAULT_MEDIANS


def preprocess(raw: dict, model, scaler, medians: dict) -> pd.DataFrame:
    """Turn user inputs into the exact 9-column scaled DataFrame the model expects."""
    row = {}
    for col in FEATURE_COLS:
        val = raw[col]
        if val is None or pd.isna(val):
            val = medians.get(col, DEFAULT_MEDIANS.get(col, 0.0))
        row[col] = float(val)

    input_df = pd.DataFrame([row])[FEATURE_COLS]

    if scaler is not None:
        scaled_arr = scaler.transform(input_df)
        scaled_df = pd.DataFrame(scaled_arr, columns=FEATURE_COLS)
    else:
        scaled_df = pd.DataFrame(
            [
                {
                    col: (input_df.loc[0, col] - lo) / (hi - lo)
                    for col, (lo, hi) in SCALE_RANGES.items()
                }
            ]
        )[FEATURE_COLS]

    if hasattr(model, "feature_names_in_"):
        return scaled_df[list(model.feature_names_in_)]
    return scaled_df


# ----- UI -----
st.set_page_config(
    page_title="Water Potability Predictor",
    page_icon="💧",
    layout="centered",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header">
        <h1>💧 Drinking Water Potability Predictor</h1>
        <p>Enter physicochemical water quality parameters and click <strong>Predict</strong> to assess potability.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model_path = resolve_model_path()
if model_path is None:
    st.error(
        "No `.pkl` model file found. Place `water_potability_model.pkl` (or `model_prod.pkl`) "
        "in the same folder as `app.py`."
    )
    st.stop()

model, scaler, medians = load_artifacts(str(model_path))

with st.form("potability_form"):
    st.subheader("Water Quality Parameters")
    left, right = st.columns(2)

    with left:
        ph = st.number_input(
            "pH level (0–14)",
            min_value=0.0,
            max_value=14.0,
            value=7.04,
            step=0.01,
        )
        hardness = st.number_input(
            "Hardness (mg/L as CaCO₃)",
            min_value=0.0,
            max_value=500.0,
            value=196.97,
            step=0.5,
        )
        solids = st.number_input(
            "Total Dissolved Solids / Solids (ppm)",
            min_value=0.0,
            max_value=80_000.0,
            value=20_927.83,
            step=100.0,
        )
        chloramines = st.number_input(
            "Chloramines (ppm)",
            min_value=0.0,
            max_value=20.0,
            value=7.13,
            step=0.05,
        )
        sulfate = st.number_input(
            "Sulfate (mg/L)",
            min_value=0.0,
            max_value=700.0,
            value=333.07,
            step=1.0,
        )

    with right:
        conductivity = st.number_input(
            "Conductivity (μS/cm)",
            min_value=0.0,
            max_value=1_200.0,
            value=421.88,
            step=1.0,
        )
        organic_carbon = st.number_input(
            "Organic Carbon (ppm)",
            min_value=0.0,
            max_value=50.0,
            value=14.22,
            step=0.1,
        )
        trihalomethanes = st.number_input(
            "Trihalomethanes (μg/L)",
            min_value=0.0,
            max_value=200.0,
            value=66.62,
            step=0.5,
        )
        turbidity = st.number_input(
            "Turbidity (NTU)",
            min_value=0.0,
            max_value=15.0,
            value=3.96,
            step=0.05,
        )

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    raw = {
        "ph": ph,
        "hardness": hardness,
        "solids": solids,
        "chloramines": chloramines,
        "sulfate": sulfate,
        "conductivity": conductivity,
        "organic_carbon": organic_carbon,
        "trihalomethanes": trihalomethanes,
        "turbidity": turbidity,
    }

    X = preprocess(raw, model, scaler, medians)
    pred = int(model.predict(X)[0])
    label = DECODE.get(pred, str(pred))

    st.divider()
    if pred == 1:
        st.success("✅ Water is **Potable** (safe for drinking based on model prediction)")
    else:
        st.error("❌ Water is **Not Potable** (flagged as unsafe for drinking by model prediction)")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(proba[list(model.classes_).index(pred)])
        c1, c2 = st.columns(2)
        c1.metric("Model output", f"{pred} → {label}")
        c2.metric("Confidence", f"{confidence:.0%}")
    else:
        st.metric("Model output", f"{pred} → {label}")

    with st.expander("Processed features sent to the model"):
        st.dataframe(X.T.rename(columns={0: "scaled_value"}))
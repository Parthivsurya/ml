"""
Kepler KOI Disposition Classifier — Streamlit GUI (Review 2 bonus).

Loads the model bundle exported from classification.ipynb (Section 20) and serves
live predictions of the KOI disposition class (CANDIDATE / CONFIRMED / FALSE POSITIVE).

Run locally:
    streamlit run app/app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PATH = PROJECT_ROOT / "models" / "koi_classifier.joblib"

# Human-readable labels for the features surfaced in the UI.
FEATURE_LABELS = {
    "koi_prad": "Planetary radius (Earth radii)",
    "koi_period": "Orbital period (days)",
    "koi_depth": "Transit depth (ppm)",
    "koi_duration": "Transit duration (hours)",
    "koi_model_snr": "Transit signal-to-noise ratio",
    "koi_insol": "Insolation flux (Earth flux)",
    "koi_steff": "Stellar effective temperature (K)",
    "koi_ror": "Planet-star radius ratio",
    "koi_dor": "Planet-star distance / star radius",
    "transit_depth_per_hour": "Transit depth per hour (engineered)",
    "koi_smet_err2": "Stellar metallicity lower error",
    "koi_fwm_stat_sig": "Centroid offset significance",
    "koi_max_mult_ev": "Max multiple-event statistic",
    "koi_dikco_msky": "Centroid offset (arcsec)",
}

CLASS_HELP = {
    "CONFIRMED": "A validated exoplanet.",
    "CANDIDATE": "A possible planet not yet confirmed or ruled out.",
    "FALSE POSITIVE": "A signal caused by something other than a transiting planet.",
}

st.set_page_config(page_title="Kepler KOI Classifier", page_icon="\U0001FA90", layout="wide")


@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        return None
    return joblib.load(BUNDLE_PATH)


bundle = load_bundle()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("\U0001FA90 Kepler Objects of Interest — Disposition Classifier")
st.caption(
    "23CSE301 Machine Learning Capstone · Classification track. "
    "Predicts whether a Kepler Object of Interest is a CONFIRMED planet, "
    "a CANDIDATE, or a FALSE POSITIVE."
)

if bundle is None:
    st.error(
        "Model bundle not found. Run `classification.ipynb` (Section 20 — Model Export) "
        "first to create `models/koi_classifier.joblib`."
    )
    st.stop()

pipeline = bundle["pipeline"]
features = bundle["features"]
classes = bundle["classes"]
medians = bundle["medians"]
ui_features = bundle["ui_features"]
ui_stats = bundle["ui_stats"]
metrics = bundle.get("metrics", {})
samples = bundle.get("samples", [])

# Model summary strip
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", bundle.get("model_name", "Classifier"))
c2.metric("Test accuracy", f"{metrics.get('Accuracy', float('nan')):.3f}")
c3.metric("Weighted F1", f"{metrics.get('F1_Weighted', float('nan')):.3f}")
c4.metric("ROC-AUC (OvR)", f"{metrics.get('ROC_AUC_OvR', float('nan')):.3f}")

st.divider()


def apply_example(record):
    """Prefill the input widgets from a real held-out example."""
    for feat in ui_features:
        if feat in record and record[feat] is not None and not pd.isna(record[feat]):
            st.session_state[f"in_{feat}"] = float(record[feat])
    st.session_state["_true_label"] = record.get("__true_label__")


# ----------------------------------------------------------------------------
# Sidebar — try a real example
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Try a real example")
    st.write(
        "Load an actual held-out KOI (with its true label) to see the model in action, "
        "then tweak the values on the right."
    )
    if samples:
        idx = st.selectbox(
            "Held-out example #",
            options=list(range(len(samples))),
            format_func=lambda i: f"Example {i + 1} — true: {samples[i].get('__true_label__', '?')}",
        )
        if st.button("Load this example", use_container_width=True):
            apply_example(samples[idx])
            st.success("Example loaded into the inputs.")
    else:
        st.info("No bundled examples available.")

    st.divider()
    st.caption(
        "Only the most influential / interpretable features are shown. "
        "All other model inputs are set to their training-set median."
    )

# ----------------------------------------------------------------------------
# Main — feature inputs
# ----------------------------------------------------------------------------
st.subheader("Object measurements")
st.write("Adjust the measurements below and predict the disposition class.")

input_values = {}
cols = st.columns(2)
for i, feat in enumerate(ui_features):
    stats = ui_stats[feat]
    label = FEATURE_LABELS.get(feat, feat)
    default = float(st.session_state.get(f"in_{feat}", stats["median"]))
    with cols[i % 2]:
        input_values[feat] = st.number_input(
            label,
            value=default,
            key=f"in_{feat}",
            help=f"code: {feat} · typical range {stats['p05']:.3g} to {stats['p95']:.3g}",
            format="%.5g",
        )

predict_clicked = st.button("Predict disposition", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if predict_clicked:
    # Build a full single-row frame: every model feature defaults to its training median,
    # then override with the values the user supplied.
    row = {feat: medians.get(feat, np.nan) for feat in features}
    row.update({feat: input_values[feat] for feat in ui_features})
    X_row = pd.DataFrame([row], columns=features)

    pred = pipeline.predict(X_row)[0]
    proba = pipeline.predict_proba(X_row)[0]
    proba_by_class = dict(zip(pipeline.classes_, proba))
    confidence = proba_by_class[pred]

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Prediction")
        st.markdown(f"### \U0001F52D {pred}")
        st.caption(CLASS_HELP.get(pred, ""))
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        true_label = st.session_state.get("_true_label")
        if true_label:
            if true_label == pred:
                st.success(f"Matches the true label: **{true_label}** ✅")
            else:
                st.warning(f"Loaded example's true label was **{true_label}**.")
    with right:
        st.subheader("Class probabilities")
        proba_df = (
            pd.DataFrame(
                {"Probability": [proba_by_class[c] for c in classes]},
                index=classes,
            )
            .sort_values("Probability", ascending=False)
        )
        st.bar_chart(proba_df)
        st.dataframe(
            proba_df.style.format("{:.3f}"),
            use_container_width=True,
        )

st.divider()
st.caption(
    "Model: tuned SVC inside a train-only imputation + standardisation pipeline. "
    "Built for the 23CSE301 ML capstone (bonus GUI). AI tools were used for scaffolding."
)

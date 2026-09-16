# Kepler KOI — Machine Learning Capstone (Classification Track)

**23CSE301 Machine Learning · B.Tech CSE III Year · Capstone Project**

End-to-end classification pipeline on the **NASA Kepler Objects of Interest (KOI)**
dataset: predicting a KOI's official disposition — **CANDIDATE**, **CONFIRMED**, or
**FALSE POSITIVE** — from tabular transit and stellar measurements.

## Problem statement
Given the numeric measurements of a Kepler Object of Interest, classify its
`koi_disposition` into one of three classes. This covers the **Classification
Part A** deliverable (five algorithms) of the capstone.

## Dataset
- **Source:** NASA Exoplanet Archive — Kepler cumulative KOI table
  (`https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv`)
- **Rows:** ~9,564 KOIs · **Target:** `koi_disposition` (3 classes)
- **Class balance:** FALSE POSITIVE 50.6% · CONFIRMED 28.7% · CANDIDATE 20.7%
  (moderately imbalanced → stratified split + weighted metrics)
- The raw file is committed at [`data/kepler_koi.csv`](data/kepler_koi.csv); the
  notebook also re-downloads it automatically if absent.

## Repository structure
```
.
├── README.md                     Project overview, results, how-to-run
├── requirements.txt              Python dependencies (pinned)
├── data/
│   └── kepler_koi.csv            Raw KOI dataset
├── notebooks/
│   ├── classification.ipynb      Full classification pipeline (run top-to-bottom)
│   └── figures/                  Saved confusion matrices, ROC curves, importances
├── models/
│   └── koi_classifier.joblib     Exported tuned-SVC pipeline + metadata (for the app)
└── app/
    ├── app.py                    Streamlit prediction GUI (bonus)
    ├── requirements.txt          App-only dependencies
    └── README.md                 App usage + public deployment guide
```

## Pipeline
Data audit → EDA (distributions, correlation heatmap, class-separation plots) →
cleaning → **feature engineering** (`transit_depth_per_hour = koi_depth / koi_duration`)
→ stratified 80/20 split → **train-only** median imputation + standardisation pipeline
→ five classifiers → consolidated comparison.

All preprocessing is fitted on the training set only (no data leakage);
`random_state=42` throughout for reproducibility.

## Results — Part A comparison (held-out test set)
| Model | Accuracy | Precision (w) | Recall (w) | F1 (w) | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| **Support Vector Classifier (tuned)** | **0.8223** | 0.8179 | 0.8223 | **0.8197** | **0.9442** |
| Decision Tree (tuned) | 0.8207 | 0.8180 | 0.8207 | 0.8190 | 0.9233 |
| Logistic Regression | 0.8197 | 0.8116 | 0.8197 | 0.8132 | 0.9368 |
| Decision Tree (baseline) | 0.7930 | 0.7950 | 0.7930 | 0.7938 | 0.8383 |
| K-Nearest Neighbors (tuned) | 0.7846 | 0.7849 | 0.7846 | 0.7822 | 0.9133 |
| Gaussian Naive Bayes | 0.6142 | 0.7241 | 0.6142 | 0.6186 | 0.8769 |

**Best model:** tuned **SVC** (RBF, `C=10`, `gamma='scale'`). `CANDIDATE` is the
hardest class across all models, as candidates share features with both other classes.

## Setup
```bash
pip install -r requirements.txt
```

## Run the notebook
```bash
jupyter notebook notebooks/classification.ipynb
```
Run all cells top-to-bottom. This regenerates the figures under `notebooks/figures/`
and re-exports `models/koi_classifier.joblib` (Section 20).

## Run the prediction GUI (bonus)
```bash
streamlit run app/app.py
```
Opens at http://localhost:8501. See [`app/README.md`](app/README.md) for public
deployment instructions (Streamlit Community Cloud / Hugging Face Spaces).

## Notes
- Metrics required by the rubric are all reported per algorithm: Accuracy, Precision,
  Recall, weighted F1, confusion matrix, and one-vs-rest ROC-AUC.
- AI tools were used for code scaffolding and formatting; all analysis,
  interpretation, and feature-engineering decisions are the team's own.

# Kepler KOI Classifier — Prediction GUI

Interactive web app for the classification track. It loads the tuned SVC pipeline
exported from `classification.ipynb` (Section 20) and predicts a Kepler Object of
Interest's disposition: **CANDIDATE**, **CONFIRMED**, or **FALSE POSITIVE**.

## Features
- Enter the most influential / interpretable measurements; all other model inputs
  default to their training-set median (so you only fill a handful of fields).
- "Load a real held-out example" button to see the model predict actual KOIs with
  their true labels.
- Predicted class, confidence, and a per-class probability chart.

## Run locally
From the project root:

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

Then open http://localhost:8501.

> The app needs `models/koi_classifier.joblib`. If it's missing, run the
> `classification.ipynb` notebook once (Section 20 — Model Export) to create it.

## Deploy publicly (Streamlit Community Cloud) — the +1 deployment bonus

1. Push this repository to GitHub, **including** `models/koi_classifier.joblib`
   (it must be committed — ~2.8 MB).
2. Go to https://share.streamlit.io → **New app** → sign in with GitHub.
3. Select this repo and branch, and set:
   - **Main file path:** `app/app.py`
4. Under **Advanced settings**, set the requirements file to `app/requirements.txt`
   (or copy those pins into a root `requirements.txt`).
5. **Deploy.** You'll get a public URL like
   `https://<your-app>.streamlit.app` — put that link in the project README for
   the deployment bonus.

### Alternative: Hugging Face Spaces
Create a new **Streamlit** Space, upload `app/app.py`, `app/requirements.txt`, and
`models/koi_classifier.joblib` (keeping the `models/` path), and it deploys
automatically.

## Version note
`app/requirements.txt` pins **scikit-learn==1.8.0** — this must match the version
used to export the model bundle, or unpickling will fail on the server.

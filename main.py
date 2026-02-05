from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI(
    title="Thyroid Disease Prediction API",
    version="3.0",
    description="RandomForest-based Thyroid Classifier (Normal / Hypo / Hyper)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load("thyroid_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    raise RuntimeError(f"Model files missing: {e}")


CLASS_MAP = {
    0: "Normal",
    1: "Hypothyroidism",
    2: "Hyperthyroidism"
}

CONFIDENCE_THRESHOLD = 0.60  # avoid borderline wrong calls


FEATURES = [
    "age",
    "sex",
    "on_thyroxine",
    "query_on_thyroxine",
    "on_antithyroid_medication",
    "sick",
    "pregnant",
    "thyroid_surgery",
    "I131_treatment",
    "query_hypothyroid",
    "query_hyperthyroid",
    "lithium",
    "goitre",
    "tumor",
    "hypopituitary",
    "psych",
    "TSH",
    "T3",
    "TT4",
    "T4U",
    "FTI"
]


@app.get("/")
def health():
    return {"status": "API running"}


@app.post("/predict")
def predict(data: dict):
    try:
        missing = [f for f in FEATURES if f not in data]
        if missing:
            raise HTTPException(400, f"Missing fields: {missing}")
        df = pd.DataFrame([data], columns=FEATURES)
        df = df.astype(float)

        if (df[["TSH", "T3", "TT4", "T4U", "FTI"]] < 0).any().any():
            raise HTTPException(400, "Lab values must be non-negative")

        df["T3"]  = df["T3"] / 10000
        df["TT4"] = df["TT4"] / 100
        df["FTI"] = df["FTI"] / 1000
        X_scaled = scaler.transform(df)

        pred = model.predict(X_scaled)
        proba = model.predict_proba(X_scaled)[0]

        raw_index = int(pred[0])
        confidence = float(proba[raw_index])

        diagnosis = CLASS_MAP[raw_index]

        if df["TSH"].iloc[0] < 0.1 and df["T3"].iloc[0] > 0.02:
            diagnosis = "Hyperthyroidism"
            confidence = 1.0

        if df["TSH"].iloc[0] > 5 and df["TT4"].iloc[0] < 6:
            diagnosis = "Hypothyroidism"
            confidence = 1.0

        if confidence < CONFIDENCE_THRESHOLD:
            diagnosis = "Borderline – Retest recommended"

       
        return {
            "diagnosis": diagnosis,
            "confidence_percent": round(confidence * 100, 2),
            "raw_class_index": raw_index,
            "probabilities": {
                "Normal": round(proba[0]*100, 2),
                "Hypothyroidism": round(proba[1]*100, 2),
                "Hyperthyroidism": round(proba[2]*100, 2)
            }
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

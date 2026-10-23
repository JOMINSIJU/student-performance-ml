import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "model.joblib"
METADATA_PATH = ARTIFACTS / "metadata.json"

app = FastAPI(title="Student Performance API", version="1.0.0")

model = None
metadata = None
load_error = None

EXPECTED_FEATURES = [
    "study_hours", "attendance", "previous_score"
]

def load_artifacts():
    global model, metadata, load_error

    try:
        if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
            raise FileNotFoundError("Required model artifacts are missing")

        new_metadata = json.loads(METADATA_PATH.read_text())
        new_model = joblib.load(MODEL_PATH)

        if new_metadata.get("features") != EXPECTED_FEATURES:
            raise ValueError("Metadata feature list is incompatible")

        if list(getattr(new_model, "feature_names_in_", [])) != EXPECTED_FEATURES:
            raise ValueError("Model feature names are incompatible")

        model = new_model
        metadata = new_metadata
        load_error = None

    except Exception as exc:
        model = None
        metadata = None
        load_error = str(exc)

@app.on_event("startup")
def startup():
    load_artifacts()

class StudentInput(BaseModel):
    study_hours: float = Field(ge=0, le=10)
    attendance: float = Field(ge=0, le=100)
    previous_score: float = Field(ge=0, le=100)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready"
        )
    return {"status": "ready"}

@app.get("/model-info")
def model_info():
    if metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Model information is unavailable"
        )
    return metadata

@app.post("/predict")
def predict(data: StudentInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    features = pd.DataFrame(
        [data.model_dump()],
        columns=EXPECTED_FEATURES
    )

    prediction = float(model.predict(features)[0])

    # Scores are bounded for this demo's defined target.
    prediction = max(0.0, min(100.0, prediction))

    return {
        "predicted_score": round(prediction, 2),
        "units": "marks out of 100",
        "model_version": metadata["model_version"]
    }
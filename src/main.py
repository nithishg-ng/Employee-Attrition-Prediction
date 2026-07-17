
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from src.preprocessing import transform
from src.schemas import EmployeeFeatures, HealthResponse, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
PREPROCESSING_PATH = BASE_DIR / "models" / "preprocessing_objects.pkl"

ml_objects = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists() or not PREPROCESSING_PATH.exists():
        raise RuntimeError(
            "Model artifacts not found. Run notebooks/data_preprocessing.ipynb "
            "and notebooks/model_training.ipynb first."
        )

    model_bundle = joblib.load(MODEL_PATH)
    ml_objects["model"] = model_bundle["model"]
    ml_objects["model_name"] = model_bundle["model_name"]
    ml_objects["n_features"] = model_bundle["n_features"]
    ml_objects["preprocessing"] = joblib.load(PREPROCESSING_PATH)

    yield

    ml_objects.clear()


app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Predicts whether an employee is likely to leave the company.",
    version="1.0.0",
    lifespan=lifespan,
)


def _risk_level(probability: float) -> str:
    if probability >= 0.6:
        return "High"
    if probability >= 0.3:
        return "Medium"
    return "Low"


@app.get("/", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        model_name=ml_objects["model_name"],
        n_features=ml_objects["n_features"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(employee: EmployeeFeatures):
    try:
        features = transform(employee, ml_objects["preprocessing"])
        model = ml_objects["model"]
        probability = float(model.predict_proba(features)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(
        attrition_prediction="Yes" if probability >= 0.5 else "No",
        attrition_probability=round(probability, 4),
        risk_level=_risk_level(probability),
        model_name=ml_objects["model_name"],
    )

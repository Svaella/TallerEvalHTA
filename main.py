from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat
from typing import Literal
import joblib
import pandas as pd
from pathlib import Path
import gdown

app = FastAPI(title="HTA Model API")

# --- Mapeo a enlaces de Google Drive (actualizar con tus IDs reales) ---
MODEL_URLS = {
    "random_forest": "https://drive.google.com/uc?id=1ErWTn9NDvOBEcUWS21ZziLlOlgAur6cQ",
    "decision_tree": "https://drive.google.com/uc?id=1dXTgiO-ARR9ISEdyI_8JJPTlEAakI1kb",
    "xgboost": "https://drive.google.com/uc?id=1el7pDgjRomfQbsbhBvhNlSiNSnbD67F7",
    "adaboost": "https://drive.google.com/uc?id=1AoQIiSi3yd1lQN5U1taUVJwzUL75mQtQ",
    "lightgbm": "https://drive.google.com/uc?id=1wsE7-mgJb5vqF1UE6ye7UHjUyezz_L8I",
    "catboost": "https://drive.google.com/uc?id=18kdwHRLfK-gc9_k9N0gTMQTR3y2O_WKa",
    "red_neuronal": "https://drive.google.com/uc?id=10FyPnFxo1tbE4k1wCqvCNKRAtkWSKQ03",
    "svm": "https://drive.google.com/uc?id=1z7Y5DWQmnp4ywfcOjZym-WbK0vO9MxnQ",
}

# --- Request desde Flutter ---
class PredictRequest(BaseModel):
    model: str
    Sexo: Literal["Hombre", "Mujer"]
    Edad: conint(gt=0)
    Peso: confloat(gt=0)
    Altura: confloat(gt=0)
    Fuma: Literal["Nunca", "Anteriormente", "Frecuentemente"]
    Alcohol: Literal["No consumo", "Bajo", "Moderado", "Alto"]
    Actividad: Literal["Bajo", "Moderado", "Alto"]
    Suenho: confloat(ge=0, le=24)
    Antecedentes: Literal["Sí", "No"]
    Estres: conint(ge=1, le=9)
    Sal: Literal["Bajo", "Moderado", "Alto"]

class PredictResponse(BaseModel):
    model: str
    probability: float
    risk: str

_model_cache: dict[str, any] = {}

def descargar_modelo(name: str) -> Path:
    url = MODEL_URLS[name]
    dest = Path(f"modelos/modelo_{name}_pipeline.pkl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        gdown.download(url, str(dest), quiet=False)
    return dest

def cargar_modelo(name: str):
    if name in _model_cache:
        return _model_cache[name]
    path = descargar_modelo(name)
    model = joblib.load(path)
    _model_cache[name] = model
    return model

def transformar_a_lista(req: PredictRequest) -> list:
    if req.model == "svm":
        sexo = 1 if req.Sexo == "Hombre" else 0
        fuma = {"Nunca": 0, "Anteriormente": 1, "Frecuentemente": 2}[req.Fuma]
        actividad = {"Alto": 0, "Moderado": 1, "Bajo": 2}[req.Actividad]
        alcohol = {"No consumo": 0, "Bajo": 1, "Moderado": 2, "Alto": 3}[req.Alcohol]
        sal = {"Bajo": 0, "Moderado": 1, "Alto": 2}[req.Sal]
        bmi = round(req.Peso / ((req.Altura / 100) ** 2), 2)
        antecedentes = 1 if req.Antecedentes == "Sí" else 0
        return [sexo, req.Edad, bmi, actividad, req.Suenho, fuma, antecedentes, req.Estres, sal, alcohol]

    sexo_map = {"Hombre": "Male", "Mujer": "Female"}
    fuma_map = {"Nunca": "Never", "Anteriormente": "Former", "Frecuentemente": "Current"}
    act_map = {"Bajo": "Low", "Moderado": "Moderate", "Alto": "High"}
    alc_map = {"No consumo": "None", "Bajo": "Low", "Moderado": "Moderate", "Alto": "High"}
    sal_map = {"Bajo": "Low", "Moderado": "Moderate", "Alto": "High"}
    bmi = round(req.Peso / ((req.Altura / 100) ** 2), 2)
    antecedentes = 1 if req.Antecedentes == "Sí" else 0
    return [
        sexo_map[req.Sexo], req.Edad, bmi, act_map[req.Actividad], req.Suenho,
        fuma_map[req.Fuma], antecedentes, req.Estres, alc_map[req.Alcohol], sal_map[req.Sal]
    ]

@app.post("/predict", response_model=PredictResponse)
def predecir(req: PredictRequest):
    try:
        modelo = cargar_modelo(req.model)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    columnas = [
        "Gender", "Age", "BMI", "Physical_Activity_Level", "Sleep_Duration",
        "Smoking_Status", "Family_History", "Stress_Level", "Alcohol_Level", "Salt_Level"
    ]
    datos = transformar_a_lista(req)
    df = pd.DataFrame([datos], columns=columnas)

    if req.model == "svm":
        df.rename(columns={"Alcohol_Level": "Alcohol_Intake", "Salt_Level": "Salt_Intake"}, inplace=True)
        df = df[[
            "Gender", "Age", "BMI", "Physical_Activity_Level", "Sleep_Duration",
            "Smoking_Status", "Family_History", "Stress_Level", "Salt_Intake", "Alcohol_Intake"
        ]]

    try:
        prob = float(modelo.predict_proba(df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inferencia error: {e}")

    pct = round(prob * 100, 2)
    riesgo = "Alto" if prob >= 0.75 else "Moderado" if prob >= 0.5 else "Bajo"

    return PredictResponse(model=req.model, probability=pct, risk=riesgo)
@app.get("/metricas")
def metricas():
    return {
        "random_forest": {"accuracy": 0.88, "precision": 0.75, "recall": 0.80, "f1_score": 0.82},
        "decision_tree": {"accuracy": 0.75, "precision": 0.74, "recall": 0.75, "f1_score": 0.75},
        "xgboost": {"accuracy": 0.78, "precision": 0.78, "recall": 0.78, "f1_score": 0.78},
        "adaboost": {"accuracy": 0.75, "precision": 0.75, "recall": 0.75, "f1_score": 0.75},
        "lightgbm": {"accuracy": 0.79, "precision": 0.80, "recall": 0.79, "f1_score": 0.79},
        "catboost": {"accuracy": 0.82, "precision": 0.86, "recall": 0.82, "f1_score": 0.83},
        "red_neuronal": {"accuracy": 0.50, "precision": 0.59, "recall": 0.50, "f1_score": 0.52},
        "svm": {"accuracy": 0.50, "precision": 0.59, "recall": 0.50, "f1_score": 0.52}
    }

@app.get("/auc_curvas")
def auc_curvas():
    return {
        "random_forest": {
            "fpr": [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0],
            "tpr": [0.0, 0.6, 0.8, 0.88, 0.93, 0.97, 0.99, 0.995, 0.997, 0.999, 1.0]
        },
        "decision_tree": {
            "fpr": [0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0],
            "tpr": [0.0, 0.45, 0.9, 0.95, 0.97, 0.98, 1.0]
        },
        "xgboost": {
            "fpr": [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0],
            "tpr": [0.0, 0.35, 0.6, 0.75, 0.85, 0.93, 0.97, 0.985, 0.99, 1.0]
        },
        "adaboost": {
            "fpr": [0.0, 0.18, 1.0],
            "tpr": [0.0, 0.67, 1.0]
        },
        "catboost": {
            "fpr": [0.0, 0.04, 0.08, 0.12, 0.2, 0.3, 1.0],
            "tpr": [0.0, 0.55, 0.75, 0.85, 0.95, 1.0, 1.0]
        },
        "lightgbm": {
            "fpr": [0.0, 0.04, 0.08, 0.12, 0.2, 0.3, 1.0],
            "tpr": [0.0, 0.55, 0.75, 0.9, 1.0, 1.0, 1.0]
        },
        "svm": {
            "fpr": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "tpr": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        },
        "red_neuronal": {
            "fpr": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "tpr": [0.0, 0.21, 0.42, 0.61, 0.81, 1.0]
        }
    }

#Run api: 
#uvicorn main:app --reload
#uvicorn main:app --reload --host 0.0.0.0 --port 8000




from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat
from typing import Literal
import joblib
import pandas as pd
from pathlib import Path
import gdown
import numpy as np
import warnings

warnings.filterwarnings("ignore")  # silenciar advertencias

app = FastAPI(title="HTA Model API")

MODEL_URLS = {
    "random_forest": "https://drive.google.com/uc?id=17kNgBN52bxwV5aKs78-lZm2WkSqAX1tz",
    "decision_tree": "https://drive.google.com/uc?id=1q1_qpmdbBuxDBYjmi0I1n5LyxYKt1oDT",
    "xgboost": "https://drive.google.com/uc?id=1qH8isdosuvqkBSH40XKIxS6UxTW566sC",
    "adaboost": "https://drive.google.com/uc?id=15V1oi9KORNr8J2M0KZlyg7fVEw8iG0BC",
    "lightgbm": "https://drive.google.com/uc?id=10VfCN_tcz2AKwS2pUyGjITUrj6yCFvQk",
    "catboost": "https://drive.google.com/uc?id=1nI6eLS0imUpEn26x-EKk0H4EpEPh1lqd",
    "red_neuronal": "https://drive.google.com/uc?id=CbOfVENf9457nFVd6L7S-v0A3yv_Wwwa",
    "svm_rbf": "https://drive.google.com/uc?id=1I1nj9mnP_RMZLgVm0CK1U8f1raxxMN1x"
}

class PredictRequest(BaseModel):
    model: str
    Age: conint(ge=1, le=13)
    Sex: Literal["Hombre", "Mujer"]
    Peso: confloat(ge=20, le=180)
    Altura: confloat(ge=100, le=220)
    MentHlth: conint(ge=0, le=30)
    Frutas: Literal["Sí", "No"]
    Verduras: Literal["Sí", "No"]
    Sal: Literal["Sí", "No"]
    Actividad: Literal["Sí", "No"]
    Fuma: Literal["Fumador Actual - Todos los días", "Fumador Actual - Algunos días", "Exfumador", "No Fumo"]
    Vapeo: Literal["Todos los días", "Algunos días", "Raramente", "Nunca he usado"]
    Alcohol30d: Literal["Sí", "No"]
    Diabetes: Literal["Sí", "No"]
    Colesterol: Literal["Sí", "No"]

class PredictResponse(BaseModel):
    model: str
    probability: float
    risk: str

_model_cache: dict[str, any] = {}

def descargar_modelo(name: str) -> Path:
    if name not in MODEL_URLS:
        raise ValueError(f"Modelo '{name}' no encontrado.")
    url = MODEL_URLS[name]
    dest = Path(f"modelos/modelo_{name}.pkl")
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

def transformar_a_vector(req: PredictRequest) -> list:
    sexo = 1 if req.Sex == "Hombre" else 0
    frutas = 1 if req.Frutas == "Sí" else 0
    verduras = 1 if req.Verduras == "Sí" else 0
    sal = 1 if req.Sal == "Sí" else 0
    actividad = 1 if req.Actividad == "Sí" else 0
    alcohol = 1 if req.Alcohol30d == "Sí" else 0
    colesterol = 1 if req.Colesterol == "Sí" else 0
    diabetes = 1 if req.Diabetes == "Sí" else 0
    fuma = {
        "Fumador Actual - Todos los días": 1,
        "Fumador Actual - Algunos días": 2,
        "Exfumador": 3,
        "No Fumo": 4
    }[req.Fuma]
    vapeo = {
        "Todos los días": 1,
        "Algunos días": 2,
        "Raramente": 3,
        "Nunca he usado": 4
    }[req.Vapeo]

    bmi = round(req.Peso / ((req.Altura / 100) ** 2), 2)

    return [
        req.Age, sexo, bmi, req.MentHlth, frutas, verduras, sal,
        actividad, fuma, vapeo, alcohol, diabetes, colesterol
    ]

@app.post("/predict", response_model=PredictResponse)
def predecir(req: PredictRequest):
    try:
        modelo = cargar_modelo(req.model)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    columnas = [
        "Age", "Sex", "BMI", "MentHlth", "Fruits", "Veggies",
        "Salt", "PhysActivity", "Smoker", "Vaper",
        "Alcohol", "Diabetes", "HighChol"
    ]

    datos = transformar_a_vector(req)
    df = pd.DataFrame([datos], columns=columnas)

    print("📤 Datos enviados al modelo:")
    print(df)

    try:
        if req.model in ["adaboost"]:
            prob = float(modelo.predict_proba(df.to_numpy())[0][1])
        else:
            prob = float(modelo.predict_proba(df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")

    pct = round(prob * 100, 2)
    riesgo = "Alto" if prob >= 0.65 else "Moderado" if prob >= 0.35 else "Bajo"
    return PredictResponse(model=req.model, probability=pct, risk=riesgo)


@app.get("/metricas")
def metricas():
    return {
        "random_forest": {
            "accuracy": 0.970651,
            "precision": 0.973545,
            "recall": 0.961064,
            "f1_score": 0.967264
        },
        "decision_tree": {
            "accuracy": 0.687661,
            "precision": 0.643744,
            "recall": 0.688984,
            "f1_score": 0.665596
        },
        "xgboost": {
            "accuracy": 0.853470,
            "precision": 0.845817,
            "recall": 0.825376,
            "f1_score": 0.835656
        },
        "adaboost": {
            "accuracy": 0.731977,
            "precision": 0.722924,
            "recall": 0.741366,
            "f1_score": 0.730229
        },
        "lightgbm": {
            "accuracy": 0.732946,
            "precision": 0.725869,
            "recall": 0.737834,
            "f1_score": 0.731802
        },
        "catboost": {
            "accuracy": 0.732946,
            "precision": 0.726043,
            "recall": 0.737441,
            "f1_score": 0.731698
        },
        "red_neuronal": {
            "accuracy": 0.71,
            "precision": 0.71,
            "recall": 0.71,
            "f1_score": 0.71
        },
        "svm_rbf": {
            "accuracy": 0.699229,
            "precision": 0.635312,
            "recall": 0.782526,
            "f1_score": 0.701277
        }
    }

@app.get("/auc_curvas")
def auc_curvas():
    return {
        "random_forest": {
            "fpr": [0.0, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0],
            "tpr": [0.0, 0.92, 0.96, 0.975, 0.985, 0.99, 0.995, 0.997, 0.998, 0.999, 1.0],
        },
        "decision_tree": {
            "fpr": [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            "tpr": [0.0, 0.2, 0.4, 0.65, 0.82, 0.92, 1.0],
        },
        "xgboost": {
            "fpr": [0.0, 0.03, 0.08, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0],
            "tpr": [0.0, 0.5, 0.75, 0.88, 0.94, 0.97, 0.985, 0.995, 1.0],
        },
        "adaboost": {
            "fpr": [0.0, 0.05, 0.15, 0.35, 0.5, 0.75, 1.0],
            "tpr": [0.0, 0.35, 0.6, 0.78, 0.89, 0.96, 1.0],
        },
        "catboost": {
            "fpr": [0.0, 0.06, 0.15, 0.35, 0.5, 0.75, 1.0],
            "tpr": [0.0, 0.45, 0.7, 0.85, 0.92, 0.97, 1.0],
        },
        "lightgbm": {
            "fpr": [0.0, 0.06, 0.15, 0.35, 0.5, 0.75, 1.0],
            "tpr": [0.0, 0.45, 0.72, 0.86, 0.93, 0.975, 1.0],
        },
        "svm_rbf": {
            "fpr": [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            "tpr": [0.0, 0.25, 0.45, 0.65, 0.82, 0.93, 1.0],
        },
        "red_neuronal": {
            "fpr": [0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0],
            "tpr": [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0],
        }
    }

@app.get("/loss_rate")
def loss_rate():
    return {
        "random_forest": {"train": 0.1616, "test": 0.1769},
        "decision_tree": {"train": 0.5501, "test": 0.5753},
        "xgboost": {"train": 0.3058, "test": 0.3418},
        "adaboost": {"train": 0.6163, "test": 0.6198},
        "catboost": {"train": 0.4614, "test": 0.5131},
        "lightgbm": {"train": 0.4463, "test": 0.5131},
        "svm_rbf": {"train": 0.7489, "test": 0.7825},
        "red_neuronal": {"train": 0.5327, "test": 0.5586}
    }

@app.get("/tiempos_inferencia")
def tiempos_inferencia():
    return {
        "random_forest": 0.030724,
        "decision_tree": 0.000170,
        "xgboost": 0.013525,
        "adaboost": 0.028975,
        "catboost": 0.001440,
        "lightgbm": 0.038512,
        "svm_rbf": 0.130879,
        "red_neuronal": 0.001964
    }

#Run api: 
#uvicorn main:app --reload
#uvicorn main:app --reload --host 0.0.0.0 --port 8000
import pytest
#import warnings
from fastapi.testclient import TestClient
from main import app

# Ignorar warning de LightGBM por nombres de columnas
#warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

client = TestClient(app)

# Datos base para pruebas válidas
datos_base = {
    "Sexo": "Hombre",
    "Edad": 40,
    "Peso": 80.0,
    "Altura": 170.0,
    "Fuma": "Nunca",
    "Alcohol": "Moderado",
    "Actividad": "Moderado",
    "Suenho": 7.5,
    "Antecedentes": "Sí",
    "Estres": 5,
    "Sal": "Moderado"
}

modelos = [
    "random_forest",
    "decision_tree",
    "xgboost",
    "adaboost",
    "lightgbm",
    "catboost",
    "red_neuronal",
    "svm"
]

# ----------- Pruebas de predicción por modelo válido -----------

@pytest.mark.parametrize("modelo", modelos)
def test_predict_por_modelo(modelo):
    datos = datos_base.copy()
    datos["model"] = modelo
    response = client.post("/predict", json=datos)
    assert response.status_code == 200
    body = response.json()
    assert "probability" in body
    assert "risk" in body
    assert body["model"] == modelo

# ----------- Endpoints generales -----------

def test_metricas():
    response = client.get("/metricas")
    assert response.status_code == 200
    metricas = response.json()
    for modelo in modelos:
        assert modelo in metricas
        for metrica in ["accuracy", "precision", "recall", "f1_score"]:
            assert metrica in metricas[modelo]

def test_auc_curvas():
    response = client.get("/auc_curvas")
    assert response.status_code == 200
    curvas = response.json()
    for modelo in modelos:
        assert modelo in curvas
        assert "fpr" in curvas[modelo]
        assert "tpr" in curvas[modelo]
        assert isinstance(curvas[modelo]["fpr"], list)
        assert isinstance(curvas[modelo]["tpr"], list)

def test_loss_rate():
    response = client.get("/loss_rate")
    assert response.status_code == 200
    losses = response.json()
    for modelo in modelos:
        assert modelo in losses
        assert "train" in losses[modelo]
        assert "test" in losses[modelo]

def test_tiempos_inferencia():
    response = client.get("/tiempos_inferencia")
    assert response.status_code == 200
    tiempos = response.json()
    for modelo in modelos:
        assert modelo in tiempos
        assert isinstance(tiempos[modelo], float)

# ----------- Pruebas negativas y validación -----------

def test_modelo_inexistente():
    datos = datos_base.copy()
    datos["model"] = "modelo_x"
    response = client.post("/predict", json=datos)
    assert response.status_code == 404
    assert "detail" in response.json()

def test_edad_invalida():
    datos = datos_base.copy()
    datos["Edad"] = -10
    datos["model"] = "svm"
    response = client.post("/predict", json=datos)
    assert response.status_code == 422

def test_texto_fuera_literal():
    datos = datos_base.copy()
    datos["Fuma"] = "Mucho"
    datos["model"] = "svm"
    response = client.post("/predict", json=datos)
    assert response.status_code == 422

def test_suenho_fuera_rango():
    datos = datos_base.copy()
    datos["Suenho"] = 30.0
    datos["model"] = "svm"
    response = client.post("/predict", json=datos)
    assert response.status_code == 422

def test_estres_fuera_rango():
    datos = datos_base.copy()
    datos["Estres"] = 15
    datos["model"] = "svm"
    response = client.post("/predict", json=datos)
    assert response.status_code == 422

# ----------- Pruebas de valores extremos válidos -----------

@pytest.mark.parametrize("campo,valor", [
    ("Edad", 18),
    ("Edad", 100),
    ("Peso", 20.0),
    ("Peso", 180.0),
    ("Altura", 100.0),
    ("Altura", 220.0),
    ("Suenho", 0.0),
    ("Suenho", 24.0),
    ("Estres", 1),
    ("Estres", 9),
])
def test_valores_extremos_validos(campo, valor):
    datos = datos_base.copy()
    datos[campo] = valor
    datos["model"] = "svm"
    response = client.post("/predict", json=datos)
    assert response.status_code == 200
    body = response.json()
    assert "probability" in body
    assert "risk" in body

# ----------- Pruebas de valores extremos inválidos -----------

@pytest.mark.parametrize("campo,valor", [
    ("Edad", 17),
    ("Edad", 101),
    ("Peso", 19.9),
    ("Peso", 180.1),
    ("Altura", 99.9),
    ("Altura", 220.1),
    ("Suenho", -1.0),
    ("Suenho", 25.0),
    ("Estres", 0),
    ("Estres", 10),
])
def test_valores_extremos_invalidos(campo, valor):
    datos = datos_base.copy()
    datos[campo] = valor
    datos["model"] = "svm"
    response = client.post("/predict", json=datos)
    assert response.status_code == 422

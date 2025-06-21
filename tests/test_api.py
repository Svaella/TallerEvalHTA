import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# =================== DATOS BASE ===================

datos_validos = {
    "model": "random_forest",
    "Age": 10,
    "Sex": "Hombre",
    "Peso": 72.0,
    "Altura": 178.0,
    "MentHlth": 6,
    "Frutas": "No",
    "Verduras": "Sí",
    "Sal": "No",
    "Actividad": "Sí",
    "Fuma": "No Fumo",
    "Vapeo": "Algunos días",
    "Alcohol30d": "No",
    "Diabetes": "No",
    "Colesterol": "No"
}

modelos = [
    "random_forest", "decision_tree", "xgboost",
    "adaboost", "lightgbm", "catboost",
    "red_neuronal", "svm_rbf"
]

# ========== 1. PRUEBAS DEL ENDPOINT /predict ==========

@pytest.mark.parametrize("modelo", modelos)
def test_predict_por_modelo(modelo):
    datos = datos_validos.copy()
    datos["model"] = modelo
    response = client.post("/predict", json=datos)
    assert response.status_code == 200
    body = response.json()
    assert "probability" in body
    assert "risk" in body
    assert "model" in body
    assert body["model"] == modelo

def test_modelo_inexistente():
    datos = datos_validos.copy()
    datos["model"] = "modelo_fake"
    response = client.post("/predict", json=datos)
    assert response.status_code == 404

def test_campos_incompletos():
    datos = datos_validos.copy()
    del datos["Altura"]
    response = client.post("/predict", json=datos)
    assert response.status_code == 422

@pytest.mark.parametrize("campo, valor", [
    ("Age", 0), ("Age", 14),
    ("Peso", 10.0), ("Peso", 200.0),
    ("Altura", 90.0), ("Altura", 230.0),
    ("MentHlth", -1), ("MentHlth", 31),
])
def test_valores_fuera_de_rango(campo, valor):
    datos = datos_validos.copy()
    datos[campo] = valor
    response = client.post("/predict", json=datos)
    assert response.status_code == 422

@pytest.mark.parametrize("campo, valor", [
    ("Age", 1), ("Age", 13),
    ("Peso", 20.0), ("Peso", 180.0),
    ("Altura", 100.0), ("Altura", 220.0),
    ("MentHlth", 0), ("MentHlth", 30),
])
def test_valores_extremos_validos(campo, valor):
    datos = datos_validos.copy()
    datos[campo] = valor
    response = client.post("/predict", json=datos)
    assert response.status_code == 200
    body = response.json()
    assert "probability" in body
    assert "risk" in body

def test_opcion_literal_invalida():
    datos = datos_validos.copy()
    datos["Fuma"] = "Mucho"
    response = client.post("/predict", json=datos)
    assert response.status_code == 422

# ========== 2. PRUEBAS DE ENDPOINTS GET ==========

def test_metricas():
    response = client.get("/metricas")
    assert response.status_code == 200
    data = response.json()
    for modelo in modelos:
        assert modelo in data
        for metrica in ["accuracy", "precision", "recall", "f1_score"]:
            assert metrica in data[modelo]

def test_auc_curvas():
    response = client.get("/auc_curvas")
    assert response.status_code == 200
    data = response.json()
    for modelo in modelos:
        assert modelo in data
        assert "fpr" in data[modelo]
        assert "tpr" in data[modelo]
        assert isinstance(data[modelo]["fpr"], list)
        assert isinstance(data[modelo]["tpr"], list)

def test_loss_rate():
    response = client.get("/loss_rate")
    assert response.status_code == 200
    data = response.json()
    for modelo in modelos:
        assert "train" in data[modelo]
        assert "test" in data[modelo]

def test_tiempos_inferencia():
    response = client.get("/tiempos_inferencia")
    assert response.status_code == 200
    data = response.json()
    for modelo in modelos:
        assert isinstance(data[modelo], float)

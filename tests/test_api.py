import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Datos base válidos
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
    "random_forest", "decision_tree", "xgboost",
    "adaboost", "lightgbm", "catboost",
    "red_neuronal", "svm"
]

# ----------- Pruebas de predicción por modelo válido -----------

@pytest.mark.parametrize("modelo", modelos)
def test_predict_por_modelo(modelo):
    """Verifica que cada modelo válido retorne probabilidad, nivel de riesgo y nombre del modelo."""
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
    """Verifica que el endpoint /metricas devuelva métricas por modelo: accuracy, precision, recall, f1_score."""
    response = client.get("/metricas")
    assert response.status_code == 200
    metricas = response.json()
    for modelo in modelos:
        assert modelo in metricas
        for metrica in ["accuracy", "precision", "recall", "f1_score"]:
            assert metrica in metricas[modelo]

def test_auc_curvas():
    """Verifica que /auc_curvas retorne correctamente fpr y tpr por modelo."""
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
    """Verifica que /loss_rate retorne tasas de pérdida para entrenamiento y prueba."""
    response = client.get("/loss_rate")
    assert response.status_code == 200
    losses = response.json()
    for modelo in modelos:
        assert modelo in losses
        assert "train" in losses[modelo]
        assert "test" in losses[modelo]

def test_tiempos_inferencia():
    """Verifica que /tiempos_inferencia devuelva los tiempos (en segundos) por modelo."""
    response = client.get("/tiempos_inferencia")
    assert response.status_code == 200
    tiempos = response.json()
    for modelo in modelos:
        assert modelo in tiempos
        assert isinstance(tiempos[modelo], float)

# ----------- Pruebas negativas y validación -----------

def test_modelo_inexistente():
    """Debe retornar 404 si se indica un modelo que no existe."""
    datos = datos_base.copy()
    datos["model"] = "modelo_x"
    response = client.post("/predict", json=datos)
    assert response.status_code == 404
    assert "detail" in response.json()

def test_edad_invalida():
    """Debe retornar 422 si la edad está fuera del rango permitido (e.g. -10)."""
    datos = datos_base.copy()
    datos["Edad"] = -10
    datos["model"] = "svm"
    response = client.post("/predict", json=datos)
    assert response.status_code == 422

def test_texto_fuera_literal():
    """Debe retornar 422 si se envía una opción no válida en campos tipo selección (e.g. Fuma="Mucho")."""
    datos = datos_base.copy()
    datos["Fuma"] = "Mucho"
    datos["model"] = "svm"
    response = client.post("/predict", json=datos)
    assert response.status_code == 422

def test_suenho_fuera_rango():
    """Debe retornar 422 si las horas de sueño están fuera del rango 4–12 horas."""
    datos = datos_base.copy()
    datos["Suenho"] = 2.0
    datos["model"] = "svm"
    response = client.post("/predict", json=datos)
    assert response.status_code == 422

def test_estres_fuera_rango():
    """Debe retornar 422 si el nivel de estrés está fuera del rango 1–9 (e.g. 15)."""
    datos = datos_base.copy()
    datos["Estres"] = 15
    datos["model"] = "svm"
    response = client.post("/predict", json=datos)
    assert response.status_code == 422

# ----------- Pruebas de valores extremos válidos -----------

@pytest.mark.parametrize("campo,valor", [
    ("Edad", 18), ("Edad", 100),
    ("Peso", 20.0), ("Peso", 180.0),
    ("Altura", 100.0), ("Altura", 220.0),
    ("Suenho", 4.0), ("Suenho", 12.0),
    ("Estres", 1), ("Estres", 9),
])
def test_valores_extremos_validos(campo, valor):
    """Debe aceptar valores en el límite del rango permitido sin errores."""
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
    ("Edad", 17), ("Edad", 101),
    ("Peso", 19.9), ("Peso", 180.1),
    ("Altura", 99.9), ("Altura", 220.1),
    ("Suenho", 3.9), ("Suenho", 12.1),
    ("Estres", 0), ("Estres", 10),
])
def test_valores_extremos_invalidos(campo, valor):
    """Debe rechazar valores fuera del límite permitido retornando 422."""
    datos = datos_base.copy()
    datos[campo] = valor
    datos["model"] = "svm"
    response = client.post("/predict", json=datos)
    assert response.status_code == 422


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ml"))

import pytest
from fastapi.testclient import TestClient

from app import app, ARTIFACTS

client = TestClient(app)

BANANE = {"poids_g": 120, "diametre_mm": 35, "longueur_mm": 180,
          "sucre_brix": 20, "acidite_ph": 5.0, "fermete": 2.5}
ANANAS = {"poids_g": 1500, "diametre_mm": 120, "longueur_mm": 250,
          "sucre_brix": 13, "acidite_ph": 3.6, "fermete": 5.5}


@pytest.fixture(scope="session", autouse=True)
def trained_models():
    """Entraîne une fois pour toute la session de tests."""
    from train import train
    if not (ARTIFACTS / "models.joblib").exists():
        train()


def test_home_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "Fruit MLOps" in r.text


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["models_ready"] is True


def test_metrics_reports_three_models():
    m = client.get("/api/metrics").json()
    assert set(m["classification"]["resultats"]) == {
        "regression_logistique", "random_forest", "reseau_neurones_mlp",
    }
    # tous les modèles doivent être bons sur ce dataset bien séparé
    for r in m["classification"]["resultats"].values():
        assert r["accuracy"] > 0.9
    assert m["regression_prix"]["r2"] > 0.6


def test_predict_recognizes_banana_and_pineapple():
    r = client.post("/api/predict", json=BANANE)
    assert r.status_code == 200
    d = r.json()
    assert d["fruit"] == "Banane"
    assert d["confiance"] > 0.5
    assert d["prix_estime_ar_kg"] > 0
    assert abs(sum(d["probabilites"].values()) - 1) < 0.01

    assert client.post("/api/predict", json=ANANAS).json()["fruit"] == "Ananas"


def test_predict_validates_input():
    r = client.post("/api/predict", json={**BANANE, "poids_g": -5})
    assert r.status_code == 422


def test_batch_prediction_returns_csv():
    csv = ("poids_g,diametre_mm,longueur_mm,sucre_brix,acidite_ph,fermete\n"
           "120,35,180,20,5,2.5\n1500,120,250,13,3.6,5.5\n")
    r = client.post("/api/predict-batch", files={"file": ("fruits.csv", csv, "text/csv")})
    assert r.status_code == 200
    assert "fruit_predit" in r.text.splitlines()[0]
    assert "Banane" in r.text and "Ananas" in r.text


def test_batch_rejects_missing_columns():
    r = client.post("/api/predict-batch", files={"file": ("bad.csv", "a,b\n1,2\n", "text/csv")})
    assert r.status_code == 400
    assert "Colonnes manquantes" in r.json()["detail"]


def test_charts_served():
    for name in ("confusion_matrix", "feature_importance", "learning_curve"):
        r = client.get(f"/api/charts/{name}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
    assert client.get("/api/charts/inconnu").status_code == 404


def test_history_and_monitoring():
    for _ in range(6):
        client.post("/api/predict", json=BANANE)
    hist = client.get("/api/history").json()
    assert len(hist) >= 6
    assert hist[0]["fruit"] == "Banane"

    mon = client.get("/api/monitoring").json()
    assert mon["statut"] in ("ok", "alerte dérive")
    assert len(mon["drift"]) == 6  # une ligne par caractéristique

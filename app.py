"""Fruit MLOps v2 — classification de fruits + prix, interface web complète.

Fonctionnalités : prédiction interactive, prédiction en lot (CSV),
comparaison de modèles (ML vs réseau de neurones), graphiques,
historique, détection de dérive des données, ré-entraînement en un clic.
"""
import io
import json
import sys
import threading
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent / "ml"))

ARTIFACTS = Path(__file__).parent / "ml" / "artifacts"
HISTORY_PATH = ARTIFACTS / "history.json"

app = FastAPI(title="Fruit MLOps v2", description="ML + Deep Learning — classification de fruits et prédiction de prix")
templates = Jinja2Templates(directory="templates")

_bundle = None
_lock = threading.Lock()


def get_models():
    global _bundle
    if _bundle is None:
        if not (ARTIFACTS / "models.joblib").exists():
            from train import train
            train()
        _bundle = joblib.load(ARTIFACTS / "models.joblib")
    return _bundle


def get_metrics() -> dict:
    path = ARTIFACTS / "metrics.json"
    if not path.exists():
        raise HTTPException(503, "Modèles pas encore entraînés — POST /api/retrain")
    return json.loads(path.read_text(encoding="utf-8"))


def load_history() -> list:
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    return []


def save_history(entries: list) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(entries[-500:], ensure_ascii=False), encoding="utf-8")


class FruitIn(BaseModel):
    poids_g: float = Field(gt=0, le=10000)
    diametre_mm: float = Field(gt=0, le=1000)
    longueur_mm: float = Field(gt=0, le=1000)
    sucre_brix: float = Field(ge=0, le=40)
    acidite_ph: float = Field(ge=1, le=9)
    fermete: float = Field(ge=0, le=10)


def predict_one(fruit: FruitIn) -> dict:
    models = get_models()
    metrics = get_metrics()
    X = pd.DataFrame([fruit.model_dump()])[metrics["features"]]

    clf = models["classifier"]
    proba = clf.predict_proba(X)[0]
    classes = list(clf.classes_)
    fruit_pred = classes[int(proba.argmax())]
    prix = float(models["price_model"].predict(X)[0])

    return {
        "fruit": fruit_pred,
        "confiance": round(float(proba.max()), 3),
        "probabilites": {c: round(float(p), 3) for c, p in zip(classes, proba)},
        "prix_estime_ar_kg": round(prix),
    }


# ---------- interface ----------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------- API ----------

@app.get("/api/health")
def health():
    return {"status": "ok", "app": "Fruit MLOps v2", "models_ready": (ARTIFACTS / "models.joblib").exists()}


@app.get("/api/metrics")
def metrics():
    return get_metrics()


@app.post("/api/predict")
def predict(fruit: FruitIn):
    result = predict_one(fruit)
    history = load_history()
    history.append({
        "horodatage": datetime.now().isoformat(timespec="seconds"),
        "entree": fruit.model_dump(),
        "fruit": result["fruit"],
        "confiance": result["confiance"],
        "prix_estime_ar_kg": result["prix_estime_ar_kg"],
    })
    save_history(history)
    return result


@app.post("/api/predict-batch")
async def predict_batch(file: UploadFile):
    """Prédit un fichier CSV entier (colonnes = les 6 caractéristiques)."""
    metrics_data = get_metrics()
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "CSV illisible")
    missing = [c for c in metrics_data["features"] if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Colonnes manquantes : {', '.join(missing)}")
    if len(df) > 10_000:
        raise HTTPException(400, "Maximum 10 000 lignes")

    models = get_models()
    X = df[metrics_data["features"]]
    df_out = df.copy()
    df_out["fruit_predit"] = models["classifier"].predict(X)
    df_out["confiance"] = models["classifier"].predict_proba(X).max(axis=1).round(3)
    df_out["prix_estime_ar_kg"] = models["price_model"].predict(X).round(0)

    buffer = io.StringIO()
    df_out.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="predictions.csv"'},
    )


@app.get("/api/charts/{name}")
def chart(name: str):
    allowed = {"confusion_matrix", "feature_importance", "learning_curve"}
    if name not in allowed:
        raise HTTPException(404)
    path = ARTIFACTS / "charts" / f"{name}.png"
    if not path.exists():
        raise HTTPException(404, "Graphique pas encore généré — entraîner d'abord")
    return FileResponse(path, media_type="image/png")


@app.get("/api/history")
def history():
    return list(reversed(load_history()))[:50]


@app.get("/api/monitoring")
def monitoring():
    """Détection de dérive : compare les entrées récentes aux données d'entraînement.

    Un z-score moyen > 2 sur une caractéristique signale une dérive
    (les données reçues ne ressemblent plus à celles de l'entraînement).
    """
    m = get_metrics()
    entries = load_history()
    if len(entries) < 5:
        return {"statut": "pas assez de données", "n_predictions": len(entries), "drift": []}

    recent = pd.DataFrame([e["entree"] for e in entries[-100:]])
    drift = []
    for feature in m["features"]:
        mean_train = m["moyennes_entrainement"][feature]
        std_train = m["ecarts_types_entrainement"][feature] or 1
        z = abs(recent[feature].mean() - mean_train) / std_train
        drift.append({
            "feature": feature,
            "moyenne_entrainement": mean_train,
            "moyenne_recente": round(float(recent[feature].mean()), 2),
            "z_score": round(float(z), 2),
            "alerte": bool(z > 2),
        })
    return {
        "statut": "alerte dérive" if any(d["alerte"] for d in drift) else "ok",
        "n_predictions": len(entries),
        "drift": drift,
    }


@app.post("/api/retrain")
def retrain():
    """Ré-entraîne tous les modèles (données régénérées) et recharge en mémoire."""
    global _bundle
    with _lock:
        from train import train
        result = train()
        _bundle = None
    return {"statut": "ré-entraîné", "classification": result["classification"]}

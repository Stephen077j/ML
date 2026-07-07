"""Entraînement comparatif ML vs Deep Learning.

Classification du fruit :
- Régression logistique (baseline ML)
- Random Forest (ML ensembliste)
- Réseau de neurones MLP à 2 couches cachées (deep learning)

Régression du prix au kilo :
- Gradient Boosting

Produit : modèles + métriques (metrics.json) + graphiques PNG
(matrice de confusion, importance des features, courbe d'apprentissage).
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
    f1_score, mean_absolute_error, r2_score,
)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import DATA_DIR, FEATURES, generate

ARTIFACTS = DATA_DIR
CHARTS = ARTIFACTS / "charts"

CLASSIFIERS = {
    "regression_logistique": LogisticRegression(max_iter=2000),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "reseau_neurones_mlp": MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", max_iter=800,
        random_state=42,
    ),
}


def train() -> dict:
    ARTIFACTS.mkdir(exist_ok=True)
    CHARTS.mkdir(exist_ok=True)

    df = generate()
    df.to_csv(ARTIFACTS / "fruits.csv", index=False)

    X, y_class, y_price = df[FEATURES], df["fruit"], df["prix_ar_kg"]
    X_tr, X_te, yc_tr, yc_te, yp_tr, yp_te = train_test_split(
        X, y_class, y_price, test_size=0.25, random_state=42, stratify=y_class,
    )

    # ---- classification : comparaison des 3 modèles ----
    results, best_name, best_model, best_f1 = {}, None, None, -1.0
    for name, clf in CLASSIFIERS.items():
        model = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        model.fit(X_tr, yc_tr)
        pred = model.predict(X_te)
        acc = accuracy_score(yc_te, pred)
        f1 = f1_score(yc_te, pred, average="macro")
        results[name] = {"accuracy": round(acc, 4), "f1_macro": round(f1, 4)}
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1

    # ---- régression du prix ----
    reg = GradientBoostingRegressor(random_state=42)
    reg.fit(X_tr, yp_tr)
    price_pred = reg.predict(X_te)
    price_metrics = {
        "mae_ar": round(float(mean_absolute_error(yp_te, price_pred))),
        "r2": round(float(r2_score(yp_te, price_pred)), 3),
    }

    # ---- graphiques ----
    classes = sorted(y_class.unique())
    cm = confusion_matrix(yc_te, best_model.predict(X_te), labels=classes)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matrice de confusion — {best_name}")
    fig.tight_layout()
    fig.savefig(CHARTS / "confusion_matrix.png", dpi=110)
    plt.close(fig)

    rf = Pipeline([("scaler", StandardScaler()),
                   ("clf", RandomForestClassifier(n_estimators=200, random_state=42))])
    rf.fit(X_tr, yc_tr)
    importances = rf.named_steps["clf"].feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(FEATURES)[order], importances[order], color="#1d4ed8")
    ax.set_title("Importance des caractéristiques (Random Forest)")
    fig.tight_layout()
    fig.savefig(CHARTS / "feature_importance.png", dpi=110)
    plt.close(fig)

    sizes, train_scores, val_scores = learning_curve(
        best_model, X, y_class, cv=4, train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, train_scores.mean(axis=1), "o-", label="entraînement")
    ax.plot(sizes, val_scores.mean(axis=1), "s-", label="validation")
    ax.set_xlabel("taille du jeu d'entraînement")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Courbe d'apprentissage — {best_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "learning_curve.png", dpi=110)
    plt.close(fig)

    # ---- sauvegarde ----
    metrics = {
        "classification": {"modele_retenu": best_name, "resultats": results},
        "regression_prix": price_metrics,
        "n_donnees": len(df),
        "features": FEATURES,
        "classes": classes,
        "moyennes_entrainement": X_tr.mean().round(2).to_dict(),
        "ecarts_types_entrainement": X_tr.std().round(2).to_dict(),
    }
    joblib.dump({"classifier": best_model, "price_model": reg}, ARTIFACTS / "models.joblib")
    (ARTIFACTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return metrics


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    m = train()
    print(json.dumps(m["classification"], indent=2, ensure_ascii=False))
    print("Prix :", m["regression_prix"])

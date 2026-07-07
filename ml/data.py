"""Jeu de données fruits (synthétique mais réaliste).

4 fruits de Madagascar avec des caractéristiques mesurables distinctes.
Chaque fruit a aussi un prix au kilo qui dépend de ses caractéristiques
(un fruit plus sucré et plus gros se vend plus cher).
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "artifacts"
FEATURES = ["poids_g", "diametre_mm", "longueur_mm", "sucre_brix", "acidite_ph", "fermete"]

# profils moyens : (poids, diamètre, longueur, sucre, acidité, fermeté, prix base Ar/kg)
PROFILES = {
    "Banane":  (120, 35, 180, 20, 5.0, 2.5, 4000),
    "Mangue":  (350, 90, 110, 15, 4.0, 4.0, 6000),
    "Litchi":  (20, 32, 35, 17, 4.5, 3.0, 9000),
    "Ananas":  (1500, 120, 250, 13, 3.6, 5.5, 5000),
}


def generate(n_per_class: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for fruit, (poids, diam, longu, sucre, ph, fermete, prix_base) in PROFILES.items():
        for _ in range(n_per_class):
            p = max(5, rng.normal(poids, poids * 0.30))
            d = max(10, rng.normal(diam, diam * 0.22))
            lo = max(15, rng.normal(longu, longu * 0.22))
            s = max(5, rng.normal(sucre, 3.5))
            a = np.clip(rng.normal(ph, 0.5), 2.5, 7.0)
            f = np.clip(rng.normal(fermete, 1.2), 0.5, 8.0)
            # le prix monte avec le sucre et le calibre, baisse si trop mûr (fermeté basse)
            prix = prix_base * (1 + 0.03 * (s - sucre) + 0.10 * (p / poids - 1) + 0.05 * (f - fermete))
            prix = max(500, prix + rng.normal(0, prix_base * 0.05))
            rows.append({
                "fruit": fruit, "poids_g": round(p, 1), "diametre_mm": round(d, 1),
                "longueur_mm": round(lo, 1), "sucre_brix": round(s, 1),
                "acidite_ph": round(a, 2), "fermete": round(f, 2),
                "prix_ar_kg": round(prix),
            })
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(DATA_DIR / "fruits.csv", index=False)
    print(f"{len(df)} fruits générés -> {DATA_DIR / 'fruits.csv'}")
    print(df.groupby('fruit')[['poids_g', 'sucre_brix', 'prix_ar_kg']].mean().round(1))

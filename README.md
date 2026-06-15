# Fruit MLOps

API:
POST /predict

Entrée:
{
  "feature_x": 43.7
}

Sortie:
{
  "prediction": 52.31
}

Docker:
docker build -t fruit-api .

Déploiement:
GitHub + Docker + Render

Monitoring:
pip install evidently

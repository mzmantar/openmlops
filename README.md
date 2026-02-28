# OpenMLOps

Pipeline MLOps de bout en bout pour entraîner, évaluer, versionner, monitorer et re-déclencher un entraînement d’un modèle CNN sur CIFAR-10, en s’appuyant sur ZenML, MLflow, DVC et Evidently.

## 🎥 Démonstration vidéo

- Lien de la démo: [Voir la vidéo](https://drive.google.com/file/d/19AHxkMH7kOTWgu4__hy3lFixJjV5zxiP/view?usp=sharing)

## 1) Objectif du projet

Ce projet démontre un workflow MLOps complet orienté production:

- ingestion de données versionnées (DVC),
- pipeline d’entraînement orchestré (ZenML),
- tracking d’expériences et registre de modèles (MLflow + PostgreSQL),
- génération de rapports de drift (Evidently),
- boucle de monitoring pouvant déclencher une logique de retrain.

Cas d’usage: base technique pour un MVP MLOps reproductible et extensible (CI/CD, registry policy, déploiement, observabilité).

---

## 2) Stack technique

- **Orchestration pipeline**: ZenML `0.67.0`
- **Experiment tracking / Model Registry**: MLflow `2.12.1`
- **Versioning data**: DVC `3.50.1`
- **Drift monitoring**: Evidently `0.4.33`
- **DL framework**: PyTorch `2.3.1`
- **Data/ML libs**: NumPy, Pandas, scikit-learn, matplotlib, pyarrow
- **Object storage data**: MinIO (S3 compatible)
- **Infra locale**: Docker Compose (app + mlflow + postgres + minio)

---

## 3) Architecture fonctionnelle

### Pipeline entraînement

`src/pipelines/training_pipeline.py`

1. `ingest_data`: `dvc pull` + vérification du dataset CIFAR-10.
2. `upload_data_to_minio`: upload du dataset vers MinIO.
3. `validate_data`: checks de forme / labels.
4. `split_data`: séparation train/val + test.
5. `preprocess`: reshape + normalisation + création des DataLoader.
6. `train`: entraînement CNN + logs MLflow.
7. `evaluate`: métriques + artifacts (matrice de confusion, report).
8. `register_model`: enregistrement dans le Model Registry MLflow.
9. `export_model`: export TorchScript.

### Pipeline monitoring

`src/pipelines/monitoring_pipeline.py`

1. `load_latest_model`: charge la dernière version enregistrée.
2. `collect_inference_data`: échantillonnage du test set + prédictions.
3. `run_evidently_report`: rapport HTML/JSON de drift.
4. `trigger_decision`: décision de retrain via `share_of_drifted_columns`.
5. `store_monitoring_artifacts`: push des rapports dans MLflow.

---

## 4) Arborescence (principale)

```text
docker-compose.yml
docker/Dockerfile
requirements.txt
src/
	pipelines/
		training_pipeline.py
		monitoring_pipeline.py
	steps/
		training/
		monitoring/
	utils/
data/
monitoring/
artifacts/
```

---

## 5) Variables de configuration

Définies dans `src/utils/settings.py` et/ou via environnement:

- `MLFLOW_TRACKING_URI` (défaut: `http://mlflow:5000`)
- `MLFLOW_EXPERIMENT_NAME` (défaut: `cifar10_cnn`)
- `SIMULATE_DRIFT` (`0` ou `1`)
- `MINIO_ENDPOINT_URL` (défaut: `http://minio:9000`)
- `MINIO_ACCESS_KEY` (défaut: `minioadmin`)
- `MINIO_SECRET_KEY` (défaut: `minioadmin`)
- `MINIO_BUCKET` (défaut: `dvc`)
- `MINIO_DATA_PREFIX` (défaut: `datasets/cifar10/raw`)
- `DATA_RAW_DIR` (défaut: `data/raw`)
- `MONITORING_DIR` (défaut: `monitoring`)
- `ARTIFACTS_DIR` (défaut: `artifacts`)
- `MODEL_NAME` (défaut: `cifar10_cnn`)

---

## 6) Prérequis

- Docker + Docker Compose
- (Optionnel) Python 3.13.2 + venv pour exécuter hors conteneur

---

## 7) Démarrage rapide (recommandé: Docker)

### 7.1 Lancer les services

```bash
docker compose -f docker-compose.yml up -d --build
```

Services exposés:

- MLflow UI/API: `http://localhost:5000`
- ZenML Dashboard/API: `http://localhost:8237`
- PostgreSQL: `localhost:5432`
- MinIO API: `http://localhost:9000`
- MinIO Console: `http://localhost:9001`

Identifiants MinIO par défaut:

- User: `minioadmin`
- Password: `minioadmin`

### 7.2 Exécuter le pipeline d’entraînement

```bash
docker compose -f docker-compose.yml exec app bash -lc "python -m src.pipelines.training_pipeline"
```

### 7.3 Exécuter le pipeline de monitoring (sans drift simulé)

```bash
docker compose -f docker-compose.yml exec app bash -lc "python -m src.pipelines.monitoring_pipeline"
```

### 7.4 Exécuter le monitoring avec drift simulé

```bash
docker compose -f docker-compose.yml exec -e SIMULATE_DRIFT=1 app bash -lc "python -m src.pipelines.monitoring_pipeline"
```

### 7.5 Arrêter l’environnement

```bash
docker compose -f docker-compose.yml down
```

---

## 8) Exécution locale (hors Docker)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell

pip install -r requirements.txt

python -m src.pipelines.training_pipeline
python -m src.pipelines.monitoring_pipeline
```

> En local, vérifie que `MLFLOW_TRACKING_URI` pointe vers une instance MLflow active.

---

## 9) Observabilité & artifacts

### MLflow

- Experiments: métriques d’entraînement et de test.
- Model Registry: versions du modèle `cifar10_cnn`.
- Artifacts: rapport de classification, confusion matrix, rapports monitoring.

### Fichiers locaux

- `artifacts/model_torchscript.pt`
- `artifacts/confusion_matrix.png`
- `artifacts/classification_report.txt`
- `monitoring/evidently_report.html`
- `monitoring/evidently_report.json`
- `monitoring/inference.parquet`

---

## 10) Commandes utiles opérateur MLOps

### Ouvrir un shell dans l’app

```bash
docker compose -f docker-compose.yml exec app bash
```

### Voir les logs MLflow

```bash
docker compose -f docker-compose.yml logs -f mlflow
```

### Voir les logs ZenML Server

```bash
docker compose -f docker-compose.yml logs -f zenml-server
```

### Vérifier les containers

```bash
docker compose -f docker-compose.yml ps
```

---

## 11) Dépannage

### Warning protobuf / mlflow

Le warning autour de `google.protobuf.service` est un avertissement de dépréciation amont et ne bloque pas l’exécution du pipeline.

### Erreur “model is not JSON serializable” (ZenML)

Résolue dans cette base: le modèle est désormais chargé via une **step ZenML** (`load_latest_model`) et transmis comme artifact, pas comme paramètre Python brut.

### Dataset introuvable

Vérifie:

- la présence de `data/raw/cifar-10-batches-py`,
- la cohérence DVC (`dvc pull`),
- les volumes Docker montés sur `/workspace`.

### MLflow vide (aucun run affiché)

Vérifie:

- que tu es bien dans l’expérience `cifar10_cnn` (ou la valeur de `MLFLOW_EXPERIMENT_NAME`),
- que la page MLflow est rafraîchie (`http://localhost:5000`),
- que les pipelines ont été relancés après les changements de logging.

---

## 12) Limites actuelles et évolutions recommandées

- Ajouter des tests unitaires/intégration pour chaque step.
- Ajouter validation de schéma/qualité de données (Great Expectations ou similaire).
- Industrialiser le trigger retrain (event-driven / scheduler / policy registry).
- Ajouter CI/CD (lint, tests, build image, pipeline smoke test).
- Déployer vers orchestrateur distant (Kubernetes + stack ZenML dédiée).

---

## 13) Étape 13 — Exécuter (les commandes)

Cette séquence permet de lancer l’infrastructure, initialiser ZenML, exécuter l’entraînement, puis valider le monitoring avec et sans drift simulé.

### 13.1 Lancer l’infra (si pas déjà fait)

```bash
docker compose -f docker-compose.yml up -d --build
```

### 13.2 Initialiser ZenML

```bash
docker compose -f docker-compose.yml exec app bash -lc "zenml init"
```

### 13.3 Exécuter le pipeline d’entraînement

```bash
docker compose -f docker-compose.yml exec app bash -lc "python -m src.pipelines.training_pipeline"
```

### 13.4 Exécuter le pipeline de monitoring (sans drift)

```bash
docker compose -f docker-compose.yml exec app bash -lc "python -m src.pipelines.monitoring_pipeline"
```

### 13.5 Simuler un drift et déclencher la logique de retrain (démo)

```bash
docker compose -f docker-compose.yml exec -e SIMULATE_DRIFT=1 app bash -lc "python -m src.pipelines.monitoring_pipeline"
```

> Note: dans ce dépôt, le fichier présent est `docker-compose.yml` à la racine. Si vous ne disposez pas de `compose/docker-compose.yml`, utilisez `-f docker-compose.yml`.

---

## 14) Bonnes pratiques pour contribution

- Garder les steps **purs**, déterministes et idempotents.
- Typage explicite des entrées/sorties de steps pour compatibilité materializers.
- Traiter toute nouvelle dépendance au niveau `requirements.txt` + Dockerfile.
- Ne pas coupler logique métier et configuration runtime (utiliser variables d’environnement).

---

## 15) Résumé exécutable (TL;DR)

```bash
# 1) Infra
docker compose -f docker-compose.yml up -d --build

# 2) Train
docker compose -f docker-compose.yml exec app bash -lc "python -m src.pipelines.training_pipeline"

# 3) Monitoring
docker compose -f docker-compose.yml exec app bash -lc "python -m src.pipelines.monitoring_pipeline"

# 4) Monitoring avec drift simulé
docker compose -f docker-compose.yml exec -e SIMULATE_DRIFT=1 app bash -lc "python -m src.pipelines.monitoring_pipeline"
```


import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
SIMULATE_DRIFT = os.getenv("SIMULATE_DRIFT", "0") == "1"

DATA_RAW_DIR = "data/raw"
MONITORING_DIR = "monitoring"
ARTIFACTS_DIR = "artifacts"
MODEL_NAME = "cifar10_cnn"
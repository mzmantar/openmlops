import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "cifar10_cnn")
SIMULATE_DRIFT = os.getenv("SIMULATE_DRIFT", "0") == "1"

MINIO_ENDPOINT_URL = os.getenv("MINIO_ENDPOINT_URL", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "dvc")
MINIO_DATA_PREFIX = os.getenv("MINIO_DATA_PREFIX", "datasets/cifar10/raw")

DATA_RAW_DIR = "data/raw"
MONITORING_DIR = "monitoring"
ARTIFACTS_DIR = "artifacts"
MODEL_NAME = "cifar10_cnn"
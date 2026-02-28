from zenml import step
import mlflow
import mlflow.pytorch
import torch
from src.utils.settings import MLFLOW_TRACKING_URI, MODEL_NAME


@step
def load_latest_model() -> torch.nn.Module:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    versions = client.search_model_versions(
        filter_string=f"name='{MODEL_NAME}'",
        max_results=1,
        order_by=["creation_timestamp DESC"],
    )
    if not versions:
        raise ValueError(f"No registered versions found for model '{MODEL_NAME}'.")

    latest = versions[0]
    model_uri = f"models:/{MODEL_NAME}/{latest.version}"
    return mlflow.pytorch.load_model(model_uri)
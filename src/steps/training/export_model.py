from zenml import step
import os
import torch
import mlflow
from src.utils.settings import ARTIFACTS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

@step
def export_model(model: torch.nn.Module) -> str:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    export_path = os.path.join(ARTIFACTS_DIR, "model_torchscript.pt")

    model.eval()
    example = torch.randn(1, 3, 32, 32)
    traced = torch.jit.trace(model.cpu(), example)
    traced.save(export_path)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.log_artifact(export_path)

    return export_path
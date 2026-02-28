from zenml import step
import mlflow
import mlflow.pytorch
import torch
from src.utils.settings import MLFLOW_TRACKING_URI, MODEL_NAME, MLFLOW_EXPERIMENT_NAME

@step
def register_model(model: torch.nn.Module) -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    # log model to current run
    mlflow.pytorch.log_model(model, artifact_path="model")
    # register from the run artifact
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, MODEL_NAME)
    return model_uri
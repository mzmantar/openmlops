from zenml import step
import mlflow
from src.utils.settings import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

@step(enable_cache=False)
def store_monitoring_artifacts(html_report_path: str, json_report_path: str):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    while mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name="store_monitoring_artifacts"):
        mlflow.log_params({"monitoring_report_count": 2})
        mlflow.log_metrics({"monitoring_artifacts_logged": 2})
        mlflow.log_artifact(html_report_path)
        mlflow.log_artifact(json_report_path)
    return True
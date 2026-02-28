from zenml import step
import mlflow

from src.utils.minio_utils import upload_directory_to_minio
from src.utils.settings import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MINIO_ENDPOINT_URL,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_BUCKET,
    MINIO_DATA_PREFIX,
)


@step(enable_cache=False)
def upload_data_to_minio(cifar_dir: str) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    result = upload_directory_to_minio(
        local_dir=cifar_dir,
        endpoint_url=MINIO_ENDPOINT_URL,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        bucket_name=MINIO_BUCKET,
        prefix=MINIO_DATA_PREFIX,
    )

    while mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name="upload_data_to_minio"):
        mlflow.log_params(
            {
                "minio_endpoint_url": MINIO_ENDPOINT_URL,
                "minio_bucket": MINIO_BUCKET,
                "minio_data_prefix": MINIO_DATA_PREFIX,
            }
        )
        mlflow.log_metrics({"minio_uploaded_files": result["uploaded_files"]})

    return result

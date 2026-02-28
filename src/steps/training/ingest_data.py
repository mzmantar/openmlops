from zenml import step
from src.utils.dvc_utils import dvc_pull
from src.utils.settings import DATA_RAW_DIR
import os

@step
def ingest_data() -> str:
    dvc_pull()
    path = os.path.join(DATA_RAW_DIR, "cifar-10-batches-py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CIFAR-10 not found at {path}")
    return path
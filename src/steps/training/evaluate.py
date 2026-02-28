from zenml import step
import torch
import numpy as np
import mlflow
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os
from src.utils.settings import ARTIFACTS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

@step(enable_cache=False)
def evaluate(model: torch.nn.Module, preprocess_out: tuple) -> dict:
    _, _, test_loader = preprocess_out

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(yb.numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    test_samples = len(y_true)
    test_batch_size = getattr(test_loader, "batch_size", None)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    report = classification_report(y_true, y_pred)
    rep_path = os.path.join(ARTIFACTS_DIR, "classification_report.txt")
    with open(rep_path, "w") as f:
        f.write(report)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    while mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name="evaluate_cnn"):
        mlflow.log_params(
            {
                "test_samples": test_samples,
                "test_batch_size": test_batch_size,
                "evaluation_average": "macro",
            }
        )
        mlflow.log_metrics(
            {
                "test_accuracy": acc,
                "test_f1_macro": f1,
                "test_error_rate": 1.0 - acc,
            }
        )
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(rep_path)

    return {"accuracy": acc, "f1_macro": f1}
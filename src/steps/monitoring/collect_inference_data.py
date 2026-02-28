from zenml import step
import os, time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from src.utils.settings import MONITORING_DIR, DATA_RAW_DIR, SIMULATE_DRIFT

def _load_test_batch(cifar_dir: str):
    p = os.path.join(cifar_dir, "test_batch")
    with open(p, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.array(d[b"labels"])
    return X, y

@step(enable_cache=False)
def collect_inference_data(model: torch.nn.Module, n_samples: int = 200) -> str:
    os.makedirs(MONITORING_DIR, exist_ok=True)
    cifar_dir = os.path.join(DATA_RAW_DIR, "cifar-10-batches-py")

    X, y = _load_test_batch(cifar_dir)
    idx = np.random.choice(len(X), size=n_samples, replace=False)
    X = X[idx]
    y = y[idx]

    # simulate drift: strong brightness shift
    if SIMULATE_DRIFT:
        X = np.clip(X * 0.5 + 0.4, 0, 1)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X))
        probs = F.softmax(logits, dim=1).numpy()
        pred = probs.argmax(axis=1)

    df = pd.DataFrame(probs, columns=[f"proba_{i}" for i in range(10)])
    df["pred_class"] = pred
    df["true_class"] = y
    df["ts"] = int(time.time())

    out_path = os.path.join(MONITORING_DIR, "inference.parquet")
    df.to_parquet(out_path, index=False)

    return out_path
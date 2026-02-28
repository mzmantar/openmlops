from zenml import step
import os, pickle
import numpy as np
from sklearn.model_selection import train_test_split

def _load_all_train(cifar_dir: str):
    xs, ys = [], []
    for i in range(1, 6):
        p = os.path.join(cifar_dir, f"data_batch_{i}")
        with open(p, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        xs.append(d[b"data"])
        ys.append(d[b"labels"])
    X = np.concatenate(xs, axis=0)  # (50000, 3072)
    y = np.array(sum(ys, []))
    return X, y

def _load_test(cifar_dir: str):
    p = os.path.join(cifar_dir, "test_batch")
    with open(p, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = d[b"data"]  # (10000, 3072)
    y = np.array(d[b"labels"])
    return X, y

@step
def split_data(cifar_dir: str) -> tuple:
    X, y = _load_all_train(cifar_dir)
    X_test, y_test = _load_test(cifar_dir)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    return (X_train, y_train, X_val, y_val, X_test, y_test)
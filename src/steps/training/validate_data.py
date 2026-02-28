from zenml import step
import pickle
import os
import numpy as np

def _load_batch(file_path: str):
    with open(file_path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"]  # (10000, 3072)
    y = np.array(d[b"labels"])
    return x, y

@step
def validate_data(cifar_dir: str) -> str:
    batch = os.path.join(cifar_dir, "data_batch_1")
    x, y = _load_batch(batch)

    assert x.shape[1] == 3072
    assert y.min() >= 0 and y.max() <= 9
    assert len(x) == len(y)

    return cifar_dir
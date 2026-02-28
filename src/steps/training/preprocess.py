from zenml import step
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def _to_images(X_flat: np.ndarray) -> np.ndarray:
    # (N, 3072) -> (N, 3, 32, 32)
    return X_flat.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

def _normalize(X: np.ndarray) -> np.ndarray:
    mean = np.array(CIFAR_MEAN, dtype=np.float32).reshape(1,3,1,1)
    std = np.array(CIFAR_STD, dtype=np.float32).reshape(1,3,1,1)
    return (X - mean) / std

@step
def preprocess(split_data_out: tuple, batch_size: int = 128) -> tuple:
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_out

    X_train = _normalize(_to_images(X_train))
    X_val   = _normalize(_to_images(X_val))
    X_test  = _normalize(_to_images(X_test))

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))
    test_ds  = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
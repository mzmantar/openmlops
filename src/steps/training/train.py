from zenml import step
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from src.utils.settings import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

@step(enable_cache=False)
def train(preprocess_out: tuple, epochs: int = 3, lr: float = 1e-3) -> torch.nn.Module:
    train_loader, val_loader, _ = preprocess_out

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    train_batch_size = getattr(train_loader, "batch_size", None)
    val_batch_size = getattr(val_loader, "batch_size", None)
    best_val_acc = 0.0

    while mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name="train_cnn"):
        mlflow.log_params(
            {
                "epochs": epochs,
                "lr": lr,
                "device": device,
                "optimizer": "Adam",
                "loss_fn": "CrossEntropyLoss",
                "train_samples": train_samples,
                "val_samples": val_samples,
                "train_batch_size": train_batch_size,
                "val_batch_size": val_batch_size,
                "num_classes": model.net[-1].out_features,
            }
        )

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            train_correct = 0
            train_total = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
                train_correct += (logits.argmax(dim=1) == yb).sum().item()
                train_total += yb.size(0)
            train_loss = total_loss / len(train_loader.dataset)
            train_acc = train_correct / max(train_total, 1)

            # val acc/loss
            model.eval()
            correct = 0
            total = 0
            val_loss_sum = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    val_loss_sum += criterion(logits, yb).item() * xb.size(0)
                    pred = logits.argmax(dim=1)
                    correct += (pred == yb).sum().item()
                    total += yb.size(0)
            val_acc = correct / total
            val_loss = val_loss_sum / len(val_loader.dataset)
            best_val_acc = max(best_val_acc, val_acc)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

        mlflow.log_metrics({"best_val_acc": best_val_acc, "final_train_acc": train_acc})

        return model
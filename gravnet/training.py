import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def val_epoch(model: nn.Module, loss_fn: nn.Module, val_loader: DataLoader[torch.Tensor]) -> float:
    model.eval()

    val_loss, num_batches = 0.0, len(val_loader)
    with torch.no_grad():
        for X, y in val_loader:
            pred = model(X.to(device))
            val_loss += loss_fn(pred, y.to(device))
    
    return val_loss/num_batches

def train_epoch(
    model: nn.Module, loss_fn: nn.Module, optim: torch.optim.Optimizer,
    train_loader: DataLoader[torch.Tensor], desc: str
) -> float:
    model.train()

    train_loss, num_batches = 0.0, len(train_loader)
    progress_bar = tqdm(total=num_batches, desc=desc)
    for batch, (X, y) in enumerate(train_loader, start=1):
        optim.zero_grad()

        pred = model(X.to(device))
        loss: torch.Tensor = loss_fn(pred, y.to(device))
        loss.backward()
        optim.step()

        train_loss += loss.item()
        progress_bar.update(1)
        progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}", "Train Loss": f"{train_loss/batch:.4f}"})

    progress_bar.close()
    return train_loss/num_batches

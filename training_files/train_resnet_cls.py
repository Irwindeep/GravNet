import torch
import torch.nn as nn
from gravnet.utils.datasets import WaveDetData
from torch.utils.data import DataLoader
from gravnet.resnet import ResNetClassifier
from gravnet.training import train_epoch, val_epoch
from tqdm.auto import tqdm

torch.manual_seed(12)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

train_dataset = WaveDetData(root="gw_data", split="train", download=True)
val_dataset = WaveDetData(root="gw_data", split="val", download=False)
test_dataset = WaveDetData(root="gw_data", split="test", download=False)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

model = ResNetClassifier(in_channels=1, num_params=3).to(device)
epochs, lr = 40, 1e-3
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

train_losses, val_losses = [], []
best_val_loss = float("inf")
for epoch in range(1, epochs+1):
    desc = f"Epoch [{epoch:02d}/{epochs}]"
    train_loss = train_epoch(model, loss_fn, optim, train_loader, desc)
    val_loss = val_epoch(model, loss_fn, val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "model_weights.pth")

print()

mse_loss, mae_loss = 0.0, 0.0
with torch.no_grad():
    for X, y in tqdm(test_loader, desc="Evaluating Model"):
        pred = model(X.to(device))
        mse_loss += loss_fn(pred, y.to(device)).item()
        mae_loss += torch.mean(torch.abs(pred - y.to(device))).item()

print(f"MSE Loss: {mse_loss/len(test_loader):.4f}")
print(f"MAE Loss: {mae_loss/len(test_loader):.4f}")

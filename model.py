"""
房价预测调参实验

Assembled from your step-by-step solutions.
"""

# Step 2 - standardize_numeric (not yet solved)
# TODO: implement

# Step 3 - one_hot_encode (not yet solved)
# TODO: implement

# Step 4 - build_features (not yet solved)
# TODO: implement

# Step 5 - get_net
import torch
import torch.nn as nn

def get_net(n_features: int) -> nn.Module:
    layer = nn.Linear(n_features, 1)
    nn.init.normal_(layer.weight, mean=0, std=0.01)
    nn.init.normal_(layer.bias, mean=0, std=0.01)
    return layer

# Step 6 - log_rmse
import torch
import torch.nn as nn

def log_rmse(net: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        pred = net(features)
        clipped = torch.max(pred, torch.tensor(1.0))
        log_pred = torch.log(clipped)
        log_label = torch.log(labels)
        mse = torch.mean(torch.square(log_pred - log_label))
        rmse = torch.sqrt(mse)
    return float(rmse)

# Step 7 - train_housing (not yet solved)
# TODO: implement

# Step 8 - compare_losses (not yet solved)
# TODO: implement

# Step 9 - log_rmse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def log_rmse(net: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        pred = net(features)
        clipped = torch.max(pred, torch.tensor(1.0, device=pred.device))
        log_pred = torch.log(clipped)
        log_y = torch.log(labels)
        mse_val = torch.mean(torch.square(log_pred - log_y))
        rmse_val = torch.sqrt(mse_val)
    return float(rmse_val)

def get_net(n_features: int):
    net = nn.Linear(n_features, 1)
    nn.init.normal_(net.weight, mean=0, std=0.01)
    nn.init.normal_(net.bias, mean=0, std=0.01)
    return net

def best_learning_rate(
    X: "torch.Tensor",
    y: "torch.Tensor",
    lrs: list[float] = None,
    num_epochs: int = 12,
    batch_size: int = 16,
    seed: int = 0,
) -> float:
    if lrs is None:
        lrs = [0.01, 0.05, 0.2]
    dataset = TensorDataset(X, y)
    best_lr = None
    best_score = float("inf")
    for i, lr in enumerate(lrs):
        torch.manual_seed(seed + i)
        net = get_net(X.shape[1])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for _ in range(num_epochs):
            for bx, by in loader:
                opt.zero_grad()
                out = net(bx)
                loss = criterion(out, by)
                loss.backward()
                opt.step()
        score = log_rmse(net, X, y)
        if score < best_score:
            best_score = score
            best_lr = lr
    return best_lr

# Step 10 - compare_optimizers (not yet solved)
# TODO: implement

# Step 11 - tuning_report (not yet solved)
# TODO: implement

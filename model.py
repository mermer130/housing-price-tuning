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

# Step 8 - log_rmse
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


def compare_losses(
    X: "torch.Tensor",
    y: "torch.Tensor",
    num_epochs: int = 15,
    lr: float = 0.05,
    batch_size: int = 16,
    seed: int = 0,
) -> tuple:

    dataset = TensorDataset(X, y)

    # ===== MSE训练 seed =====
    torch.manual_seed(seed)
    net_mse = get_net(X.shape[1])
    loader_mse = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt_mse = torch.optim.SGD(net_mse.parameters(), lr=lr)
    mse_crit = nn.MSELoss()
    for _ in range(num_epochs):
        for bx, by in loader_mse:
            opt_mse.zero_grad()
            out = net_mse(bx)
            loss = mse_crit(out, by)
            loss.backward()
            opt_mse.step()
    log_rmse_mse = log_rmse(net_mse, X, y)

    # ===== MAE(L1Loss)训练 seed+1 =====
    torch.manual_seed(seed + 1)
    net_mae = get_net(X.shape[1])
    loader_mae = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt_mae = torch.optim.SGD(net_mae.parameters(), lr=lr)
    mae_crit = nn.L1Loss()
    for _ in range(num_epochs):
        for bx, by in loader_mae:
            opt_mae.zero_grad()
            out = net_mae(bx)
            loss = mae_crit(out, by)
            loss.backward()
            opt_mae.step()
    log_rmse_mae = log_rmse(net_mae, X, y)

    return (log_rmse_mse, log_rmse_mae)

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

# Step 10 - log_rmse
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

def compare_optimizers(
    X: "torch.Tensor",
    y: "torch.Tensor",
    num_epochs: int = 15,
    lr: float = 0.05,
    batch_size: int = 16,
    seed: int = 0,
) -> tuple:
    X = X.float()
    y = y.float()
    y = y.reshape(-1, 1)
    dataset = TensorDataset(X, y)

    # Adam seed
    torch.manual_seed(seed)
    net_adam = get_net(X.shape[1])
    loader_adam = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt_adam = torch.optim.Adam(net_adam.parameters(), lr=lr)
    crit = nn.MSELoss()
    for _ in range(num_epochs):
        for bx, by in loader_adam:
            opt_adam.zero_grad()
            out = net_adam(bx)
            loss = crit(out, by)
            loss.backward()
            opt_adam.step()
    log_rmse_adam = log_rmse(net_adam, X, y)

    # SGD seed+1
    torch.manual_seed(seed + 1)
    net_sgd = get_net(X.shape[1])
    loader_sgd = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt_sgd = torch.optim.SGD(net_sgd.parameters(), lr=lr)
    for _ in range(num_epochs):
        for bx, by in loader_sgd:
            opt_sgd.zero_grad()
            out = net_sgd(bx)
            loss = crit(out, by)
            loss.backward()
            opt_sgd.step()
    log_rmse_sgd = log_rmse(net_sgd, X, y)

    return (log_rmse_adam, log_rmse_sgd)

# Step 11 - log_rmse
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

def make_net(n_features: int, init_type: str):
    net = nn.Linear(n_features, 1)
    if init_type == "normal":
        nn.init.normal_(net.weight, mean=0, std=0.01)
        nn.init.normal_(net.bias, mean=0, std=0.01)
    elif init_type == "xavier":
        nn.init.xavier_uniform_(net.weight)
        nn.init.zeros_(net.bias)
    return net

def train_once(X_train, y_train, X_eval, y_eval, n_features, init_type, seed, epochs, lr=0.05, batch_size=16):
    torch.manual_seed(seed)
    net = make_net(n_features, init_type)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit = nn.MSELoss()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            out = net(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
    return log_rmse(net, X_eval, y_eval)

def tuning_report(
    X: "torch.Tensor",
    y: "torch.Tensor",
    k: int = 3,
    seed: int = 0,
) -> tuple:
    X = X.float()
    y = y.float()
    y = y.reshape(-1, 1)
    n, n_features = X.shape

    rmse_normal = train_once(X, y, X, y, n_features, "normal", seed, 12)
    rmse_xavier = train_once(X, y, X, y, n_features, "xavier", seed + 1, 12)
    rmse_short = train_once(X, y, X, y, n_features, "normal", seed + 2, 8)
    rmse_long = train_once(X, y, X, y, n_features, "normal", seed + 3, 20)

    fold_rmses = []
    for fold in range(k):
        val_mask = torch.tensor([i % k == fold for i in range(n)])
        train_mask = ~val_mask
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        fold_seed = seed + 10 + fold
        val_rmse = train_once(X_tr, y_tr, X_val, y_val, n_features, "normal", fold_seed, 10)
        fold_rmses.append(val_rmse)
    kfold_mean = sum(fold_rmses) / len(fold_rmses)

    return (rmse_normal, rmse_xavier, rmse_short, rmse_long, kfold_mean)

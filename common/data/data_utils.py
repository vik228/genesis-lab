"""
common/data/data_utils.py

Expectation:
Implement generic helper functions for dataset splitting and DataLoader creation.

    - `create_train_val_test_split(X, Y, train_ratio, val_ratio, device=None)`:
        1. Slice input tensors `X` and `Y` by mathematically splitting length combinations.
        2. Optimization: If `device` is passed, explicitly call `.to(device)` so VRAM/RAM holds them natively.
        3. Return split tuples `(train_X, train_Y), ...`
        
    - `get_dataloaders(splits, batch_size, num_workers=0)`:
        1. Combine splits inside `TensorDataset()`.
        2. Generate PyTorch `DataLoader` constructs for train (shuffle=True), val, and test.
        3. Warning: Keep `num_workers=0` on Apple Silicon (MPS) to avoid kernel crashes when data is pre-allocated onto device memory.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset


def create_train_val_test_split(
    X: torch.tensor, Y: torch.tensor, train_ratio: float, val_ratio: float, device=None
):
    data_size = X.shape[0]
    n1 = int(train_ratio * data_size)
    n2 = int((train_ratio + val_ratio) * data_size)

    train_X, train_Y = X[:n1], Y[:n1]
    val_X, val_Y = X[n1:n2], Y[n1:n2]
    test_X, test_Y = X[n2:], Y[n2:]
    if device:
        train_X, train_Y = train_X.to(device), train_Y.to(device)
        val_X, val_Y = val_X.to(device), val_Y.to(device)
        test_X, test_Y = test_X.to(device), test_Y.to(device)

    return (train_X, train_Y), (val_X, val_Y), (test_X, test_Y)


def get_dataloaders(
    splits: list[tuple[torch.tensor, torch.tensor]],
    batch_size: int,
    num_workers: int = 0,
):
    train_dataset = TensorDataset(splits[0][0], splits[0][1])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataset = TensorDataset(splits[1][0], splits[1][1])
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dataset = TensorDataset(splits[2][0], splits[2][1])
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader

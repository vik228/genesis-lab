"""
common/utils/trainer.py

Expectation:
Implement a reusable `GenericTrainer` class that abstracts standard PyTorch training operations for any experiment across the project:

    - `__init__(self, model, optimizer, criterion, device=None)`: Save references and explicitly move the `model` to the provided `device`.
    - `train_epoch(self, dataloader, scheduler=None)`: 
        1. Set `model.train()`.
        2. Iterate over batches.
        3. Move `data` and `target` to `device` (no-op if already pre-loaded on GPU).
        4. Forward pass, loss calculation, `loss.backward()`, and `optimizer.step()`.
        5. Calculate and return the average training loss.
    - `evaluate(self, dataloader)`:
        1. Set `model.eval()`.
        2. Use `with torch.no_grad():` block.
        3. Accumulate validation loss dynamically.
        4. Return validation loss, and potentially metrics like perplexity.
    - `fit(self, train_loader, val_loader, epochs, scheduler=None)`: 
        Execute training loop for `epochs`, tracking and logging epoch-wise metrics without assuming a particular model structure.
"""

import torch
from typing import Any


class GenericTrainer:

    def __init__(self, model: Any, optimiser: Any, criterion: Any, device=None) -> None:
        self.model = model
        self.optimiser = optimiser
        self.criterion = criterion
        self.device = device
        if self.device:
            self.model = self.model.to(self.device)

    def train_epoch(self, dataloader: Any, scheduler=None) -> float:
        self.model.train()
        total_loss = 0
        for idx, (data, target) in enumerate(dataloader):
            if self.device:
                data, target = data.to(self.device), target.to(self.device)
            logits = self.model(data)
            loss = self.criterion(logits, target)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def fit(
        self, train_loader: Any, val_loader: Any, epochs: int, scheduler=None
    ) -> tuple:
        train_losses = []
        val_losses = []
        val_perplexities = []
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, scheduler)
            val_loss, val_perplexity = self.evaluate(val_loader)
            print(
                f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Perplexity = {val_perplexity:.4f}"
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_perplexities.append(val_perplexity)
        return train_losses, val_losses, val_perplexities

    def evaluate(self, dataloader: Any) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, target in dataloader:
                if self.device:
                    data, target = data.to(self.device), target.to(self.device)
                logits = self.model(data)
                loss = self.criterion(logits, target)
                total_loss += loss.item()
        avg_total_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_total_loss)).item()
        return avg_total_loss, perplexity

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Trainer:
    def __init__(
        self,
        num_epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer

    def train(self) -> None:
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()

            # Evaluate
            if epoch % 20 == 0:
                valid_loss = self.evaluate(self.val_loader)
                logging.info(
                    f"Epoch: {epoch} | "
                    f"Train loss: {train_loss:.4f} | "
                    f"Validation loss: {valid_loss:.4f}"
                )

    def train_epoch(self) -> float:
        self.model.train()  # Set PyTorch module to training mode
        total_loss = 0.0

        for batch in self.train_loader:
            X, Y = batch

            # Forward pass
            logits, loss = self.model(idx=X, targets=Y)

            # Backward pass
            self.model.zero_grad(set_to_none=True)  # Zero the gradients
            loss.backward()

            # Parameter updates
            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        mean_loss = total_loss / len(self.train_loader)
        return mean_loss

    @torch.inference_mode()  # More efficient than torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate the model on the validation or test set."""
        self.model.eval()  # Set PyTorch module to evaluation mode

        losses = []
        for batch in dataloader:
            X, Y = batch
            logits, loss = self.model(idx=X, targets=Y)
            losses.append(loss.item())

        mean_loss = torch.tensor(losses).mean().item()
        return mean_loss

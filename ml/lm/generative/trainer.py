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
        device: torch.device,
    ) -> None:
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.device = device

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

        losses = []
        for batch in self.train_loader:
            X, Y = batch

            # Move to device
            X = X.to(self.device, non_blocking=True)
            Y = Y.to(self.device, non_blocking=True)

            # Forward pass
            logits, loss = self.model(idx=X, targets=Y)

            # Backward pass
            # Zero the gradients: setting `set_to_none=True` will
            # deallocate the gradients, which saves memory
            self.model.zero_grad(set_to_none=True)
            loss.backward()

            # Parameter updates
            self.optimizer.step()

            # Append loss
            losses.append(loss.item())

        mean_loss = torch.tensor(losses).mean().item()
        return mean_loss

    @torch.inference_mode()  # More efficient than torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate the model on the validation or test set."""
        self.model.eval()  # Set PyTorch module to evaluation mode

        losses = []
        for batch in dataloader:
            X, Y = batch
            X = X.to(self.device, non_blocking=True)
            Y = Y.to(self.device, non_blocking=True)
            logits, loss = self.model(idx=X, targets=Y)
            losses.append(loss.item())

        mean_loss = torch.tensor(losses).mean().item()

        self.model.train()  # Reset module back to training mode

        return mean_loss

import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ml.diffusion.ddpm.diffuser import DiffuserDDPM

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
        T: int,
        diffuser: DiffuserDDPM,
        denoising_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.T = T
        self.diffuser = diffuser
        self.denoising_model = denoising_model
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
        self.denoising_model.train()  # Set PyTorch module to training mode

        losses = []
        for batch in self.train_loader:
            # Extract image: first element is image, second element is label (not used)
            x_0, _ = batch

            # Extract batch size
            batch_size = x_0.size(0)

            # Move to device
            x_0 = x_0.to(self.device)

            # Noising process
            #   1. Sample `t` from a discrete uniform distribution (Algorithm 1 line 3)
            #   2. From the original image x_0, sample a noised image at timestep `t`
            t = torch.randint(
                low=0, high=self.T, size=(batch_size,), device=self.device
            ).long()
            x_noisy, noise = self.diffuser.noising_step(x_0, t)

            # Denoising process
            #   1. Predict the noise (forward pass through denoising model)
            #   2. Compute the L2 loss between the actual noise and the predicted noise
            #   3. Set the gradients to zero before doing the backpropagation step.
            #       This is necessary because, by default, PyTorch accumulates the
            #       gradients on subsequent backward passes i.e. subsequent calls of
            #       loss.backward(). Note: setting `set_to_none=True` will deallocate
            #       the gradients, which saves memory.
            #   4. Backward pass (through the denoising model)
            #   5. Update parameters of the denoising model
            pred_noise = self.denoising_model(x_noisy, t)
            loss = F.mse_loss(noise, pred_noise)
            self.denoising_model.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # Append loss
            losses.append(loss.item())

        mean_loss = torch.tensor(losses).mean().item()
        return mean_loss

    @torch.inference_mode()  # More efficient than torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate the model on the validation or test set."""
        self.denoising_model.eval()  # Set PyTorch module to evaluation mode

        losses = []
        for batch in dataloader:
            X, Y = batch
            X = X.to(self.device, non_blocking=True)
            Y = Y.to(self.device, non_blocking=True)
            logits, loss = self.denoising_model(idx=X, targets=Y)
            losses.append(loss.item())

        mean_loss = torch.tensor(losses).mean().item()

        self.denoising_model.train()  # Reset module back to training mode

        return mean_loss

import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.autoencoders.ae import Autoencoder
from ml.autoencoders.vae import VAE
from ml.tensor import Tensor


def train(
    X: Tensor, learning_rate: float = 1e-3, batch_size: int = 128, num_epochs: int = 15
):
    # Data
    X = torch.tensor(X).float()
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model and optimizer
    model = Autoencoder(d_x=X.shape[1], d_hidden=256, d_z=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()  # Zero the gradients
            x = batch[0]  # Get batch
            output = model(x)
            loss = F.mse_loss(output, x, reduction="sum")  # Reconstruction loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(X)}")


if __name__ == "__main__":
    X = torch.randn(1000, 100)
    train(X)

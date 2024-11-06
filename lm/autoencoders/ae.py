import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from lm.tensor import Tensor


class Autoencoder(nn.Module):
    def __init__(self, x_dim: int, hidden_dim: int, z_dim: int = 10):
        super(Autoencoder, self).__init__()
        self.enc_layer1 = nn.Linear(x_dim, hidden_dim)
        self.enc_layer2 = nn.Linear(hidden_dim, z_dim)
        self.dec_layer1 = nn.Linear(z_dim, hidden_dim)
        self.dec_layer2 = nn.Linear(hidden_dim, x_dim)

    def encoder(self, x):
        x = F.relu(self.enc_layer1(x))
        z = F.relu(self.enc_layer2(x))
        return z

    def decoder(self, z):
        output = F.relu(self.dec_layer1(z))
        output = F.relu(self.dec_layer2(output))
        return output

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output


def train(
    X: Tensor, learning_rate: float = 1e-3, batch_size: int = 128, num_epochs: int = 15
):
    # Data
    X = torch.tensor(X).float()
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model and optimizer
    model = Autoencoder(x_dim=X.shape[1], hidden_dim=256, z_dim=50)
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ml.tensor import Tensor


class Autoencoder(nn.Module):
    def __init__(self, d_x: int, d_hidden: int, d_z: int):
        """Simple autoencoder with a single hidden layer.

        Args:
            d_x: feature/embedding dimensionality of input.
            d_hidden: dimensionality of the intermediate compressed
                (resp. reconstructed) representation for the encoder (resp. decoder).
            d_z: dimensionality of the latent space. This is the dimensionality of the
                output (resp. input) of the encoder (resp. decoder).
        """
        super(Autoencoder, self).__init__()
        self.enc_layer1 = nn.Linear(d_x, d_hidden)
        self.enc_layer2 = nn.Linear(d_hidden, d_z)
        self.dec_layer1 = nn.Linear(d_z, d_hidden)
        self.dec_layer2 = nn.Linear(d_hidden, d_x)

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

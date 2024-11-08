import torch.nn as nn
import torch.nn.functional as F

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

    def encoder(self, x: Tensor) -> Tensor:
        x = F.relu(self.enc_layer1(x))
        z = F.relu(self.enc_layer2(x))
        return z

    def decoder(self, z: Tensor) -> Tensor:
        output = F.relu(self.dec_layer1(z))
        output = F.relu(self.dec_layer2(output))
        return output

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        output = self.decoder(z)
        return output

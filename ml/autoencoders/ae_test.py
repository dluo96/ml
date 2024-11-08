import unittest

import torch

from ml.autoencoders.ae import Autoencoder


class TestAutoencoder(unittest.TestCase):
    def setUp(self):
        self.d_x = 20
        self.d_hidden = 10
        self.d_z = 5
        self.B = 1  # Batch size of 1
        self.model = Autoencoder(self.d_x, self.d_hidden, self.d_z)

    def test_encoder(self):
        x = torch.randn(self.B, self.d_x)
        z = self.model.encoder(x)
        assert z.shape == (1, self.d_z)

    def test_decoder(self):
        z = torch.randn(self.B, self.d_z)
        output = self.model.decoder(z)
        assert output.shape == (1, self.d_x)

    def test_forward(self):
        x = torch.randn(self.B, self.d_x)
        output = self.model(x)
        assert x.shape == output.shape


if __name__ == "__main__":
    unittest.main()

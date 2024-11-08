"""Variational Autoencoder (VAE) implementation.

References:
    - https://www.jeremyjordan.me/variational-autoencoders/
    - https://mbernste.github.io/posts/vae/
    - https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.tensor import Tensor


class VAE(nn.Module):
    """Minimal implementation of a Variational Autoencoder (VAE).

    The crucial thing to remember about a standard autoencoder is that its encoder
    outputs a single value for each latent dimension. The decoder network then
    subsequently takes these values and attempts to reconstruct the original input.

    Using a VAE however, we can describe latent attributes probabilistically.
    Concretely, we will now represent each latent attribute for a given input as a
    probability distribution. When decoding from the latent state, we'll randomly
    sample from each latent state distribution to generate a vector as input for our
    decoder model.

    The main benefit of a VAE is that we can learn smooth latent state representations
    of the input data. In contrast, standard autoencoders simply need to learn an
    encoding which allows us to reconstruct the input.

    Note on nomenclature: for VAEs,
        - The encoder model is sometimes referred to as the recognition model.
        - The decoder model is sometimes referred to as the generative model.
    """

    def __init__(self, x_dim: int, hidden_dim: int, z_dim: int = 10):
        super(VAE, self).__init__()

        # Encoder layers
        self.enc_layer1 = nn.Linear(x_dim, hidden_dim)
        self.enc_layer2_mu = nn.Linear(hidden_dim, z_dim)
        self.enc_layer2_log_var = nn.Linear(hidden_dim, z_dim)

        # Decoder layers
        self.dec_layer1 = nn.Linear(z_dim, hidden_dim)
        self.dec_layer2 = nn.Linear(hidden_dim, x_dim)

    def encoder(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Rather than directly outputting values for the latent state (as would be
        the case in a standard autoencoder), the VAE encoder outputs parameters that
        describe the distribution for each dimension in the latent space.

        Here, we assume that our prior distribution is normal. Thus, the encoder will
        output two vectors:
        - Vector of means (one mean per latent dimension).
        - Vector of log-variances (one log-variance per latent dimension).
            NB: if we were to build a true multivariate Gaussian model, we'd need to
            define a covariance matrix describing how each of the dimensions are
            correlated. However, we'll make a simplifying assumption that our
            covariance matrix only has nonzero values on the diagonal, allowing us to
            describe this information in a simple vector.

        NOTE: to deal with the fact that the network may learn negative values for σ,
        we instead make the network learn log(σ^2) = 2*log(σ). In the decoder, we
        simply divide by 2 and exponentiate this quantity to get σ (the standard
        deviation σ of the latent distribution).
        """
        x = F.relu(self.enc_layer1(x))
        mu = F.relu(self.enc_layer2_mu(x))
        log_var = F.relu(self.enc_layer2_log_var(x))
        return mu, log_var

    def sample_z(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """This sampling process requires some extra attention. When training
        the model, we can't backpropagate through a random sampling process. To solve
        this, we use the reparametrization trick: now, the randomness comes from
        another tensor `epsilon` which isn't part of the computational graph.
        """
        std = torch.exp(log_var / 2)

        # Reparametrization trick
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon

        return z

    def decoder(self, z: Tensor) -> Tensor:
        output = F.relu(self.dec_layer1(z))
        output = F.relu(self.dec_layer2(output))
        return output

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, log_var = self.encoder(x)
        z = self.sample_z(mu, log_var)
        output = self.decoder(z)
        return output, z, mu, log_var


def loss_function(output: Tensor, x: Tensor, mu: Tensor, log_var: Tensor) -> Tensor:
    """The loss function for our VAE consists of two terms:
    - The first term penalises reconstruction error (can be thought of maximizing
        the reconstruction likelihood).
    - The second term encourages our learned distribution q(z|x) to be similar to
        the true prior distribution p(z|x). We assume that q(z|x) follows a
        standard normal distribution N(0, 1) for each dimension of the latent space.
    """
    batch_size = x.shape[0]
    recon_loss = F.mse_loss(output, x, reduction="sum") / batch_size

    # To see how the KL divergence loss is derived, see:
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    kl_div_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + 0.002 * kl_div_loss

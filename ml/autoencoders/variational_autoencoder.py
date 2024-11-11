"""Variational Autoencoder (VAE) implementation.

References:
    - https://www.jeremyjordan.me/variational-autoencoders/
    - https://mbernste.github.io/posts/vae/
    - https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
    - https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.tensor import Tensor


class VAE(nn.Module):
    """Minimal implementation of a Variational Autoencoder (VAE) whose encoder and
     decoder are fully connected layers.

    VAEs can be viewed as probabilistic autoencoders: instead of mapping each x
    directly to z, the VAE maps x to a distribution over z from which z is sampled.
    The crucial thing to remember about a standard autoencoder is that its encoder
    outputs a single value for each latent dimension. The decoder network then
    subsequently takes these values and attempts to reconstruct the original input.
    Using a VAE however, we can describe latent attributes probabilistically.
    Specifically, each latent attribute for a given input is represented by a
    probability distribution. When decoding from the latent state, we will randomly
    sample from each latent state distribution to generate a vector as input for the
    decoder.

    The probabilistic nature of VAEs makes them ideal for generative modelling: their
    latent spaces are continuous by design, which allows easy random sampling and
    interpolation. Thus, VAEs can learn smooth latent state representations of the
    input data. For example, training a standard autoencoder on MNIST typically results
    in distinct clusters in the (say) 2D latent space, with each cluster representing
    a digit. While this clustering helps the decoder accurately reconstruct images, it
    creates gaps in the latent space. In generative models, these gaps are problematic
    because sampling from them can result in unrealistic outputs, as the decoder has
    no training data from those regions to guide generation.

    Note on nomenclature: for VAEs,
        - The encoder model is sometimes referred to as the recognition model.
        - The decoder model is sometimes referred to as the generative model.
    """

    def __init__(self, d_x: int, d_hidden: int, d_z: int = 10):
        super(VAE, self).__init__()

        # Encoder layers
        self.enc_layer1 = nn.Linear(d_x, d_hidden)
        self.enc_layer2_mu = nn.Linear(d_hidden, d_z)
        self.enc_layer2_log_var = nn.Linear(d_hidden, d_z)

        # Decoder layers
        self.dec_layer1 = nn.Linear(d_z, d_hidden)
        self.dec_layer2 = nn.Linear(d_hidden, d_x)

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
        """Sample an encoding from the latent space distribution.

        By sampling encodings from the latent space distribution, the decoder will
        learn to recognise that nearby points in latent space correspond to similar
        data. This continuity enables the decoder to generalize and decode slight
        variations in the latent space, ultimately making the VAE a better generative
        model.

        NOTE: the sampling process requires special attention. When training the model,
        we cannot backpropagate through a random sampling process. To get around this,
        we use the reparametrization trick: now, the randomness instead comes from
        sampling another tensor `epsilon` which isn't part of the computational graph.

        Args:
            mu: vector of means, one for each latent dimension.
            log_var: vector of log-variances, one for each latent dimension.

        Returns:
            z: sampled encoding from the latent space distribution.
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


def loss_fn_vae(output: Tensor, x: Tensor, mu: Tensor, log_var: Tensor) -> Tensor:
    """Loss function for a VAE.

    Consists of two terms:
    - Reconstruction error: this term penalises reconstruction error.
    - KL divergence: this term encourages our learned distribution q(z|x) to be similar
        to the true prior distribution p(z|x). We assume that q(z|x) follows a standard
        normal distribution N(0, 1) for each latent space dimension. We can think of
        the KL-term as a regularization term on the reconstruction loss. That is, the
        model seeks to reconstruct each sample x, however it also seeks to ensure that
        the latent z follows a normal distribution! Intuitively, the KL divergence
        encourages the encoder to distribute encodings (for all types of inputs, e.g.
        all MNIST numbers) around the center of the latent space distribution, N(0, I).
        This is great: when randomly generating, we can sample a vector from N(0, I)
        and the decoder will be able to decode it. And for interpolation, there are no
        sudden gaps between clusters, but a smooth mix of features that a decoder can
        understand.
    """
    batch_size = x.shape[0]
    recon_loss = F.mse_loss(output, x, reduction="sum") / batch_size

    # To see how the KL divergence loss is derived, see:
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    kl_div_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + 0.002 * kl_div_loss

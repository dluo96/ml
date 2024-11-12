import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.diffusion.ddpm.dataset import create_datasets, show_tensor_image
from ml.diffusion.ddpm.noise_schedule import linear_beta_schedule
from ml.diffusion.ddpm.utils import get_tensor_value_at_index
from ml.tensor import Tensor


class DDPM:
    """Class for representing the forward (noising) process and backward (denoising)
    process in a denoising diffusion probabilistic model (DDPM). This implementation
    is based on the original DDPM paper: https://arxiv.org/pdf/2006.11239.pdf.

    Forward process, noise/variance schedule, and sampling: we must first create the
    inputs for our network, which are noisy versions of our original images. Rather
    than doing this sequentially, we can use the closed form in Equation 4 of the
    paper to create a noised image x_t given the original image x_0 and a timestep t
    in {1, ..., T}. Importantly,
    - The noise-levels can be pre-computed.
    - There are different types of variance schedules.
    - We can directly sample x_t from x_0 and t (Equation 4 in paper). This follows
      from the reparametrization trick and the fact that the sum of independent
      Gaussian distributions is also Gaussian.
    - No ML model/network is needed in the forward/noising process.
    """

    def __init__(self, T: int) -> None:
        """Initialize the DDPM model.

        Args:
            T: number of time steps.
        """
        self.T = T

        # Create a linear beta schedule which defines the noise schedule
        # and precompute useful quantities.
        self.betas = linear_beta_schedule(T, 0.001, 0.01)
        self.alphas = 1.0 - self.betas
        self.alphabars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphabars = torch.sqrt(self.alphabars)
        self.sqrt_one_minus_alphabars = torch.sqrt(1.0 - self.alphabars)
        self.alphabars_prev = torch.nn.functional.pad(
            input=self.alphabars[:-1], pad=(1, 0), value=1.0
        )  # Remove the last value and then add 1.0 as the first value (via padding)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphabars_prev) / (1.0 - self.alphabars)
        )

    def create_noised_image(
        self, x_0: Tensor, t: Tensor, device: str = "cpu"
    ) -> tuple[Tensor, Tensor]:
        """Create a noised version of an uncorrupted image. Used in the forward process.

        Specifically, given an uncorrupted image at t=0 and timestep `t`, this method
        returns a noisy version of the image. The level of noise added depends on the
        timestep `t`.

        Note: because the noise added from t-1 to t follows a Gaussian distribution, a
        convenient property follows: one can directly sample the noised image `x_t` at
        any timestep `t` from a distribution q(x_t|x_0) with closed form. To this end,
        this method implements Equation (4) in the original DDPM paper.

        Args:
            x_0: the original image, i.e. the image at t=0. Shape (B, C, H, W).
            t: timestep to sample. Can take values in {1, ..., T}. Shape (B,).
            device: device on which to place the resulting noised image.

        Returns:
            A tuple containing:
            - `x_t` (a noisy version of the input image). Shape (B, C, H, W).
            - The noise that was added to the input image to get `x_t`.
              Shape (B, C, H, W).
        """
        # Compute factor of x_0 in normal distribution in Equation 4 of DDPM paper
        sqrt_alphabar_t = get_tensor_value_at_index(self.sqrt_alphabars, t, x_0.shape)

        # Compute standard deviation of normal distribution in Equation 4 of DDPM paper
        sqrt_one_minus_alphabar_t = get_tensor_value_at_index(
            self.sqrt_one_minus_alphabars, t, x_0.shape
        )

        # Sample epsilon from the standard normal distribution (Algorithm 1 line 4)
        noise = torch.randn_like(x_0)

        # Sample from the normal distribution (Equation 4 in the DDPM paper)
        # using the reparametrization trick: x = μ + sigma * ε
        mean = sqrt_alphabar_t.to(device) * x_0.to(device)
        std = sqrt_one_minus_alphabar_t.to(device)
        noised_image = mean + std * noise.to(device)

        return noised_image, noise.to(device)

    @torch.no_grad()
    def perform_denoising_step(
        self, x_t: Tensor, t: Tensor, model: nn.Module
    ) -> Tensor:
        """Denoise a noisy image to get a slightly less noisy image.

        The following steps are done:
        - Call U-net model to predict the total noise in the noised input image `x_t`.
        - Subtract this from `x_t` to give an estimate of the final image `x_0`.
        - If not in the final timestep, we apply noise (but less than previously) which
          results in x_{t-1} which is (hopefully) a slight improvement over `x_t`.

        This is used in Algorithm 2 in the original DDPM paper: it is the successive
        application of this method (starting from pure noise at t=T) that results in a
        final generated/denoised image.

        Args:
            x_t: the noisy image at timestep `t`. Shape (B, C, H, W).
            t: the timestep in question. Shape (B,).
            model: the U-net model used to predict the "total" noise in the noisy image.

        Returns:
            - x_{t-1} if t > 0. This a noised version of the final image. It can be
              thought of as the result of applying a denoising step on `x_t`.
              Shape (B, C, H, W).
            - x_0 if t = 0. This is the final generated image. Shape (B, C, H, W).
        """
        beta_t = get_tensor_value_at_index(self.betas, t, x_t.shape)
        sqrt_one_minus_alphabar_t = get_tensor_value_at_index(
            self.sqrt_one_minus_alphabars, t, x_t.shape
        )
        sqrt_recip_alpha_t = get_tensor_value_at_index(
            self.sqrt_recip_alphas, t, x_t.shape
        )

        # Calculate µ_θ in Equation 11 of the DDPM paper, where the noise is predicted
        # using the U-net model (represented as ε_θ(x_t, t) in the paper).
        model_mean = sqrt_recip_alpha_t * (
            x_t - beta_t * model(x_t, t) / sqrt_one_minus_alphabar_t
        )

        # Compute (sigma_t)^2 in Algorithm 2 line 4
        posterior_variance_t = get_tensor_value_at_index(
            self.posterior_variance, t, x_t.shape
        )

        if t == 0:
            x_0 = model_mean
            # As pointed out by Luis Pereira (see YouTube comment)
            # The t's are offset from the t's in the paper
            return x_0

        # Sample z (in Algorithm 2) from a standard normal distribution
        noise = torch.randn_like(x_t)

        # Return x_{t-1} in Algorithm 2 line 4. This is basically
        # sampling from the distribution q(x_{t-1}|x_t) in Equation 4.
        return model_mean + torch.sqrt(posterior_variance_t) * noise


if __name__ == "__main__":
    # Prepare dataset
    train_dataset, test_dataset = create_datasets()
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)

    # Initialize DDPM model
    T_ = 100
    ddpm = DDPM(T=T_)

    # Simulate forward diffusion for an example image
    example_image = next(iter(dataloader))[0]

    plt.figure(figsize=(15, 15))
    plt.axis("off")
    num_images = 10
    step_size = int(T_ / num_images)

    # Plot the sequence of progressively noiser images for the example image
    for idx in range(0, T_, step_size):
        t_ = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images + 1, int(idx / step_size) + 1)
        noised_img, _ = ddpm.create_noised_image(x_0=example_image, t=t_)
        show_tensor_image(noised_img)

    plt.show()

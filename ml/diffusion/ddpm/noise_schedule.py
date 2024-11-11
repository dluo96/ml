import torch

from ml.tensor import Tensor


def linear_beta_schedule(
    T: int, beta_start: float = 1e-4, beta_final: float = 0.02
) -> Tensor:
    """Return a tensor of linearly spaced beta values, defining the noise schedule
    (aka variance schedule).

    Args:
        T: number of time steps.
        beta_start: initial value of beta from t=0 to t=1.
        beta_final: final value of beta from timestep t=T-1 to t=T.

    Returns:
        1D tensor of shape (T,) representing linearly spaced beta values used in the
        noise schedule of the forward process.
    """
    return torch.linspace(beta_start, beta_final, T)

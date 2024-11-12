import torch

from ml.tensor import Tensor


def linear_beta_schedule(
    T: int, beta_start: float, beta_final: float, device: str
) -> Tensor:
    """Return a tensor of linearly spaced beta values, defining the noise schedule
    (aka variance schedule).

    Args:
        T: number of time steps.
        beta_start: initial value of beta from t=0 to t=1.
        beta_final: final value of beta from timestep t=T-1 to t=T.
        device: device on which to create the resulting tensor.

    Returns:
        1D tensor of shape (T,) representing linearly spaced beta values used in the
        noise schedule of the forward process.
    """
    return torch.linspace(beta_start, beta_final, T, device=device)

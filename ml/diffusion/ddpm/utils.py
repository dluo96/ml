"""Utility functions for the DDPM model."""
import torch


def get_tensor_value_at_index(
    tensor: torch.Tensor, t: torch.Tensor, x_shape: tuple[int]
) -> torch.Tensor:
    """Returns the value at index `t` of the input `tensor`, all while
    considering the batch dimension.

    Args:
        tensor: the tensor of values. Shape (B,). This could for example be
            the β_1, ..., β_T values which define the noise schedule.
        t: the index from which to get the desired value from `tensor`.
        x_shape: shape of image, usually (B, C, H, W).

    Returns
        Tensor containing the value of the input `tensor` at the specified
            index `t`. If `x_shape` is (B, C, H, W), the output tensor has
            shape (1, 1, 1, 1).

    Example:
        >>> betas = torch.linspace(0.001, 0.02, 100)
        >>> t = torch.Tensor([50]).type(torch.int64)
        >>> x_shape = torch.randn(size=(8, 3, 28, 28)).shape
        >>> beta_t = get_tensor_value_at_index(betas, t, x_shape)
        >>> beta_t
        tensor([[[[0.0080]]]])
        >>> beta_t.shape
        (1, 1, 1, 1)
    """
    batch_size = t.shape[0]

    # Get the value of `tensor` at index `t`
    out = tensor.gather(dim=-1, index=t.cpu())

    # Reshape to (1, 1, 1, 1) i.e. same rank as the image.
    non_batch_shape = (1,) * (len(x_shape) - 1)
    out = out.reshape(batch_size, *non_batch_shape).to(t.device)

    return out

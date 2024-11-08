import torch


class LayerNorm:
    """The key difference with BatchNorm is that LayerNorm computes mean and variance
    across the feature dimension (dim=1) instead of the batch dimension (dim=0).
    Because the calculation of mean and variance is not across examples, we don't need
    a running mean and variance. It also follows that there is no distinction between
    training and evaluation.

    With batch normalization, there was coupling between examples in the batch. This
    turned out to be good for training because it regularises the network. However,
    some ML practitioners decided to get around this by introducing other normalization
    methods such as LayerNorm, InstanceNorm, and GroupNorm.
    """

    def __init__(self, n_embd: int, eps: float = 1e-5):
        self.eps = eps

        # Parameters (trained with backpropagation)
        self.gamma = torch.ones(n_embd)  # Determines scale
        self.beta = torch.zeros(n_embd)  # Determines shift

    def parameters(self) -> list[torch.Tensor]:
        return [self.gamma, self.beta]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Key difference with BatchNorm is that LayerNorm computes mean and variance
        # across the feature dimension (dim=1) instead of the batch dimension (dim=0)
        xmean = x.mean(dim=1, keepdim=True)
        xvar = x.var(dim=1, keepdim=True, unbiased=True)

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        return self.out

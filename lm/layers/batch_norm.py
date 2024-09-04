import torch


class BatchNorm1D:
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Parameters (trained with backpropagation)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # Buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def parameters(self) -> list[torch.Tensor]:
        return [self.gamma, self.beta]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
        if self.training:
            xmean = x.mean(dim=0, keepdim=True)  # Batch mean
            xvar = x.var(dim=0, keepdim=True, unbiased=True)  # Batch variance
        else:
            # Eval or inference
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        # Update the buffers using 'momentum update'
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean  # fmt: skip
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar  # fmt: skip

        return self.out

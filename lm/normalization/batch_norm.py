import torch


class BatchNorm1D:
    """Batch normalization is used to control the statistics of activations in a
    neural network. It helps ensure that the pre-activation values are not too spread
    out or too close together. This can help with training stability because it can
    prevent the gradients from becoming too small or too large. It is common to put it
    after layers with a multiplication (such as a linear layer or a convolutional
    layer) and prior to activation functions.

    Consider the case where the activation function is tanh(x). If x is too spread out,
    tanh(x) will saturate at -1 or +1, causing the gradients to vanish (recall that
    d/dx tanh(x) = 1 - tanh^2(x)). This can lead to dead neurons that do not learn.
    The same reasoning applies to other activation functions including ReLU and sigmoid.

    Before batch normalization, ML practitioners avoided the saturation problem by
    initialising the neural network weights carefully such as using Kaiming/He or
    Xavier initialisation.

    The batch norm layer computes the mean and variance of the pre-activations feeding
    into it, across the batch. It has trainable (via backpropagation) parameters gamma
    and beta for scaling and shifting, respectively. This allows the distribution of
    the pre-activations (output of batch norm layer) to deviate from the standard
    normal distribution (mean 0 and variance 1) if necessary.

    The batch norm layer also keeps a running mean and a running variance, which are
    used during evaluation/inference. In particular, it allows us to forward individual
    examples at inference time.
    """

    def __init__(self, n_embd: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Parameters (trained with backpropagation)
        self.gamma = torch.ones(n_embd)  # Determines scale
        self.beta = torch.zeros(n_embd)  # Determines shift

        # Buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(n_embd)
        self.running_var = torch.ones(n_embd)

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

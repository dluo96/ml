from ml.tensor import Tensor


class AdamOptimizer:
    """Minimal implementation of Adam (Adaptive Moment Estimation).

    Adam combines (gradient descent with) Momentum and (gradient descent with) RMSProp:
        - Momentum: uses an exponentially weighted moving average of gradients.
        - RMSProp: uses an exponentially weighted moving average of squared gradients.

    NOTE: this implementation does NOT have
        - Weight decay.
        - Learning rate decay.
    """

    def __init__(
        self,
        lr: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """Initialize the Adam optimizer.

        Args:
            lr: learning rate.
            beta_1: exponential decay rate for the first moment estimates.
            beta_2: exponential decay rate for the second moment estimates.
            epsilon: small value to prevent division by zero.
        """
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.lr = lr

        # Momentum: initialize the (exponentially weighted) moving average of the
        # gradient `dw`.
        self.m_dw = 0

        # RMSProp: initialize the (exponentially weighted) moving average of the
        # squared gradient `dw**2`.
        self.v_dw = 0

    def update(self, t: int, w: Tensor, dw: Tensor) -> Tensor:
        """Update the weights and biases using the Adam optimizer.

        - Momentum update:
            m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t

        - RMSProp update:
            v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2

        - Bias correction:
            m_t_corr = m_t / (1 - beta_1^t)
            v_t_corr = v_t / (1 - beta_2^t)

            This improves the accuracy of initial exponentially weighted averages
            which are typically very small if the moving averages are initialized to 0.

        - Parameter update:
            w_t = w_{t-1} - lr / (sqrt(v_t_corr) + epsilon) * m_t_corr

        Args:
            t: current iteration.
            w: weights.
            dw: gradients of `w` (wrt. the loss function) in the current batch.
        """
        # Momentum: update moving average of gradients
        self.m_dw = self.beta_1 * self.m_dw + (1 - self.beta_1) * dw

        # RMSProp: update moving average of squared gradients
        self.v_dw = self.beta_2 * self.v_dw + (1 - self.beta_2) * dw**2

        # Bias corrections
        m_dw_corr = self.m_dw / (1 - self.beta_1**t)
        v_dw_corr = self.v_dw / (1 - self.beta_2**t)

        # Parameter update
        w = w - self.lr * m_dw_corr / (v_dw_corr.sqrt() + self.epsilon)

        return w

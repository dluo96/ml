from lm.tensor import Tensor


class AdamOptimizer:
    def __init__(
        self,
        lr: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """Initialize the Adam optimizer. Adam stands for Adaptive Moment Estimation.

        Adam combines momentum and RMSProp. As such, it relies on two moments:
            - First-order moment estimate of the mean,
            - Second-order moment estimate of the variance.

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

        # Initialize the (exponentially weighted) moving averages of the gradients
        # `dw` and `db`. These serve as first moment estimates.
        self.m_dw = 0
        self.m_db = 0

        # Initialize the (exponentially weighted) moving averages of the squared
        # gradients, `dw**2` and `db**2`. These serve as second moment estimates.
        self.v_dw = 0
        self.v_db = 0

    def update(
        self, t: int, w: Tensor, dw: Tensor, b: Tensor, db: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Update the weights and biases using the Adam optimizer.

        First, do the momentum-like update (exponentially weighted average):

            m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t

        Then, do the RMSProp-like update:

            v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2

        Since most algorithms that depend on moving averages (such as SGD and RMSProp)
        are biased, we need a bias correction step:

            m_t_hat = m_t / (1 - beta_1^t)
            v_t_hat = v_t / (1 - beta_2^t)

        Finally, we update the weights and biases:

            w_t = w_{t-1} - eta / (sqrt(v_t_hat) + epsilon) * m_t_hat

        Args:
            t: current iteration.
            w: weights.
            dw: gradients of `w` in the current batch.
            b: biases.
            db: gradients of `b` in the current batch.
        """
        # Update moving averages of gradients
        self.m_dw = self.beta_1 * self.m_dw + (1 - self.beta_1) * dw
        self.m_db = self.beta_1 * self.m_db + (1 - self.beta_1) * db

        # Update moving averages of squared gradients
        self.v_dw = self.beta_2 * self.v_dw + (1 - self.beta_2) * dw**2
        self.v_db = self.beta_2 * self.v_db + (1 - self.beta_2) * db**2

        # Bias correction
        m_dw_hat = self.m_dw / (1 - self.beta_1**t)
        m_db_hat = self.m_db / (1 - self.beta_1**t)
        v_dw_hat = self.v_dw / (1 - self.beta_2**t)
        v_db_hat = self.v_db / (1 - self.beta_2**t)

        # Update weights and biases
        w = w - self.lr * m_dw_hat / (v_dw_hat.sqrt() + self.epsilon)
        b = b - self.lr * m_db_hat / (v_db_hat.sqrt() + self.epsilon)

        return w, b

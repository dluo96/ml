import torch
import torch.nn.functional as F


class MLP:
    """Character-level MLP model."""

    def __init__(self, words: list[str]):
        self.words = words

        # Build vocabulary of characters and mappings to/from indices
        unique_chars = ["."] + sorted(list(set("".join(words))))
        self.n_unique_chars = len(unique_chars)
        self.ctoi = {c: i for i, c in enumerate(unique_chars)}
        self.itoc = {i: c for i, c in enumerate(unique_chars)}

    def create_dataset(self, words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        # Context length: how many characters do we take to predict the next one
        block_size = 3

        # X contains inputs to neural net, Y contains labels
        X, Y = [], []
        for w in words:
            context = [0] * block_size  # Begin with `block_size` start tokens "."
            for ch in w + ".":
                ix = self.ctoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]  # Crop and append

        X = torch.tensor(X)
        Y = torch.tensor(Y)

        return X, Y

    def train(self):
        """The lookup table can be interpreted in two equivalents way:
        1. Indexing the character index into the lookup table.
        2. The first layer of the neural net - this layer doesn't have any
            non-linearity and the weight matrix is simply the lookup table.
            This interpretation would require each input character (index)
            to be one-hot encoded before being multiplied by the lookup table.
        """
        X, Y = self.create_dataset(self.words)
        g = torch.Generator().manual_seed(2147483647)
        C = torch.randn((self.n_unique_chars, 2), generator=g)
        W1 = torch.randn((6, 100))
        b1 = torch.randn(100)
        W2 = torch.randn((100, self.n_unique_chars))
        b2 = torch.randn(self.n_unique_chars)
        parameters = [C, W1, b1, W2, b2]

        for p in parameters:
            p.requires_grad = True

        for _ in range(20000):
            # Batching
            ix = torch.randint(0, X.shape[0], (32,))

            emb = C[X[ix]]  # Shape: (n_samples, block_size, 2)

            # In order to perform matrix multiplication, we must first concatenate the
            # embeddings along the second dimension. The shape goes from
            # (n_samples, block_size, 2) to (n_samples, 2 * block_size)
            emb = emb.view(emb.shape[0], -1)

            h = torch.tanh(emb @ W1 + b1)

            logits = h @ W2 + b2  # Shape: (n_samples, n_unique_chars)

            loss = F.cross_entropy(logits, Y[ix])

            # Zero gradients and backpropagate
            for p in parameters:
                p.grad = None
            loss.backward()
            for p in parameters:
                p.data -= 0.01 * p.grad

        print(loss.item())

        block_size = 3
        g = torch.Generator().manual_seed(2147483647 + 1)
        for _ in range(20):
            out = []
            context = [0] * block_size
            while True:
                emb = C[torch.tensor(context)]  # Shape (1, block_size, d)
                h = torch.tanh(emb.view(1, -1) @ W1 + b1)
                logits = h @ W2 + b2
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1, generator=g).item()
                context = context[1:] + [ix]
                out.append(self.itoc[ix])
                if ix == 0:
                    break

            print("".join(out))

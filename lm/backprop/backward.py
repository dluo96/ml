import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lm.datasets import MultiCharDataset


def compare(name: str, dt: torch.Tensor, t: torch.Tensor):
    exact = torch.all(dt == t.grad).item()
    approx = torch.allclose(dt, t.grad, atol=1e-7)  # In case of floating point precision
    max_diff = (dt - t.grad).abs().max().item()
    print(f"{name:15s}: Exact={str(exact):5s} | Approx={str(approx)} | Max diff={max_diff}")
    return exact or approx


def test_manual_backward():
    # Prepare data
    words = [
        "emma",
        "isabella",
        "camila",
        "sadie",
        "faith",
        "margaret",
        "jasmine",
        "kayla",
        "morgan",
        "parker",
        "jacqueline",
        "veronica",
        "winter",
        "alexia",
        "itzel",
    ]
    dataset = MultiCharDataset(words, block_size=3)
    vocab_size = dataset.get_vocab_size()
    block_size = dataset.get_output_length()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    n_embd = 10
    n_hidden = 64
    g = torch.Generator("cpu")
    C = torch.randn((vocab_size, n_embd), generator=g)

    # Layer 1
    W1 = torch.randn(
        (n_embd * block_size, n_hidden), generator=g
    ) * (5/3)/((n_embd * block_size) ** 0.5)
    b1 = torch.randn((n_hidden,), generator=g) * 0.1

    # Layer 2
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
    b2 = torch.randn((vocab_size,), generator=g) * 0.1

    # Batch norm parameters
    bngain = torch.randn((1, n_hidden), generator=g) * 0.1 + 1.0
    bnbias = torch.randn((1, n_hidden), generator=g) * 0.1

    parameters = [C, W1, b1, W2, b2, bngain, bnbias]
    for p in parameters:
        p.requires_grad = True

    B = 32  # batch size
    x_batch, y_batch = next(iter(dataloader))  # Get a batch of data

    # Forward pass
    emb = C[x_batch]  # Embedding layer. (B, block_size, n_embd)
    embcat = emb.view(B, -1)  # Concatenation. (B, block_size * n_embd)
    hprebn = embcat @ W1 + b1  # # First linear layer. (B, n_hidden)
    bnmeani = 1/B * hprebn.sum(dim=0, keepdim=True)  # Batch norm. (1, n_hidden)
    bndiff = hprebn - bnmeani  # (B, n_hidden)
    bndiff2 = bndiff ** 2  # (B, n_hidden)
    bnvar = 1/(B-1) * bndiff2.sum(dim=0, keepdim=True)  # (1, n_hidden)  Bessel's correction, dividing by B-1 instead of B
    bnvar_inv = (bnvar + 1e-5).rsqrt()  # (1, n_hidden)
    bnraw = bndiff * bnvar_inv  # (B, n_hidden)
    hpreact = bngain * bnraw + bnbias  # (B, n_hidden)
    # Linear layer 2
    h = torch.tanh(hpreact)  # (B, n_hidden)
    logits = h @ W2 + b2  # (B, vocab_size)

    # Cross entropy loss
    logit_maxes = logits.max(dim=1, keepdim=True).values
    norm_logits = logits - logit_maxes  # Subtracting the max value for numerical stability
    counts = norm_logits.exp()
    counts_sum = counts.sum(dim=1, keepdim=True)
    counts_sum_inv = counts_sum ** -1
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    loss = -logprobs[torch.arange(B), y_batch].mean()

    # PyTorch backwards pass
    for p in parameters:
        p.grad = None
    for t in [
        logprobs, probs, counts_sum_inv, counts_sum, counts, norm_logits, logit_maxes,
        logits, h, hpreact, bnraw, bnvar_inv, bnvar, bndiff2, bndiff, bnmeani, hprebn,
        embcat, emb
    ]:
        t.retain_grad()
    loss.backward()

    # Manual backwards pass
    # loss = -logprobs[torch.arange(B), y_batch].mean()
    dlogprobs = torch.zeros_like(logprobs)
    dlogprobs[torch.arange(B), y_batch] = -1.0/B
    assert compare("dlogprobs", dlogprobs, logprobs)

    dprobs = 1.0/probs * dlogprobs  # Chain rule and d/dx ln(x) = 1/x
    assert compare("dprobs", dprobs, probs)

    # In `probs = counts * counts_sum_inv`, `counts` has shape (B, vocab_size) and
    # `counts_sum_inv` has shape (B, 1), meaning `counts_sum_inv` is broadcasted to
    # shape (B, vocab_size).
    #
    #   a11*b1 a12*b1 a13*b1 = c11 c12 c13
    #   a21*b2 a22*b2 a23*b2 = c21 c22 c23
    #   a31*b3 a32*b3 a33*b3 = c31 c32 c33
    #
    # The broadcasting means the variable (b1, b2, b3) is effectively used multiple
    # times (3 in the example above). Thus, we need to accumulate the gradients every
    # time we use it. This is also what PyTorch does during the backward pass: the
    # gradient at each 'node' is summed across all incoming edges.
    dcounts_sum_inv = (dprobs * counts).sum(dim=1, keepdim=True)  # (B, 1)
    assert compare("dcounts_sum_inv", dcounts_sum_inv, counts_sum_inv)

    dcounts = dprobs * counts_sum_inv  # Broadcasting occurs: (B, V) x (B, 1) -> (B, V)
    # Can't test this yet since counts is used twice.

    dcounts_sum = (-counts_sum ** -2) * dcounts_sum_inv
    assert compare("dcounts_sum", dcounts_sum, counts_sum)

    dcounts_other = torch.ones_like(counts) * dcounts_sum
    dcounts += dcounts_other
    assert compare("dcounts", dcounts, counts)

    dnorm_logits = norm_logits.exp() * dcounts
    assert compare("dnorm_logits", dnorm_logits, norm_logits)

    dlogits = dnorm_logits * 1.0
    # Can't check dlogits yet because there is another edge leading to it from logit_maxes

    # Sum needed bc of broadcasting:
    #       c11 c12 c13 = a11 a12 a13   b1
    #       c21 c22 c23 = a21 a22 a23 - b2
    #       c31 c32 c33 = a31 a32 a33   b3
    # For example, c32 = a32 - b2.
    # (b1, b2, b3) is broadcasted across columns: thus, in the backward pass, because
    # we keep reusing it, these are all separate branches of use of that one variable,
    # and so we need to sum the gradients across the columns.
    dlogit_maxes = (dnorm_logits * -1.0).sum(dim=1, keepdim=True)
    assert compare("dlogits_maxes", dlogit_maxes, logit_maxes)

    # In particular, the dlogit_maxes should be close to 0: recall that
    # subtracting it from the logits does not affect probs and therefore does not
    # affect the loss.
    assert torch.allclose(dlogit_maxes, torch.zeros_like(dlogit_maxes))

    # Second incoming edge for logits
    # logit_maxes = logits.max(dim=1, keepdim=True).values
    dlm_dl = torch.zeros_like(logits)
    indices = logits.max(dim=1, keepdim=True).indices
    dlm_dl[torch.arange(B), indices] = 1.0
    dlogits_2nd = dlogit_maxes * dlm_dl
    dlogits += dlogits_2nd
    assert compare("dlogits", dlogits, logits)

    # Note also that this contribution should be zero since dlogit_maxes is 0
    assert torch.allclose(dlogits_2nd, torch.zeros_like(dlogits_2nd))

    # To understand how backpropagation works on a matrix multiplication, consider the
    # small example f = a @ b + c
    #       f11 = a11 * b11 + a12 * b21 + c1
    #       f12 = a11 * b12 + a12 * b22 + c2
    #       f21 = a21 * b11 + a22 * b21 + c1
    #       f22 = a21 * b12 + a22 * b22 + c2
    #
    # Using the chain rule, we have:
    #       dLda11 = dLdf11 * b11 + dLdf12 * b12
    #       dLda12 = dLdf11 * b21 + dLdf12 * b22
    #       dLda21 = dLdf21 * b11 + dLdf22 * b12
    #       dLda22 = dLdf21 * b21 + dLdf22 * b22
    #
    # The addition is because we have to sum contributions.
    # Clearly, dLda = dLdf @ b^T, so the derivative of a matmul is another matmul.
    # In our case (logits = h @ W2 + b2), dLda is dh, dLdf is dlogits, and b is W2.
    dh = dlogits @ W2.T
    assert compare("dh", dh, h)

    # Doing something similar for b gives: dLdb = a^T @ dLdf
    # In our case, dLdb is dW2, a is h, and dLdf is dlogits
    dW2 = h.T @ dlogits
    assert compare("dW2", dW2, W2)

    # Now for the bias:
    #       dL/dc1 = dLdf11 * 1 + dLdf21 * 1
    #       dL/dc2 = dLdf12 * 1 + dLdf22 * 1
    # This is summing across columns, so: dLdc = dLdf.sum(dim=1)
    # In our case, dLdc is dLdb2 and dLdf is dlogits
    db2 = (dlogits * 1.0).sum(dim=0, keepdim=True)
    assert compare("db2", db2, b2)

    # h = torch.tanh(hpreact)
    # dL/dhpreact = dL/dh * dh/dhpreact = dL/dh * sech^2(hpreact)
    # = dL/dh * (1 - tanh^2(hpreact)) = dL/dh * (1-h^2)
    dhpreact = dh * (1 - h ** 2)
    assert compare("dhpreact", dhpreact, hpreact)

    # hpreact = bngain * bnraw + bnbias
    # bngain has shape (1, 64), bnraw has shape (32, 64), bnbias has shape (1, 64)
    # Thus, bngain and bnbias are broadcasted, and we need to do a sum because they
    # are effectively used multiple times.
    dbngain = (dhpreact * bnraw).sum(dim=0, keepdim=True)
    assert compare("dbngain", dbngain, bngain)
    dbnraw = dhpreact * bngain
    assert compare("dbnraw", dbnraw, bnraw)
    dbnbias = (dhpreact * 1.0).sum(dim=0, keepdim=True)
    assert compare("dbnbias", dbnbias, bnbias)

    # bnraw = bndiff * bnvar_inv
    # bndiff has shape (32, 64), bnvar_inv has shape (1, 64)
    # Thus, bnvar_inv is broadcasted and we need to do a sum
    dbndiff = dbnraw * bnvar_inv  # Can't check yet since it is part of another branch
    # assert compare("dbndiff", dbndiff, bndiff)
    dbnvar_inv = (dbnraw * bndiff).sum(dim=0, keepdim=True)
    assert compare("dbnvar_inv", dbnvar_inv, bnvar_inv)

    # bnvar_inv = (bnvar + 1e-5).rsqrt()
    # d/dx x^(-1/2) = -1/2 * x^(-3/2)
    dbnvar = dbnvar_inv * (-1/2 * (bnvar + 1e-5) ** (-3/2))
    assert compare("dbnvar", dbnvar, bnvar)

    # bnvar = 1/(B-1) * bndiff2.sum(dim=0, keepdim=True)
    # bnvar has shape (1, 64), bndiff2 has shape (32, 64)
    # This isn't due to broadcasting, so no sum is needed.
    # NOTE: notice the duality - when there is a sum over an axis in the forward pass,
    # there is a corresponding broadcasting in the backward pass. Similarly, when there
    # is a broadcast operation in the forward pass, this indicates a variable reuse and
    # in the backward pass, this turns into a sum over the same dimension.
    #
    # Small example:
    #       a11 a12
    #       a21 a22
    # Gives
    #       b1 b2
    # where
    #       b1 = a11 + a21
    #       b2 = a12 + a22
    #
    # Broadcasting: (1, 64) * (32, 64) -> (32, 64)
    dbndiff2 = dbnvar * torch.ones_like(bndiff2) * 1.0/(B-1.0)
    assert compare("dbndiff2", dbndiff2, bndiff2)

    # bndiff2 = bndiff ** 2
    # NOTE: need to accumulate
    dbndiff += dbndiff2 * (2 * bndiff)
    assert compare("dbndiff", dbndiff, bndiff)

    # bndiff = hprebn - bnmeani
    # hprebn has shape (32, 64)
    # bnmeani has shape (1, 64)
    # Thus, bnmeani is broadcasted and we need to sum
    dhprebn = dbndiff * 1.0  # Can't check yet
    dbnmeani = (dbndiff * -1.0).sum(dim=0, keepdim=True)
    assert compare("dbnmeani", dbnmeani, bnmeani)

    # bnmeani = 1 / B * hprebn.sum(dim=0, keepdim=True)
    # Sum in forward pass --> broadcast in backward pass
    # Broadcasting: (1, 64) * (32, 64) -> (32, 64)
    dhprebn += dbnmeani * (1/B * torch.ones_like(hprebn))
    assert compare("dhprebn", dhprebn, hprebn)

    # hprebn = embcat @ W1 + b1
    # embcat has shape (32, 30)
    # W1 has shape (30, 64)
    # b1 has shape (64,)
    dembcat = dhprebn @ W1.T
    assert compare("dembcat", dembcat, embcat)
    dW1 = embcat.T @ dhprebn
    assert compare("dW1", dW1, W1)
    db1 = (dhprebn * 1.0).sum(dim=0, keepdim=True)
    assert compare("db1", db1, b1)

    # embcat = emb.view(B, -1)
    # embcat has shape (B, block_size * n_embd)
    # emb has shape (B, block_size, n_embd)
    # dL/demb = dL/dembcat * dembcat/demb
    demb = dembcat.view(emb.shape)
    assert compare("demb", demb, emb)

    # emb = C[x_batch]
    # C has shape (vocab_size, n_embd)
    # x_batch has shape (B, block_size)
    # dL/dC = dL/demb * demb/dC
    dC = torch.zeros_like(C)
    for k in range(x_batch.shape[0]):
        for j in range(x_batch.shape[1]):
            ix = x_batch[k, j]
            dC[ix] += demb[k, j]  # Addition since a row could be used multiple times
    assert compare("dC", dC, C)

    # Loss with a faster calculation of loss
    loss_fast = F.cross_entropy(logits, y_batch)
    assert torch.allclose(loss_fast, loss)
    # Simpler calculation of gradient of cross entropy
    dlogits = F.softmax(logits, dim=1)
    dlogits[range(B), y_batch] -= 1
    dlogits /= B
    assert compare("dlogits", dlogits, logits)

    # Faster calculation of batch norm
    hpreact_fast = bngain * (hprebn - hprebn.mean(dim=0, keepdim=True)) \
        / torch.sqrt(hprebn.var(dim=0, keepdim=True, unbiased=True) + 1e-5) \
        + bnbias
    assert torch.allclose(hpreact_fast, hpreact)
    # Simpler calculation of gradient of batch norm
    # dL/dhprebn = dL/dhpreact * dhpreact/dhprebn

    dhprebn = bngain * bnvar_inv/B * \
        (B * dhpreact - dhpreact.sum(dim=0) - B/(B-1) * bnraw * (dhpreact * bnraw).sum(dim=0))

    assert compare("dhprebn", dhprebn, hprebn)

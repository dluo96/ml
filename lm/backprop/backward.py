import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lm.model_config import Tensor
from lm.datasets import MultiCharDataset


def compare(name: str, dt: Tensor, t: Tensor) -> bool:
    """Compare the manually calculated gradient (of a tensor) with the gradient (of
    the same tensor) as computed by PyTorch's backward().

    Args:
        name: label for the gradient tensor.
        dt: manually computed gradient tensor
        t: tensor whose gradient is computed by PyTorch's backward().

    Returns:
        Boolean indicating whether the manual computation of the tensor's gradient is
            correct (exactly or approximately).
    """
    exact = torch.all(dt == t.grad).item()
    approx = torch.allclose(dt, t.grad, atol=1e-7)  # Due to floating point precision
    max_diff = (dt - t.grad).abs().max().item()
    print(f"{name:15s}: Exact={str(exact):5s} | Approx={str(approx)} | Max diff={max_diff}")
    return exact or approx


def test_manual_backward():
    # Prepare data
    # fmt: off
    words = [
        "emma", "isabella", "camila", "sadie", "faith", "margaret", "jasmine", "kayla",
        "morgan", "parker", "jacqueline", "veronica", "winter", "alexia", "itzel",
    ]
    # fmt: on
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

    """Forward pass"""
    # Embedding lookup followed by concatenation
    emb = C[x_batch]  # (B, block_size, n_embd)
    embcat = emb.view(B, -1)  # (B, block_size * n_embd)

    # Linear layer
    hprebn = embcat @ W1 + b1  # (B, n_hidden)

    # Batch norm (with Bessel's correction: dividing by B-1 instead of B)
    bnmeani = 1/B * hprebn.sum(dim=0, keepdim=True)  # (1, n_hidden)
    bndiff = hprebn - bnmeani  # (B, n_hidden)
    bndiff2 = bndiff ** 2  # (B, n_hidden)
    bnvar = 1/(B-1) * bndiff2.sum(dim=0, keepdim=True)  # (1, n_hidden)
    bnvar_inv = (bnvar + 1e-5).rsqrt()  # (1, n_hidden)
    bnraw = bndiff * bnvar_inv  # (B, n_hidden)
    hpreact = bngain * bnraw + bnbias  # (B, n_hidden)

    # Linear layer
    h = torch.tanh(hpreact)  # (B, n_hidden)
    logits = h @ W2 + b2  # (B, vocab_size)

    # Cross entropy loss (where max value per row is subtracted for num. stability)
    logit_maxes = logits.max(dim=1, keepdim=True).values  # (B, 1)
    norm_logits = logits - logit_maxes  # (B, vocab_size)
    counts = norm_logits.exp()  # (B, vocab_size)
    counts_sum = counts.sum(dim=1, keepdim=True)  # (B, 1)
    counts_sum_inv = counts_sum ** -1   # (B, 1)
    probs = counts * counts_sum_inv  # (B, vocab_size) * (B, 1) -> (B, vocab_size)
    logprobs = probs.log()  # (B, vocab_size)
    loss = -logprobs[torch.arange(B), y_batch].mean()  # ()

    """Backpropagation with PyTorch to get the true gradients"""
    for p in parameters:
        p.grad = None
    # fmt: off
    for t in [
        logprobs, probs, counts_sum_inv, counts_sum, counts, norm_logits, logit_maxes,
        logits, h, hpreact, bnraw, bnvar_inv, bnvar, bndiff2, bndiff, bnmeani, hprebn,
        embcat, emb
    ]:
        t.retain_grad()
    # fmt: on
    loss.backward()

    """Manual backward pass.
    
    We backpropagate through each step, starting from the end:
    
    `loss = -logprobs[torch.arange(B), y_batch].mean()`
    
    Only the elements at positions (0, y_batch[0]), (1, y_batch[1]), ..., 
    (batch_size-1, y_batch[batch_size-1]) of `logprobs` affect `loss`, hence these are
    also the only elements in dloss/dlogprobs (=dlogprobs) that will have a nonzero 
    gradient. 
    """
    dlogprobs = torch.zeros_like(logprobs)
    dlogprobs[torch.arange(B), y_batch] = -1.0/B
    assert compare("dlogprobs", dlogprobs, logprobs)

    """
    `logprobs = probs.log()`  
    
    `probs` has shape (B, vocab_size).
    
    Use the chain rule and d/dx ln(x) = 1/x
    """
    dprobs = 1.0/probs * dlogprobs
    assert compare("dprobs", dprobs, probs)

    """
    `probs = counts * counts_sum_inv`
    
    `counts` has shape (B, vocab_size),
    `counts_sum_inv` has shape (B, 1).
    
    Thus, `counts_sum_inv` is broadcasted to shape (B, vocab_size) in dimension 1.
    Consider a toy example:

        a11*b1 a12*b1 a13*b1 = c11 c12 c13
        a21*b2 a22*b2 a23*b2 = c21 c22 c23
        a31*b3 a32*b3 a33*b3 = c31 c32 c33

    The broadcasting means that the variable (b1, b2, b3) is effectively used multiple
    times (3 in the example above). Thus, we need to accumulate the gradients every
    time we use it. This is also what PyTorch does during the backward pass: the
    gradient for each variable is summed across its branches.
    """
    # Can't check `dcounts` yet since `counts` is also used in another branch!
    dcounts = dprobs * counts_sum_inv  # Broadcasting occurs: (B, V) x (B, 1) -> (B, V)

    dcounts_sum_inv = (dprobs * counts).sum(dim=1, keepdim=True)  # (B, 1)
    assert compare("dcounts_sum_inv", dcounts_sum_inv, counts_sum_inv)

    """
    `counts_sum_inv = counts_sum ** -1`
    
    `counts_sum` has shape (B, 1). 
    
    d/dx x^(-1) = -1/x^2
    """
    dcounts_sum = (-counts_sum ** -2) * dcounts_sum_inv  # (B, 1)
    assert compare("dcounts_sum", dcounts_sum, counts_sum)

    """
    `counts_sum = counts.sum(dim=1, keepdim=True)`
    
    `counts` has shape (B, V).
    
    Two remarks:
      1. This is the 2nd branch of `counts`, hence we need to accumulate.
      2. Forward pass uses a sum, so backward pass needs to broadcast.
      """
    dcounts += torch.ones_like(counts) * dcounts_sum  # (B, V) * (B, 1) -> (B, V)
    assert compare("dcounts", dcounts, counts)

    """
    `counts = norm_logits.exp()`
    
    `norm_logits` has shape (B, V).
    """
    dnorm_logits = norm_logits.exp() * dcounts
    assert compare("dnorm_logits", dnorm_logits, norm_logits)

    """
    `norm_logits = logits - logit_maxes`
    
    `logits` has shape (B, V), 
    `logit_maxes` has shape (B, 1).
    
    Next, `logit_maxes` is broadcasted. Consider `c = a - b`:
    
        c11 c12 c13 = a11 a12 a13   b1
        c21 c22 c23 = a21 a22 a23 - b2
        c31 c32 c33 = a31 a32 a33   b3
        
    For example, c32 = a32 - b2. So (b1, b2, b3) is broadcasted across columns: thus, 
    in the backward pass, because we keep reusing it, these are all separate branches
    of use of that one variable, so we need to sum the gradients across the columns.
    """
    # Can't check `dlogits` yet because `logits` is used in another branch, namely the
    # one used to compute `logit_maxes`.
    dlogits = dnorm_logits * 1.0

    dlogit_maxes = (dnorm_logits * -1.0).sum(dim=1, keepdim=True)
    assert compare("dlogits_maxes", dlogit_maxes, logit_maxes)
    # In addition, `dlogit_maxes` should be close to 0: recall that subtracting it from
    # the logits does not affect probs and therefore does not affect the loss.
    assert torch.allclose(dlogit_maxes, torch.zeros_like(dlogit_maxes))

    """
    `logit_maxes = logits.max(dim=1, keepdim=True).values`
    
    `logits` has shape (B, V).
    
    This is the second branch where `logits` is used, hence we need to accumulate its
    gradient. 
    
    First, we calculate the local derivative dlogit_maxes/dlogits.
    """
    local_derivative = torch.zeros_like(logits)
    indices = logits.max(dim=1, keepdim=True).indices
    local_derivative[torch.arange(B), indices] = 1.0
    dlogits_branch_2 = dlogit_maxes * local_derivative
    dlogits += dlogit_maxes * local_derivative
    assert compare("dlogits", dlogits, logits)
    # Note also that this contribution should be zero since `dlogit_maxes` is 0
    assert torch.allclose(dlogits_branch_2, torch.zeros_like(dlogits_branch_2))

    """
    `logits = h @ W2 + b2`
    
    To understand how backpropagation works on a matrix multiplication, consider the
    small example f = a @ b + c:
    
          f11 = a11 * b11 + a12 * b21 + c1
          f12 = a11 * b12 + a12 * b22 + c2
          f21 = a21 * b11 + a22 * b21 + c1
          f22 = a21 * b12 + a22 * b22 + c2

    Using the chain rule, we have:
    
        dLda11 = dLdf11 * b11 + dLdf12 * b12
        dLda12 = dLdf11 * b21 + dLdf12 * b22
        dLda21 = dLdf21 * b11 + dLdf22 * b12
        dLda22 = dLdf21 * b21 + dLdf22 * b22

    The addition is because we have to sum contributions. Clearly, 
    
        dLda = dLdf @ b^T
        
    In our case (`logits = h @ W2 + b2`), dLda is dh, dLdf is dlogits, and b is W2.
    """
    dh = dlogits @ W2.T
    assert compare("dh", dh, h)

    """
    Doing something similar for b gives: 
    
        dLdb = a^T @ dLdf
    
    In our case, dLdb is dW2, a is h, and dLdf is dlogits.
    """
    dW2 = h.T @ dlogits
    assert compare("dW2", dW2, W2)

    """
    Now for the bias:
        dL/dc1 = dLdf11 * 1 + dLdf21 * 1
        dL/dc2 = dLdf12 * 1 + dLdf22 * 1
        
    This is summing across columns, so: dLdc = dLdf.sum(dim=1).
    In our case, dLdc is dLdb2 and dLdf is dlogits. 
    """
    db2 = (dlogits * 1.0).sum(dim=0, keepdim=True)
    assert compare("db2", db2, b2)

    """
    `h = torch.tanh(hpreact)`
    
    `hpreact` has shape (32, 64). 
    
    dL/dhpreact = dL/dh * dh/dhpreact 
                = dL/dh * sech^2(hpreact)
                = dL/dh * (1 - tanh^2(hpreact)) 
                = dL/dh * (1 - h^2)
    """
    dhpreact = dh * (1 - h ** 2)
    assert compare("dhpreact", dhpreact, hpreact)

    """
    `hpreact = bngain * bnraw + bnbias`
    
    `bngain` has shape (1, 64),
    `bnraw` has shape (32, 64), 
    `bnbias` has shape (1, 64).
    
    Thus, `bngain` and `bnbias` are broadcasted along the 0th dimension, and we need to
    do a sum because they are effectively used in multiple branches in the computation 
    graph.
    """
    dbngain = (dhpreact * bnraw).sum(dim=0, keepdim=True)
    assert compare("dbngain", dbngain, bngain)
    dbnraw = dhpreact * bngain
    assert compare("dbnraw", dbnraw, bnraw)
    dbnbias = (dhpreact * 1.0).sum(dim=0, keepdim=True)
    assert compare("dbnbias", dbnbias, bnbias)

    """
    `bnraw = bndiff * bnvar_inv`
    
    `bndiff` has shape (32, 64),
    `bnvar_inv` has shape (1, 64).
    
    Thus, `bnvar_inv` is broadcasted along dimension 0 and we need to do a sum along
    this dimension when computing `dbnvar_inv`. 
    """
    # Can't check yet because `bndiff` is used in another branch
    dbndiff = dbnraw * bnvar_inv

    dbnvar_inv = (dbnraw * bndiff).sum(dim=0, keepdim=True)
    assert compare("dbnvar_inv", dbnvar_inv, bnvar_inv)

    """
    `bnvar_inv = (bnvar + 1e-5).rsqrt()`
    
    `bnvar` has shape (1, 64).
    
    Use the power rule of differentiation:
        d/dx x^(-1/2) = -1/2 * x^(-3/2)
    """
    dbnvar = dbnvar_inv * (-1/2 * (bnvar + 1e-5) ** (-3/2))
    assert compare("dbnvar", dbnvar, bnvar)

    """
    `bnvar = 1/(B-1) * bndiff2.sum(dim=0, keepdim=True)`
    
    `bnvar` has shape (1, 64), 
    `bndiff2` has shape (32, 64).
    
    There is no broadcasting, so we don't need to sum gradients across dimension 0.
    
    NOTE: notice the duality - when there is a sum over a particular dimension in the
    forward pass, there is a corresponding broadcasting along that same dimension in 
    the backward pass. Similarly, when there is a broadcast operation in the forward 
    pass, this indicates a variable reuse and in the backward pass, this turns into a
    sum over the same dimension.

    Small example:
          a11 a12
          a21 a22
    Gives
          b1 b2
    where
          b1 = a11 + a21
          b2 = a12 + a22
    """
    # Broadcasting in backward pass: (1, n_hidden) * (B, n_hidden) -> (B, n_hidden)
    dbndiff2 = dbnvar * torch.ones_like(bndiff2) * 1.0 / (B - 1.0)
    assert compare("dbndiff2", dbndiff2, bndiff2)

    """`bndiff2 = bndiff ** 2`
    
    `bndiff` has shape (32, 64).
    
    This is the second branch where `bndiff` is used, so we need to accumulate the
    gradients. 
    """
    dbndiff += dbndiff2 * (2 * bndiff)
    assert compare("dbndiff", dbndiff, bndiff)

    """
    `bndiff = hprebn - bnmeani`
    
    `hprebn` has shape (32, 64),
    `bnmeani` has shape (1, 64).
    
    Thus, `bnmeani` is broadcasted, and we need to sum its gradients along dimension 0.
    """
    # Can't check `dhprebn` yet because `hprebn` is used in another branch
    dhprebn = dbndiff * 1.0

    dbnmeani = (dbndiff * -1.0).sum(dim=0, keepdim=True)
    assert compare("dbnmeani", dbnmeani, bnmeani)

    """
    `bnmeani = 1 / B * hprebn.sum(dim=0, keepdim=True)`
    
    `hprebn` has shape (32, 64).
    
    Since we have a sum along dimension 0 in the forward pass, we need a broadcast in
    dimension 0 in the backward pass. 
    """
    # Broadcasting: (1, n_hidden) * (B, n_hidden) -> (B, n_hidden)
    dhprebn += dbnmeani * (1/B * torch.ones_like(hprebn))
    assert compare("dhprebn", dhprebn, hprebn)

    """
    `hprebn = embcat @ W1 + b1`
    
    `embcat` has shape (32, 30),
    `W1` has shape (30, 64),
    `b1` has shape (64,).
    """
    dembcat = dhprebn @ W1.T
    assert compare("dembcat", dembcat, embcat)
    dW1 = embcat.T @ dhprebn
    assert compare("dW1", dW1, W1)
    db1 = (dhprebn * 1.0).sum(dim=0, keepdim=True)
    assert compare("db1", db1, b1)

    """
    `embcat = emb.view(B, -1)`
    
    `embcat` has shape (B, block_size * n_embd),
    `emb` has shape (B, block_size, n_embd).
    
    The view operation reorganised the array in the forward pass, so we just need to
    revert this in the backward pass.
    """
    demb = dembcat.view(emb.shape)
    assert compare("demb", demb, emb)

    """
    `emb = C[x_batch]`
    
    `C` has shape (vocab_size, n_embd),
    `x_batch` has shape (B, block_size).
    
    dL/dC = dL/demb * demb/dC
    """
    dC = torch.zeros_like(C)
    for k in range(x_batch.shape[0]):
        for j in range(x_batch.shape[1]):
            ix = x_batch[k, j]
            dC[ix] += demb[k, j]  # Addition since a row could be used multiple times
    assert compare("dC", dC, C)

    """Bonus: we backpropagate through a faster calculation of cross entropy:
     
     `loss = F.cross_entropy(logits, y_batch)`
    
    It can be shown that
        dloss/dl_i = p_i        if i≠y
        dloss/dl_i = p_i - 1    if i=y
    
    where `p_i` is a softmax probability. 
    """
    # First, verify that the loss is the same
    loss_fast = F.cross_entropy(logits, y_batch)
    assert torch.allclose(loss_fast, loss)
    # Secondly, implement and check the faster gradient computation of cross entropy.
    dlogits = F.softmax(logits, dim=1)
    dlogits[range(B), y_batch] -= 1  # Since dloss/dl_i = p_i - 1 for i=y
    dlogits /= B  # Need to divide since F.cross_entropy() computes the mean NLL
    assert compare("dlogits", dlogits, logits)

    """Bonus 2: we backpropagate through a faster calculation of batch norm. """
    # First, verify that the result of the batch norm is the same
    hpreact_fast = bngain * (hprebn - hprebn.mean(dim=0, keepdim=True)) \
        / torch.sqrt(hprebn.var(dim=0, keepdim=True, unbiased=True) + 1e-5) \
        + bnbias
    assert torch.allclose(hpreact_fast, hpreact)
    # Secondly, implement and check the faster gradient computation of the batch norm
    dhprebn = bngain * bnvar_inv/B * (
                B * dhpreact - dhpreact.sum(dim=0) -
                B/(B-1) * bnraw * (dhpreact * bnraw).sum(dim=0)
            )
    assert compare("dhprebn", dhprebn, hprebn)

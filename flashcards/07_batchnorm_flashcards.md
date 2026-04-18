# BatchNorm Flashcards

## Card 1 — Definition
**Q:** What is the formula for Batch Normalization during training?
**A:** $\text{BN}(x) = \gamma \cdot \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$ — where $\mu_B$ and $\sigma_B^2$ are the mean and variance computed **across the batch** (dim=0), and $\gamma$, $\beta$ are learnable scale and shift parameters.

## Card 2 — Implementation
**Q:** Implement Batch Normalization in PyTorch without using `F.batch_norm` or `nn.BatchNorm1d`.
**A:**
```python
def my_batch_norm(x, gamma, beta, running_mean, running_var,
                  eps=1e-5, momentum=0.1, training=True):
    if training:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(momentum * batch_mean)
            running_var.mul_(1 - momentum).add_(momentum * batch_var)
        x_norm = (x - batch_mean) / torch.sqrt(batch_var + eps)
    else:
        x_norm = (x - running_mean) / torch.sqrt(running_var + eps)
    return gamma * x_norm + beta
```

## Card 3 — Training vs Inference
**Q:** How does BatchNorm behave differently during training and inference?
**A:** During **training**, it computes mean and variance from the current mini-batch and updates running statistics. During **inference**, it uses the accumulated `running_mean` and `running_var` instead, so the output is deterministic and independent of other samples in the batch.

## Card 4 — Running Statistics Update
**Q:** How are `running_mean` and `running_var` updated during training?
**A:** Using exponential moving average: `running = (1 - momentum) * running + momentum * batch_stat`. With the default `momentum=0.1`, 90% of the old estimate is kept and 10% comes from the new batch. This accumulates a stable estimate for use at inference time.

## Card 5 — Why `torch.no_grad()` for Running Stats?
**Q:** Why must running statistics be updated inside `torch.no_grad()`?
**A:** Running statistics are **buffers**, not learnable parameters — they should not receive gradients. Without `torch.no_grad()`, the in-place updates (`.mul_()`, `.add_()`) would be tracked by autograd, causing errors during backpropagation through the in-place modification of leaf tensors.

## Card 6 — Normalization Axis: dim=0
**Q:** Why does BatchNorm compute mean/variance over `dim=0` while LayerNorm uses `dim=-1`?
**A:** BatchNorm normalizes each **feature across all samples** in the batch (dim=0 is the batch dimension). LayerNorm normalizes each **sample across all features** (dim=-1 is the feature dimension). For input `(N, D)`: BatchNorm produces one mean per feature (shape `(D,)`), LayerNorm produces one mean per sample (shape `(N, 1)`).

## Card 7 — Why `unbiased=False`?
**Q:** Why use `x.var(dim=0, unbiased=False)` instead of the default?
**A:** `unbiased=False` computes population variance (divides by $N$). BatchNorm's definition uses population variance over the batch. Using `unbiased=True` (Bessel's correction, divides by $N-1$) would not match PyTorch's `nn.BatchNorm1d` behavior and would produce incorrect normalization.

## Card 8 — When BatchNorm Fails
**Q:** In what situations does BatchNorm perform poorly?
**A:** (1) **Small batch sizes** — batch statistics become noisy and unstable. (2) **Variable-length sequences** (RNNs/Transformers) — batch statistics are unreliable when samples have different lengths. (3) **Batch size of 1** — variance is zero, normalization degenerates. These are the main reasons LayerNorm is preferred in Transformers.

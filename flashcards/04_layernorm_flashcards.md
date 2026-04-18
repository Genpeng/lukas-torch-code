# LayerNorm Flashcards

## Card 1 — Definition
**Q:** What is the mathematical formula for Layer Normalization?
**A:** $\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ — where $\mu$ and $\sigma^2$ are the mean and variance computed over the last dimension, and $\gamma$, $\beta$ are learnable scale and shift parameters.

## Card 2 — Implementation
**Q:** Implement Layer Normalization in PyTorch without using `F.layer_norm` or `torch.nn.LayerNorm`.
**A:**
```python
def my_layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_normalized = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_normalized + beta
```

## Card 3 — Why `unbiased=False`?
**Q:** Why use `x.var(dim=-1, unbiased=False)` instead of the default `unbiased=True`?
**A:** `unbiased=False` computes population variance (divides by $n$) instead of sample variance (divides by $n-1$). LayerNorm's definition uses population variance — it normalizes over the entire feature dimension of each sample, not estimating a population statistic from a sample.

## Card 4 — Why `keepdim=True`?
**Q:** Why is `keepdim=True` needed in both `mean()` and `var()`?
**A:** Without `keepdim=True`, the reduced dimension is dropped, and broadcasting with the original tensor `x` fails. `keepdim=True` preserves the dimension as size 1, so subtraction `(x - mean)` and division broadcast correctly across the last dimension.

## Card 5 — The `eps` Parameter
**Q:** What is the purpose of `eps` in the denominator $\sqrt{\sigma^2 + \epsilon}$?
**A:** It prevents division by zero when the variance is 0 (e.g., all values in the feature dimension are identical). A typical default is `1e-5`.

## Card 6 — LayerNorm vs BatchNorm
**Q:** What is the key difference between LayerNorm and BatchNorm in terms of normalization axis?
**A:** LayerNorm normalizes over the **feature dimension** (last dim) independently per sample. BatchNorm normalizes over the **batch dimension** per feature. This means LayerNorm's behavior is identical at train and inference time, making it preferred in transformers and RNNs where batch statistics are unreliable.

## Card 7 — Role of Gamma and Beta
**Q:** Why does LayerNorm include learnable $\gamma$ (scale) and $\beta$ (shift) parameters?
**A:** After normalization, the output has zero mean and unit variance, which limits representational power. $\gamma$ and $\beta$ allow the network to learn to undo the normalization when beneficial — effectively learning the optimal scale and shift for each feature.

## Card 8 — Edge Case
**Q:** What does `my_layer_norm(tensor([[3.0, 3.0, 3.0]]), ones(3), zeros(3))` output?
**A:** `tensor([[0., 0., 0.]])` — all values are equal so `x - mean = 0` everywhere. The numerator is all zeros, so the output is all zeros (plus $\beta = 0$). The `eps` prevents the division by zero in the denominator.

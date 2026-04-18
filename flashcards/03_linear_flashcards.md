# Linear Layer Flashcards

## Card 1 — Definition
**Q:** What is the mathematical formula for a fully-connected linear layer?
**A:** $y = xW^\top + b$ — each input vector $x$ is multiplied by the transposed weight matrix and shifted by a bias vector.

## Card 2 — Implementation
**Q:** Implement a `SimpleLinear` layer in PyTorch without using `torch.nn.Linear`.
**A:**
```python
class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
        self.weight = torch.randn(out_features, in_features) / math.sqrt(in_features)
        self.weight.requires_grad_()
        self.bias = torch.zeros(out_features, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias
```

## Card 3 — Weight Initialization
**Q:** Why initialize weights with `randn * (1/√in_features)` instead of plain `randn`?
**A:** Scaling by $1/\sqrt{\text{in\_features}}$ keeps the variance of each layer's output roughly equal to 1, preventing activations from exploding or vanishing as signals propagate through the network. This is the core idea behind Kaiming/He initialization.

## Card 4 — Why `requires_grad=True`?
**Q:** Why must `weight` and `bias` have `requires_grad=True`?
**A:** Without it, PyTorch's autograd will not track operations on these tensors. No gradient will be computed during `backward()`, so the parameters can never be updated by an optimizer.

## Card 5 — Shape Convention
**Q:** Why is `weight` shaped `(out_features, in_features)` rather than `(in_features, out_features)`?
**A:** This matches the `torch.nn.Linear` convention. The forward pass computes `x @ W.T + b`, where `x` has shape `(batch, in_features)` and `W.T` has shape `(in_features, out_features)`, yielding output shape `(batch, out_features)`.

## Card 6 — Gradient
**Q:** Given $y = xW^\top + b$ and a scalar loss $L$, what are $\frac{\partial L}{\partial W}$ and $\frac{\partial L}{\partial b}$?
**A:**
- $\frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial y}\right)^\top x$ — shape `(out_features, in_features)`, same as $W$.
- $\frac{\partial L}{\partial b} = \sum_{\text{batch}} \frac{\partial L}{\partial y}$ — shape `(out_features,)`, summed over the batch dimension.

## Card 7 — Edge Case
**Q:** For `SimpleLinear(8, 4)` with input `x` of shape `(2, 8)`, what is the output shape?
**A:** `(2, 4)` — the batch dimension (2) is preserved, and the feature dimension is projected from 8 → 4.

# Softmax Flashcards

## Card 1 — Definition
**Q:** What is the mathematical definition of Softmax?
**A:** $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ — converts a vector of real numbers into a probability distribution that sums to 1.0.

## Card 2 — Implementation
**Q:** Implement a numerically stable Softmax in PyTorch without using `torch.softmax`, `F.softmax`, or `torch.nn.Softmax`.
**A:**
```python
def my_softmax(x, dim=-1):
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)
```

## Card 3 — Numerical Stability
**Q:** Why subtract `x.max()` before computing `exp` in Softmax?
**A:** Large values of x cause `exp(x)` to overflow to `inf`. Subtracting the max shifts all values to ≤ 0, so `exp` outputs are in (0, 1]. This doesn't change the result because $\frac{e^{x_i - c}}{\sum e^{x_j - c}} = \frac{e^{x_i}}{\sum e^{x_j}}$ — the constant cancels out.

## Card 4 — Why `keepdim=True`?
**Q:** Why is `keepdim=True` important in both `max()` and `sum()`?
**A:** Without `keepdim=True`, the reduced dimension is dropped, and broadcasting with the original tensor fails. `keepdim=True` preserves the dimension (as size 1) so subtraction and division broadcast correctly.

## Card 5 — The `dim` Parameter
**Q:** What does the `dim` parameter control in Softmax?
**A:** It specifies which axis to normalize over. For a 2-D tensor with shape (batch, classes), `dim=-1` normalizes each row independently so each row sums to 1.0.

## Card 6 — Output Properties
**Q:** What two properties does Softmax output always satisfy?
**A:**
1. Every element is in (0, 1) — strictly positive, never exactly 0 or 1.
2. Elements along the specified dim sum to 1.0.

## Card 7 — Gradient
**Q:** What is the Jacobian of Softmax with respect to its input?
**A:** For output $s = \text{softmax}(x)$: $\frac{\partial s_i}{\partial x_j} = s_i(\delta_{ij} - s_j)$, where $\delta_{ij}$ is the Kronecker delta. In matrix form: $\text{diag}(s) - s \cdot s^\top$.

## Card 8 — Edge Case
**Q:** What does `my_softmax(tensor([1.0, 2.0, 3.0]))` output?
**A:** `tensor([0.0900, 0.2447, 0.6652])` — the largest input (3.0) gets the highest probability, and all three sum to 1.0.

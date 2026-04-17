# ReLU Flashcards

## Card 1 — Definition
**Q:** What is the mathematical definition of ReLU?
**A:** $\text{ReLU}(x) = \max(0, x)$ — outputs $x$ if positive, $0$ otherwise.

## Card 2 — Implementation
**Q:** Implement ReLU in PyTorch without using `torch.relu`, `F.relu`, or `torch.clamp`.
**A:** `return (x > 0).to(x.dtype) * x`

## Card 3 — Why `.to(x.dtype)`?
**Q:** In `(x > 0).to(x.dtype) * x`, why is `.to(x.dtype)` needed?
**A:** `x > 0` returns a `BoolTensor`. `.to(x.dtype)` explicitly casts it to `0.0/1.0`, making the intent clear and ensuring consistent dtype behavior.

## Card 4 — Autograd
**Q:** Why does the mask-multiply approach `(x > 0).to(x.dtype) * x` support autograd?
**A:** The `*` multiplication is a differentiable op that PyTorch can track. Gradients flow through `x` where the mask is 1 (positive region) and are zeroed where the mask is 0 (negative region). The comparison `x > 0` itself has no gradient — it only produces the mask.

## Card 5 — Gradient
**Q:** What is the gradient of ReLU?
**A:** $\frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ — this is exactly the boolean mask `(x > 0)`.

## Card 6 — Edge Case
**Q:** What does this ReLU implementation output for `x = tensor([-2., -1., 0., 1., 2.])`?
**A:** `tensor([0., 0., 0., 1., 2.])` — all negatives become 0, zero stays 0, positives pass through.

## Card 7 — Alternative Implementations
**Q:** Name two other ways to implement ReLU from scratch (without built-ins).
**A:**
1. `torch.where(x > 0, x, torch.zeros_like(x))`
2. `x * (x > 0).float()`

# RMSNorm Flashcards

## Card 1 — Definition
**Q:** What is the formula for Root Mean Square Layer Normalization (RMSNorm)?
**A:** $\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot w, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}$ — it normalizes by the root mean square of the input, then scales by a learnable weight $w$. Unlike LayerNorm, there is **no mean subtraction** and **no bias term**.

## Card 2 — Implementation
**Q:** Implement RMSNorm in PyTorch without using any built-in norm layers.
**A:**
```python
def rms_norm(x, weight, eps=1e-6):
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_square + eps)
    x_normalized = x / rms
    return x_normalized * weight
```

## Card 3 — RMSNorm vs LayerNorm
**Q:** What are the key differences between RMSNorm and LayerNorm?
**A:** (1) **No mean subtraction** — RMSNorm skips the re-centering step `(x - μ)`. (2) **No bias** — RMSNorm only has a scale parameter `w`, no shift `β`. (3) **Denominator** — RMSNorm divides by $\sqrt{\text{mean}(x^2) + \epsilon}$ instead of $\sqrt{\text{var}(x) + \epsilon}$. This makes RMSNorm ~10–15% faster while performing comparably in practice.

## Card 4 — Why No Mean Subtraction?
**Q:** Why does RMSNorm omit the mean subtraction that LayerNorm uses?
**A:** The original RMSNorm paper (Zhang & Sennrich, 2019) showed that the re-centering (mean subtraction) in LayerNorm is not essential for performance — the re-scaling by RMS is the critical component. Removing mean subtraction reduces computation and simplifies the implementation without hurting model quality.

## Card 5 — Why `keepdim=True`?
**Q:** Why is `keepdim=True` needed in `x.pow(2).mean(dim=-1, keepdim=True)`?
**A:** Without `keepdim=True`, the last dimension is dropped after the mean, making the shape incompatible for broadcasting with `x` during division. `keepdim=True` preserves it as size 1, so `x / rms` broadcasts correctly across the feature dimension.

## Card 6 — Where Is RMSNorm Used?
**Q:** Which major models use RMSNorm instead of LayerNorm?
**A:** RMSNorm is used in **LLaMA**, **Gemma**, **Mistral**, and many other modern LLMs. It replaced LayerNorm in these architectures because it is faster (fewer operations) and achieves similar performance, making it the de facto normalization for recent large language models.

## Card 7 — The `eps` Parameter
**Q:** What is the purpose of `eps` in $\sqrt{\text{mean}(x^2) + \epsilon}$, and why is it typically `1e-6` instead of `1e-5`?
**A:** `eps` prevents division by zero when all input values are zero. RMSNorm commonly uses `1e-6` (vs LayerNorm's `1e-5`) because modern LLMs often use mixed precision (float16/bfloat16) where tighter numerical stability is needed, and the smaller epsilon matches the convention set by LLaMA and similar models.

## Card 8 — Shape Walkthrough
**Q:** For input `x: (2, 6, 512)` and `weight: (512,)`, trace the shapes through `rms_norm`.
**A:**
1. `x.pow(2)` → `(2, 6, 512)` — element-wise square
2. `.mean(dim=-1, keepdim=True)` → `(2, 6, 1)` — mean over features
3. `torch.sqrt(... + eps)` → `(2, 6, 1)` — RMS value per position
4. `x / rms` → `(2, 6, 512)` — normalized (broadcasts the `1` dim)
5. `* weight` → `(2, 6, 512)` — scaled by learnable weight (broadcasts `(512,)`)

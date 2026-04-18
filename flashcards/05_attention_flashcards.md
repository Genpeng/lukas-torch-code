# Scaled Dot-Product Attention Flashcards

## Card 1 — Definition
**Q:** What is the formula for Scaled Dot-Product Attention?
**A:** $\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ — queries attend to keys to produce weights, which are applied to values.

## Card 2 — Implementation
**Q:** Implement Scaled Dot-Product Attention in PyTorch without using `F.scaled_dot_product_attention`.
**A:**
```python
def scaled_dot_product_attention(Q, K, V):
    scores = Q @ K.transpose(-2, -1)
    scores_scaled = scores / math.sqrt(Q.size(-1))
    attention_weights = torch.softmax(scores_scaled, dim=-1)
    return attention_weights @ V
```

## Card 3 — Why Scale by √d_k?
**Q:** Why divide the dot-product scores by $\sqrt{d_k}$?
**A:** As $d_k$ grows, the dot products $QK^\top$ grow in magnitude (their variance scales with $d_k$). Large values push softmax into saturated regions where gradients are near zero. Dividing by $\sqrt{d_k}$ keeps the variance at ~1, maintaining healthy gradients.

## Card 4 — Shape Walkthrough
**Q:** Given `Q: (B, seq_q, d_k)`, `K: (B, seq_k, d_k)`, `V: (B, seq_k, d_v)`, trace the shapes through each step.
**A:**
1. `Q @ K.T` → `(B, seq_q, seq_k)` — attention scores
2. `softmax(...)` → `(B, seq_q, seq_k)` — attention weights (rows sum to 1)
3. `weights @ V` → `(B, seq_q, d_v)` — weighted combination of values

## Card 5 — Cross-Attention
**Q:** How does this implementation support cross-attention where `seq_q ≠ seq_k`?
**A:** The matrix multiply `Q @ K.T` naturally handles different sequence lengths: `(B, seq_q, d_k) @ (B, d_k, seq_k) → (B, seq_q, seq_k)`. The only constraint is that `Q` and `K` share the same `d_k`, and `K` and `V` share the same `seq_k`.

## Card 6 — Softmax Dimension
**Q:** Why is softmax applied along `dim=-1` (the last dimension)?
**A:** Each row of the score matrix represents one query's similarity to all keys. Softmax along `dim=-1` normalizes these similarities into a probability distribution, so each query's attention weights over all keys sum to 1.

## Card 7 — Attention as Soft Lookup
**Q:** How can attention be understood as a "soft dictionary lookup"?
**A:** Keys are like dictionary keys and values are like dictionary values. The query computes a similarity score against every key, then softmax turns scores into weights. The output is a weighted average of all values — a "soft" retrieval rather than a hard lookup of a single entry.

# Causal Self-Attention Flashcards

## Card 1 — Definition
**Q:** What is causal (masked) self-attention, and how does it differ from standard self-attention?
**A:** Causal self-attention restricts each position to only attend to itself and earlier positions (no future tokens). The score matrix is modified as: $\text{scores}_{ij} = \frac{Q_i \cdot K_j}{\sqrt{d_k}}$ if $j \le i$, and $-\infty$ if $j > i$. After softmax, future positions get zero weight. This is the core mechanism in GPT-style autoregressive decoders.

## Card 2 — Implementation
**Q:** Implement causal self-attention in PyTorch without using `F.scaled_dot_product_attention`.
**A:**
```python
def causal_attention(Q, K, V):
    _, seq_len, d_k = Q.shape
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    return attn_weights @ V
```

## Card 3 — Why `-inf` Instead of `0`?
**Q:** Why do we mask future positions with `-inf` instead of `0`?
**A:** Because the mask is applied **before** softmax. Setting scores to `-inf` means $e^{-\infty} = 0$ after softmax, so those positions contribute exactly zero attention weight. If we used `0` instead, those positions would get $e^0 = 1$ — a **uniform positive weight** — which would leak future information into the output.

## Card 4 — `torch.tril` vs `torch.triu`
**Q:** How do `torch.tril` and `torch.triu` relate to the causal mask?
**A:** `torch.tril` (lower triangular) keeps elements at and below the diagonal — this gives the **allowed** positions (past + current). `torch.triu` (upper triangular) keeps elements at and above the diagonal — this gives the **forbidden** positions (future + current diagonal). For causal masking, use `torch.tril` to build the allow-mask, then fill positions where `mask == 0` with `-inf`.

## Card 5 — First Position Behavior
**Q:** What happens at position 0 in causal attention, and why?
**A:** Position 0 can only attend to itself (all other positions are masked). After softmax, its attention weight is `[1, 0, 0, ...]`, so the output at position 0 is exactly `V[0]`. This is a useful sanity check: `torch.allclose(output[:, 0], V[:, 0])` should be `True`.

## Card 6 — Shape Walkthrough
**Q:** For `Q, K, V: (2, 4, 8)`, trace the shapes through `causal_attention`.
**A:**
1. `Q @ K.transpose(-2, -1)` → `(2, 4, 8) × (2, 8, 4)` → `(2, 4, 4)` — raw scores
2. `/ math.sqrt(8)` → `(2, 4, 4)` — scaled scores
3. `torch.tril(torch.ones(4, 4))` → `(4, 4)` — lower-triangular mask
4. `masked_fill(mask == 0, -inf)` → `(2, 4, 4)` — mask broadcasts over batch
5. `softmax(dim=-1)` → `(2, 4, 4)` — attention weights (rows sum to 1)
6. `attn_weights @ V` → `(2, 4, 4) × (2, 4, 8)` → `(2, 4, 8)` — output

## Card 7 — Causal Attention in Transformers
**Q:** Where is causal attention used in the Transformer architecture, and why?
**A:** Causal attention is used in the **decoder** (GPT, LLaMA, etc.) during autoregressive generation. During training, the model sees the full sequence at once but must predict each token using only previous tokens. The causal mask enforces this constraint efficiently in a single forward pass, enabling **teacher forcing** — training on all positions in parallel while preventing information leakage from future tokens.

## Card 8 — Scaling Factor
**Q:** Why divide scores by $\sqrt{d_k}$ before masking and softmax?
**A:** Without scaling, the dot products grow proportionally to $d_k$, pushing softmax into saturated regions where gradients vanish. Dividing by $\sqrt{d_k}$ keeps the variance of scores approximately 1 regardless of dimension, ensuring softmax produces well-distributed weights with healthy gradients. This scaling is independent of the causal mask — it applies to all attention variants.

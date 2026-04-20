# Sliding Window Attention Flashcards

## Card 1 — Definition
**Q:** What is sliding window attention, and where is it used?
**A:** Sliding window attention restricts each position $i$ to attend only to positions $j$ where $|i - j| \le w$ (the window size). This creates a band-shaped attention pattern instead of full quadratic attention. It is used in **Longformer**, **Mistral**, and other models designed for efficient long-context processing, reducing complexity from $O(n^2)$ to $O(n \cdot w)$.

## Card 2 — Implementation
**Q:** Implement sliding window attention in PyTorch without sparse attention libraries.
**A:**
```python
def sliding_window_attention(Q, K, V, window_size):
    _, seq_len, d_k = Q.size()
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    i = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    j = torch.arange(seq_len).unsqueeze(0)  # (1, seq_len)
    mask = (i - j).abs() > window_size
    scores = scores.masked_fill(mask, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    return attn_weights @ V
```

## Card 3 — Mask Construction with Broadcasting
**Q:** How does the broadcasting trick with `unsqueeze` construct the sliding window mask?
**A:** `i = arange(seq_len).unsqueeze(1)` creates a column vector `(seq_len, 1)` and `j = arange(seq_len).unsqueeze(0)` creates a row vector `(1, seq_len)`. When subtracted, broadcasting produces a `(seq_len, seq_len)` distance matrix where element `(r, c)` = `r - c`. Taking `.abs() > window_size` marks all positions outside the window as `True` (to be masked).

## Card 4 — Edge Cases
**Q:** What happens with `window_size=0` and `window_size >= seq_len`?
**A:** **`window_size=0`**: each position only attends to itself. The mask zeros out everything except the diagonal, softmax gives weight 1.0 to self, so the output equals `V` exactly. **`window_size >= seq_len`**: no positions are masked (all distances are ≤ window_size), so it becomes equivalent to full (standard) self-attention.

## Card 5 — Sliding Window vs Causal Attention
**Q:** How does sliding window attention differ from causal attention?
**A:** Causal attention is **asymmetric** — position $i$ sees positions $j \le i$ (past only). Sliding window is **symmetric** — position $i$ sees positions $|i - j| \le w$ (both past and future within the window). Causal attention prevents future leakage for autoregressive models; sliding window limits attention range for efficiency while keeping bidirectional context. They can be combined (e.g., Mistral uses causal + sliding window).

## Card 6 — Shape Walkthrough
**Q:** For `Q, K, V: (2, 6, 8)` and `window_size=1`, trace the shapes through `sliding_window_attention`.
**A:**
1. `Q @ K.transpose(-2, -1)` → `(2, 6, 8) × (2, 8, 6)` → `(2, 6, 6)` — raw scores
2. `/ math.sqrt(8)` → `(2, 6, 6)` — scaled scores
3. `i: (6, 1)`, `j: (1, 6)` → `(i - j).abs()` → `(6, 6)` distance matrix
4. `> 1` → `(6, 6)` boolean mask (tridiagonal band is `False`)
5. `masked_fill(mask, -inf)` → `(2, 6, 6)` — mask broadcasts over batch
6. `softmax(dim=-1)` → `(2, 6, 6)` — weights (each row sums to 1)
7. `attn_weights @ V` → `(2, 6, 6) × (2, 6, 8)` → `(2, 6, 8)` — output

## Card 7 — Why Sliding Window for Long Sequences?
**Q:** What computational advantage does sliding window attention provide over full attention?
**A:** Full attention computes and stores an $n \times n$ score matrix — $O(n^2)$ in time and memory. With sliding window of size $w$, each position only attends to $2w + 1$ neighbors, so effective computation is $O(n \cdot w)$. For long sequences (e.g., $n = 32{,}768$) with small $w$ (e.g., $w = 4{,}096$), this is a significant saving. Information can still propagate across the full sequence through multiple stacked layers.

## Card 8 — `unsqueeze(0)` vs `unsqueeze(1)` Intuition
**Q:** When building the distance matrix, why does `unsqueeze(1)` make row indices and `unsqueeze(0)` make column indices?
**A:** `unsqueeze(1)` adds a dimension at axis 1, turning `(seq_len,)` into `(seq_len, 1)` — a column vector where each row is a different query position. `unsqueeze(0)` adds at axis 0, giving `(1, seq_len)` — a row vector where each column is a different key position. Subtraction broadcasts these into a full `(seq_len, seq_len)` matrix of pairwise distances.

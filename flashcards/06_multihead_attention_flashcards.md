# Multi-Head Attention Flashcards

## Card 1 — Definition
**Q:** What is the formula for Multi-Head Attention?
**A:**
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)$$
Each head independently attends to different parts of the representation, then results are concatenated and projected.

## Card 2 — Implementation
**Q:** Implement `MultiHeadAttention` from scratch without using `torch.nn.MultiheadAttention`.
**A:**
```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        B = Q.size(0)
        q = self.W_q(Q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1) @ v
        attn = attn.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(attn)
```

## Card 3 — Reshape: view + transpose
**Q:** Why do we use `.view(B, -1, num_heads, d_k).transpose(1, 2)` after the linear projection?
**A:** The linear layer outputs `(B, seq, d_model)`. `.view(B, -1, num_heads, d_k)` splits `d_model` into `num_heads × d_k`. `.transpose(1, 2)` moves the head dim before the seq dim, giving `(B, num_heads, seq, d_k)` — so each head's attention can be computed as a single batched matmul.

## Card 4 — Why `.contiguous()` Before `.view()`?
**Q:** Why is `.contiguous()` needed after `.transpose(1, 2)` and before `.view()`?
**A:** `.transpose()` returns a view with non-contiguous memory layout. `.view()` requires contiguous memory to reshape without copying. `.contiguous()` copies the data into a contiguous block so `.view()` can proceed.

## Card 5 — Why Multiple Heads?
**Q:** What is the benefit of using multiple heads instead of a single large attention?
**A:** Each head learns to attend to different aspects of the input (e.g., one head might focus on syntactic relations, another on semantic similarity). Multiple heads give the model richer representational power at the same computational cost as single-head attention with the same `d_model`.

## Card 6 — The Output Projection W_o
**Q:** What is the purpose of the final `W_o` linear projection?
**A:** After concatenating all heads, the output has shape `(B, seq, d_model)` but is just a stack of independent head outputs. `W_o` mixes information across heads, allowing the model to combine what different heads learned into a unified representation.

## Card 7 — Shape Walkthrough
**Q:** For `d_model=32, num_heads=4`, trace the shapes through `forward(Q, K, V)` where `Q: (2, 6, 32)`, `K/V: (2, 10, 32)`.
**A:**
1. `W_q(Q)` → `(2, 6, 32)`, then view+transpose → `(2, 4, 6, 8)`
2. `W_k(K)` → `(2, 10, 32)`, then view+transpose → `(2, 4, 10, 8)`
3. `q @ k.T` → `(2, 4, 6, 10)` — scores
4. `softmax @ v` → `(2, 4, 6, 8)` — per-head output
5. transpose+view → `(2, 6, 32)` — concat heads
6. `W_o(...)` → `(2, 6, 32)` — final output

## Card 8 — d_model Must Be Divisible by num_heads
**Q:** Why must `d_model % num_heads == 0`?
**A:** Each head gets `d_k = d_model // num_heads` dimensions. If `d_model` is not divisible, we can't evenly split the projected vectors across heads, and the `.view()` reshape would fail.

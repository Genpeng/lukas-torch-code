# MHA vs MQA vs GQA：注意力机制对比

## 1. 概览

这三种机制的核心区别在于 **K/V head 的数量**：

| 机制 | 全称 | Q heads | KV heads | KV cache 大小 |
|------|------|---------|----------|--------------|
| **MHA** | Multi-Head Attention | $H$ | $H$ | $H \times d_k$ |
| **MQA** | Multi-Query Attention | $H$ | $1$ | $1 \times d_k$ |
| **GQA** | Grouped Query Attention | $H$ | $G$（$1 < G < H$） | $G \times d_k$ |

其中 $H$ = `num_heads`，$G$ = `num_kv_heads`，$d_k$ = `d_model // num_heads`。

```
MHA:  Q₁→KV₁  Q₂→KV₂  Q₃→KV₃  Q₄→KV₄  Q₅→KV₅  Q₆→KV₆  Q₇→KV₇  Q₈→KV₈
      每个 Q head 对应独立的 KV head

MQA:  Q₁→KV₁  Q₂→KV₁  Q₃→KV₁  Q₄→KV₁  Q₅→KV₁  Q₆→KV₁  Q₇→KV₁  Q₈→KV₁
      所有 Q head 共享同一个 KV head

GQA:  Q₁→KV₁  Q₂→KV₁  Q₃→KV₁  Q₄→KV₁  Q₅→KV₂  Q₆→KV₂  Q₇→KV₂  Q₈→KV₂
      每组 Q heads 共享一个 KV head (这里 num_kv_heads=2)
```

---

## 2. MHA — Multi-Head Attention

### 核心公式

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_H)\; W^O$$

$$\text{head}_i = \text{Attention}(X W_i^Q,\; X W_i^K,\; X W_i^V)$$

### 关键点

- 每个 head 拥有**独立的** $W_i^Q$、$W_i^K$、$W_i^V$ 投影参数
- 每个 head 学习不同的注意力模式（语法关系、语义相似度等）
- KV cache 大小 = $H \times d_k \times 2$（K 和 V 各一份），随 head 数线性增长

### 实现

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)           # 全量 Q
        self.W_k = nn.Linear(d_model, d_model)           # 全量 K
        self.W_v = nn.Linear(d_model, d_model)           # 全量 V
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)

        out = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)
```

### Shape 追踪（d_model=32, num_heads=8, seq_len=6, batch=2）

```
x:              (2, 6, 32)
W_q(x):         (2, 6, 32) → view → (2, 6, 8, 4) → transpose → (2, 8, 6, 4)
W_k(x):         同上 → (2, 8, 6, 4)
W_v(x):         同上 → (2, 8, 6, 4)
q @ k.T:        (2, 8, 6, 4) × (2, 8, 4, 6) → (2, 8, 6, 6)   # scores
softmax @ v:    (2, 8, 6, 6) × (2, 8, 6, 4) → (2, 8, 6, 4)   # attn
transpose+view: (2, 6, 32)
W_o:            (2, 6, 32)
```

---

## 3. MQA — Multi-Query Attention

> 论文：*Fast Transformer Decoding: One Write-Head is All You Need*（Noam Shazeer, 2019）

### 核心思想

保留多个 Q head，但**所有 Q head 共享唯一一组 K 和 V**。

$$\text{head}_i = \text{Attention}(X W_i^Q,\; X W^K,\; X W^V)$$

注意 $W^K$ 和 $W^V$ 不再有下标 $i$——所有 head 共用。

### 关键点

- KV cache 大小从 $H \times d_k$ 降到 $1 \times d_k$，推理时显存减少 $H$ 倍
- KV 投影参数量减少：从 `2 × d_model × d_model` 降到 `2 × d_model × d_k`
- 训练质量有一定损失——所有 head 被迫用同一个 KV 表示，表达力受限

### 实现

```python
class MultiQueryAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)           # 全量 Q
        self.W_k = nn.Linear(d_model, self.d_k)          # 只有 1 个 KV head
        self.W_v = nn.Linear(d_model, self.d_k)          # 只有 1 个 KV head
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        # K/V: (B, S, d_k) → (B, 1, S, d_k) → repeat → (B, num_heads, S, d_k)
        k = self.W_k(x).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        v = self.W_v(x).unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)

        out = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)
```

### Shape 追踪（d_model=32, num_heads=8, seq_len=6, batch=2）

```
x:              (2, 6, 32)
W_q(x):         (2, 6, 32) → view → (2, 6, 8, 4) → transpose → (2, 8, 6, 4)
W_k(x):         (2, 6, 4) → unsqueeze → (2, 1, 6, 4) → expand → (2, 8, 6, 4)
W_v(x):         同上 → (2, 8, 6, 4)
                ↑ 注意 K/V 是从 1 个 head 广播出来的，不是独立的 8 个
q @ k.T:        (2, 8, 6, 6)
softmax @ v:    (2, 8, 6, 4)
transpose+view: (2, 6, 32)
W_o:            (2, 6, 32)
```

---

## 4. GQA — Grouped Query Attention

> 论文：*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*（Ainslie et al., 2023）
>
> 应用：LLaMA 2 70B, Mistral 7B, Gemma, etc.

### 核心思想

将 Q heads 分成 $G$ 组，每组共享一个 KV head。是 MHA 和 MQA 之间的折中。

$$\text{head}_i = \text{Attention}(X W_i^Q,\; X W_{g(i)}^K,\; X W_{g(i)}^V)$$

其中 $g(i) = \lfloor i \times G / H \rfloor$ 表示第 $i$ 个 Q head 对应的 KV head 编号。

### 关键点

- KV cache 大小 = $G \times d_k$，在 MHA 和 MQA 之间可调
- 当 $G = H$ 时退化为 MHA；当 $G = 1$ 时退化为 MQA
- 既显著减少 KV cache，又保留了大部分 MHA 的表达能力
- 实现上通过 `repeat_interleave` 把 $G$ 个 KV head 扩展到 $H$ 个

### 实现

```python
class GroupQueryAttention:
    def __init__(self, d_model, num_heads, num_kv_heads):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)                     # 全量 Q
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)     # 缩减 K
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)     # 缩减 V
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)

        # 把 num_kv_heads 个 KV head 重复扩展到 num_heads 个
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)

        out = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)
```

### Shape 追踪（d_model=32, num_heads=8, num_kv_heads=2, seq_len=6, batch=2）

```
x:                 (2, 6, 32)
W_q(x):            (2, 6, 32) → view+transpose → (2, 8, 6, 4)     # 8 个 Q heads
W_k(x):            (2, 6, 8)  → view+transpose → (2, 2, 6, 4)     # 只有 2 个 KV heads
W_v(x):            同上 → (2, 2, 6, 4)
repeat_interleave: (2, 2, 6, 4) → (2, 8, 6, 4)                    # 复制到 8 个
                   [kv₀, kv₁] → [kv₀, kv₀, kv₀, kv₀, kv₁, kv₁, kv₁, kv₁]
q @ k.T:           (2, 8, 6, 6)
softmax @ v:       (2, 8, 6, 4)
transpose+view:    (2, 6, 32)
W_o:               (2, 6, 32)
```

---

## 5. 参数量与 KV Cache 对比

以 `d_model=4096, num_heads=32, d_k=128` 为例：

| | MHA | GQA (G=8) | MQA |
|---|---|---|---|
| `W_q` 参数 | 4096 × 4096 | 4096 × 4096 | 4096 × 4096 |
| `W_k` 参数 | 4096 × 4096 | 4096 × 1024 | 4096 × 128 |
| `W_v` 参数 | 4096 × 4096 | 4096 × 1024 | 4096 × 128 |
| KV cache/token | 32 × 128 × 2 = **8192** | 8 × 128 × 2 = **2048** | 1 × 128 × 2 = **256** |
| KV cache 缩减 | 1× | **4×** | **32×** |
| 质量 | 最高 | 接近 MHA | 有损失 |

---

## 6. 为什么 GQA 能省显存但几乎不损失质量？

1. **KV 表示的冗余性**：实验表明，MHA 中不同 head 的 K/V 表示高度相似。让相邻 head 共享 KV 并不会显著损失信息。

2. **Q 保持完整**：每个 Q head 仍然有独立的投影参数，保留了"从不同角度提问"的能力。质量损失主要来自"答案"（K/V）的多样性下降。

3. **从 MHA 初始化**：GQA 论文提出可以从已训练好的 MHA 模型"uptrain"到 GQA，方法是把每组内的 KV head 参数取平均作为初始值，只需少量额外训练即可恢复质量。

---

## 7. 实际模型中的配置

| 模型 | 注意力类型 | num_heads | num_kv_heads | 每组 Q heads |
|------|-----------|-----------|-------------|-------------|
| GPT-3 | MHA | 96 | 96 | 1 |
| PaLM | MQA | 16 | 1 | 16 |
| LLaMA 1 | MHA | 32 | 32 | 1 |
| LLaMA 2 (70B) | GQA | 64 | 8 | 8 |
| Mistral 7B | GQA | 32 | 8 | 4 |
| Gemma 7B | GQA | 16 | 16→1（各层不同）| varies |

---

## 8. `repeat_interleave` vs `expand` vs `repeat`

GQA 实现中的 KV 扩展有多种写法，理解它们的区别很重要：

```python
# repeat_interleave: 每个元素原地重复
# [A, B] → [A, A, A, A, B, B, B, B]  ✅ GQA 用这个
k.repeat_interleave(4, dim=1)

# repeat: 整个张量平铺重复
# [A, B] → [A, B, A, B, A, B, A, B]  ❌ 分组对应关系错了
k.repeat(1, 4, 1, 1)

# expand: 不复制数据，只改 stride（零拷贝广播）
# 只能在 size=1 的维度上扩展
# 对于 num_kv_heads > 1 的 GQA 不能直接用
k.expand(-1, num_heads, -1, -1)  # ❌ 只有 num_kv_heads=1 时才行
```

`repeat_interleave` 保证了第 0~3 个 Q head 共享 KV head 0，第 4~7 个共享 KV head 1，分组关系正确。

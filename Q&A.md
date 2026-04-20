# PyTorch Q&A：统计量与维度读取

## Part A：std、var、unbiased

### Q1：torch.std 和 torch.var 的区别是什么？

- `torch.var` 返回方差（variance）。
- `torch.std` 返回标准差（standard deviation）。

两者关系：

$$
	ext{std} = \sqrt{\text{var}}
$$

它们都描述离散程度，只是尺度不同。

### Q2：为什么 `x.std(dim=-1, keepdim=True) ** 2 != x.var(dim=-1, keepdim=True)`？

常见原因有两个：

1. 浮点数误差

`std` 先开方再平方。浮点运算中，$\left(\sqrt{a}\right)^2$ 不保证逐位还原为 $a$。

2. `==` 是严格相等

`==` 要求完全一致，哪怕只有 `1e-7` 级别的误差也会得到 `False`。

更合理的比较方式：

```python
torch.allclose(
    x.std(dim=-1, keepdim=True) ** 2,
    x.var(dim=-1, keepdim=True),
    atol=1e-6,
)
```

还要确保两边方差定义一致（`unbiased` 或 `correction` 参数一致）。否则差异可能不只是数值误差。

### Q3：`unbiased` 参数是什么意思？

`unbiased` 决定方差分母用 $N$ 还是 $N-1$。

`unbiased=False`（总体方差）：

$$
\mathrm{Var}(x)=\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2
$$

`unbiased=True`（样本方差，Bessel 校正）：

$$
\mathrm{Var}(x)=\frac{1}{N-1}\sum_{i=1}^{N}(x_i-\mu)^2
$$

`unbiased=True` 的含义是：当当前数据被视为总体样本时，用 $N-1$ 可得到对总体方差的无偏估计。

深度学习中的常见实践：

- 归一化层实现（如 LayerNorm）通常用 `unbiased=False`。
- 统计估计场景更常用 `unbiased=True`。

补充：新版 PyTorch 常用 `correction` 替代 `unbiased`。

- `unbiased=True` 等价于 `correction=1`
- `unbiased=False` 等价于 `correction=0`

## Part B：Q.shape[-1] 与 Q.size(-1)

### Q4：`Q.shape[-1]` 和 `Q.size(-1)` 有什么区别？哪种更高效？

在结果上两者基本等价，都会返回最后一维长度。

- `Q.shape[-1]`：先取 `shape`（`torch.Size`），再索引。
- `Q.size(-1)`：直接按维度取大小。

性能上差异极小，通常可以忽略：

- `Q.size(-1)` 可能在实现路径上略直接。
- `Q.shape[-1]` 更接近 NumPy 习惯，阅读更直观。

在训练和推理中，这种差异远小于矩阵乘法、softmax 等算子耗时，不会成为性能瓶颈。

实践建议：

- 优先保持团队风格统一。
- 教学代码可优先用 `Q.shape[-1]`。
- 偏 PyTorch 接口风格的代码可优先用 `Q.size(-1)`。

## Part C：BatchNorm 中 running stats 与 no_grad

### Q5：`running_mean = (1 - momentum) * running_mean + momentum * batch_mean` 和 `running_mean.mul_(1 - momentum).add_(momentum * batch_mean)` 有什么区别？

两者数学公式相同，但行为不同。

- 前者是重绑定变量名，通常会创建新张量。
- 后者是原地更新（in-place），直接改写原张量内容。

在 BatchNorm 里，`running_mean` / `running_var` 通常是 buffer，需要被真实更新给外部状态使用，所以更推荐原地更新写法。

### Q6：`batch_mean.detach()` 的作用是什么？

`detach()` 会把 `batch_mean` 从当前计算图中分离出来，得到不跟踪梯度的张量。

在 running stats 更新中，这样做的意义是：

- 明确 running 统计量更新不参与反向传播。
- 减少不必要的计算图追踪和显存占用。
- 让语义更清晰：这是“状态更新”，不是“可学习路径”。

### Q7：BatchNorm 的 `no_grad` 实现思路是什么？

核心思路是把 running stats 的更新放在 `torch.no_grad()` 里，而归一化和仿射变换部分仍然保持可导。

示例：

```python
def my_batch_norm_no_grad(
    x,
    gamma,
    beta,
    running_mean,
    running_var,
    eps=1e-5,
    momentum=0.1,
    training=True,
):
    if training:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(momentum * batch_mean)
            running_var.mul_(1 - momentum).add_(momentum * batch_var)

        x_hat = (x - batch_mean) / torch.sqrt(batch_var + eps)
    else:
        x_hat = (x - running_mean) / torch.sqrt(running_var + eps)

    return gamma * x_hat + beta
```

### Q8：这个简化版和 PyTorch 官方 BatchNorm 的主要差异是什么？

主要差异包括：

- running variance 的估计与更新细节更复杂（官方实现有更细粒度处理）。
- `momentum` 语义更完整（支持 `momentum=None` 的累计平均逻辑）。
- 支持 `affine`、`track_running_stats` 等更多模式组合。
- 支持更广泛输入形状（如 1D/2D/3D BatchNorm 的通道归一化场景）。
- 官方实现走底层优化内核路径，数值稳定性和性能更强。

因此，简化版很适合教学和原理理解，官方版本更适合生产训练。

## Part D：unsqueeze 与 Sliding Window 掩码

### Q9：`unsqueeze` 函数到底在做什么？

`unsqueeze(dim)` 会在指定位置插入一个长度为 1 的新维度，数据本身不变，只改变张量形状。

例子：

```python
x = torch.tensor([10, 20, 30])  # shape: (3,)

x0 = x.unsqueeze(0)  # shape: (1, 3)
x1 = x.unsqueeze(1)  # shape: (3, 1)
```

可以把它理解成“给张量补一个壳”，目的是让维度对齐，方便广播、矩阵运算和 batch 处理。

### Q10：下面这几行 Sliding Window 代码在做什么？

```python
i = torch.arange(seq_len).unsqueeze(1)
j = torch.arange(seq_len).unsqueeze(0)
mask = (i - j).abs() > window_size
scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
```

这段代码的目标是构造“局部注意力掩码”：每个位置只能关注窗口内的 token。

逐行解释：

1. `i = torch.arange(seq_len).unsqueeze(1)`

- 先得到索引 `[0, 1, ..., seq_len - 1]`。
- `unsqueeze(1)` 后形状是 `(seq_len, 1)`，表示“行索引（查询位置）”。

2. `j = torch.arange(seq_len).unsqueeze(0)`

- 同样得到索引 `[0, 1, ..., seq_len - 1]`。
- `unsqueeze(0)` 后形状是 `(1, seq_len)`，表示“列索引（键位置）”。

3. `mask = (i - j).abs() > window_size`

- 利用广播，`i - j` 形成 `(seq_len, seq_len)` 的距离矩阵。
- 第 `(r, c)` 个元素是 `|r - c|`。
- 距离大于 `window_size` 的位置标记为 `True`（需要屏蔽）。

4. `scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))`

- `mask.unsqueeze(0)` 变成 `(1, seq_len, seq_len)`，可按 batch 维广播到 `scores`（通常是 `(batch, seq_len, seq_len)`）。
- 把需要屏蔽的位置填成 `-inf`。
- 后续 `softmax` 时，`exp(-inf)=0`，这些位置的注意力权重就变成 0。

结论：这 4 行代码本质上是在构造并应用一个“带状掩码”，只保留对角线附近窗口内的注意力连接。
## Part E：torch.matmul 与 torch.bmm

### Q11：`torch.matmul` 和 `torch.bmm` 的区别是什么？

**`torch.bmm`** 专用于 3D 张量的批量矩阵乘法：

- 输入必须严格是 `(batch, n, m)` 和 `(batch, m, p)`
- 输出 `(batch, n, p)`
- 不支持广播

**`torch.matmul`** 是通用矩阵乘法：

- 支持任意维度（1D 点积、2D 矩阵乘、3D+ 批量乘）
- 支持广播，例如 `(batch, n, m)` × `(m, p)` → `(batch, n, p)`
- 是 `@` 运算符的底层实现

示例对比：

```python
a = torch.randn(4, 8, 64)
b = torch.randn(4, 64, 32)

torch.bmm(a, b)      # ✅ (4, 8, 32)
torch.matmul(a, b)   # ✅ (4, 8, 32) — 等价

# bmm 不支持广播：
w = torch.randn(64, 32)
torch.bmm(a, w)      # ❌ 报错，维度必须是 3D
torch.matmul(a, w)   # ✅ (4, 8, 32) — 广播成功
```

实践建议：

- `torch.matmul`（或 `@`）基本可以替代 `torch.bmm`，且更灵活。
- `torch.bmm` 语义更明确，在确知输入是 3D 批量时，可读性更好。
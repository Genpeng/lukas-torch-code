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

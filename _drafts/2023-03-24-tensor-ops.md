---
layout: post
title: "(WIP) Tensor Ops"
date: 2023-03-24 14:31:04 +0800
labels: [pytorch, numpy]
---

## 动机、参考资料、涉及内容

动机：

- 在深入阅读一些深度学习模型代码时，会遇到一些关于 `Tensor` 的下标处理操作，例如：huggingface transformers 中的 `generation/logit_process.py:RepetitionPenaltyLogitsProcess`。
- numpy 的一些下标操作

参考资料：

- pytorch 官方文档
- numpy 官方文档
- 一些博客

涉及内容:
- API 解释
- 一些综合使用的例子


## torch.gather

```python
import torch
torch.gather(input, dim: int, index)
```

其中 input 的形状假设是 (4, 3, 6)，而 dim=1，index 的形状必须为 (4, K, 6)，其中 K 可以任取（因为 dim=1）。而输出的 tensor 的形状与 index 完全一致，即：(4, K, 6)。

具体的例子与解释参考[博客](https://medium.com/@mbednarski/understanding-indexing-with-pytorch-gather-33717a84ebc4).

## torch.scatter

## repeat, repeat_interleave, squeeze, unsqueeze

## broadcast_to, expand

`torch.broadcast_to` 与 `torch.expand` 完全等价, 注意 `torch.repeat` 会发生内存拷贝, 而 `torch.expand` 不会, 这两者的使用方式也不一样(适用于 `torch.expand` 的入参可能不适用 `torch.repeat`, 反之亦然)

## 索引与切片

总的来说, 有如下几类

```python
t[None]
t[:]
t[0, ...]
t[:3], t[-3:], t[2:5], t[start:end:step]
t[torch.tensor([True, False])]
t[torch.tensor([1, 2, 4])], t[torch.tensor([[1, 2, 4], [0, 1, 3]])]
```

pytorch 的文档中似乎对各种切片操作没有仔细介绍, 但应该基本上与 numpy.ndarray 的用法相似, 因此可以参照 numpy 的[文档](https://numpy.org/doc/stable/user/basics.indexing.html), 在姑且不论 view 与 copy 的区别时

首先引用 numpy 文档中的一段提示

> Note that in Python, x[(exp1, exp2, ..., expN)] is equivalent to x[exp1, exp2, ..., expN]; the latter is just syntactic sugar for the former.

如果使用 bool 数组的方式进行索引时, 可以理解成将该位置的 bool 数组转化为列表

```python
a[torch.tensor([True, False, True, False]), :]  # 等价于 a[[0, 2], :]
a[[0, 2, 1], [2, 3, 1]]  # 假设 a 只有两维, 注意返回值为 torch.tensor([a[0, 2], a[2, 3], a[1, 1]])
```
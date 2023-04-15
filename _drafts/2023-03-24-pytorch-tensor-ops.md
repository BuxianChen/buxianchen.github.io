---
layout: post
title: "(WIP) Pytorch Tensor Op"
date: 2023-03-24 14:31:04 +0800
labels: [pytorch]
---

## 动机、参考资料、涉及内容

动机：

- 在深入阅读一些深度学习模型代码时，会遇到一些关于 `Tensor` 的下标处理操作，例如：huggingface transformers 中的 `generation/logit_process.py:RepetitionPenaltyLogitsProcess`。

参考资料：

- pytorch 官方文档
- 一些博客

涉及内容与不涉及内容：

待定

## torch.gather

```python
import torch
torch.gather(input, dim: int, index)
```

其中 input 的形状假设是 (4, 3, 6)，而 dim=1，index 的形状必须为 (4, K, 6)，其中 K 可以任取（因为 dim=1）。而输出的 tensor 的形状与 index 完全一致，即：(4, K, 6)。

具体的例子与解释参考[博客](https://medium.com/@mbednarski/understanding-indexing-with-pytorch-gather-33717a84ebc4).

## torch.scatter

## slice(None), repeat, repeat_interleave, squeeze, unsqueeze
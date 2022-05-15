---
layout: post
title:  "神经网络基本单元"
date: 2022-02-11 22:31:04 +0800
---

## CTCLoss

手动实现 `CTCLoss` 的前向过程：参考[blog](https://distill.pub/2017/ctc/)
```python
import numpy as np
def ctc_manual(P, y):
    # P (float): (V, T), V为字母表的大小(含<blank>), T为输入序列的长度, 且P的每列列和为1
    # y (int): L, L为输出序列的长度, 每个元素的取值为[1, V-1]
    z = [0]*(len(y)*2+1)
    for i in range(len(y)):
        z[2*i+1] = y[i]
    table = np.zeros((len(y)*2+1, P.shape[1]), dtype=float)
    table[0][0] = P[0][0]
    table[1][0] = P[z[1]][0]
    T = P.shape[1]
    S = len(z)
    for t in range(1, T):
        for s in range(S):
            if z[s] == 0 or (z[s]!=0 and s>=2 and z[s]==z[s-2]):
                table[s][t] = table[s][t-1]
                if s >= 1:
                    table[s][t] += table[s-1][t-1]
            else:
                table[s][t] = table[s][t-1]
                if s >= 1:
                    table[s][t] += table[s-1][t-1]
                if s >= 2:
                    table[s][t] += table[s-2][t-1]
            table[s][t] *= P[z[s]][t]

    prob = table[-1][-1] + table[-2][-1]
    loss = - np.log(prob) / len(y)  # torch的实现里loss需要除以输出序列的长度
    return loss
```

对比pytorch的实现
```python
import numpy as np
# P = np.array([
#     [0.2, 0.1, 0.3, 0.1, 0.6, 0.3, 0.1],
#     [0.7, 0.2, 0.1, 0.4, 0.2, 0.6, 0.8],
#     [0.1, 0.7, 0.6, 0.5, 0.2, 0.1, 0.1]
# ])
# y = [1, 1, 2]
# x_length = 7
# y_length = 3

x_length, V = 10, 5
y_length = 3
P = np.random.rand(V, x_length)
P = np.exp(P)
P =  P / np.sum(P, axis=1, keepdims=True)
y = np.random.randint(1, V, size=(y_length,))
loss_manual = ctc_manual(P, y)

import torch
loss_fn = torch.nn.CTCLoss(blank=0)
# pytorch要求输入形状为(T, B, C), 且输入为log_softmax的形式
input = torch.tensor(P, dtype=torch.float32).transpose(0, 1).unsqueeze(1).log().detach().requires_grad_()
input_lengths = torch.full(size=(1,), fill_value=x_length, dtype=torch.long)
target =  torch.tensor([y], dtype=torch.long)
target_lengths = torch.full(size=(1,), fill_value=y_length, dtype=torch.long)
loss = loss_fn(input, target, input_lengths, target_lengths).item()

# loss_manual == loss  # yes
```
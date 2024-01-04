---
layout: post
title: "(P1) NLP Model Architecture Examples"
date: 2023-06-11 17:20:04 +0800
labels: [transformers]
---

## 动机、参考资料、涉及内容

动机

- 梳理目前较为经典的 NLP 模型的结构 (在原始 Transformers 结构上的微小改动的部分), 也为了方便模型结构/规模对推理性能的影响

参考资料

- 🤗 Transformer 的实现

涉及内容

- 模型结构
- 具体的超参数, 例如层数, 特征向量维数等
- 各个模型间的“渊源”梳理

不涉及内容

- 训练数据量, tokenizer, 训练方法 (这一部分在另一篇博客 llm-survey 中做介绍), 因此可以认为本博客只是介绍随机初始化的模型长什么样.

## 渊源

## Blocks

### RoPE 位置编码

Rotary Position Embedding 位置编码被广泛用于后续的模型结构里:

$$
\text{position\_enc} = \begin{bmatrix}
0\cdot10000^{-\frac{0}{d} }&0\cdot10000^{-\frac{2}{d} }&\cdots&0\cdot10000^{-\frac{d-2}{d} } \\
1\cdot10000^{-\frac{0}{d} }&1\cdot10000^{-\frac{2}{d} }&\cdots&1\cdot10000^{-\frac{d-2}{d} } \\
\vdots & \vdots & \ddots & \vdots\\
(L-1)\cdot10000^{-\frac{0}{d} }&(L-1)\cdot10000^{-\frac{2}{d} }&\cdots&(L-1)\cdot10000^{-\frac{d-2}{d} } \\
\end{bmatrix}
$$

- 公式参考 [作者原文公式](https://spaces.ac.cn/archives/8265)
- 实现参考: [huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py)

```python
# 使用逻辑: 每一次 attention 之前都先对 query, key, value 做旋转处理 (value 未必需要)

# ============ 旋转 ===============
# query, key, value: (B, num_head, L, head_dim)
position_enc = torch.tensor([[i * 10000 ** (-2*j/head_dim) for j in range(head_dim//2)] for i in range(L)])  # (L, head_dim // 2)
# sin: (L, head_dim // 2), cos: (L, head_dim // 2)
sin, cos = torch.sin(position_enc), torch.cos(position_enc)

# sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
# cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
# rotate_half_query [-q1,q0,-q3,q2......,-qd-1,qd-2]
rotate_half_query = torch.stack([-query[..., 1::2], query[..., ::2]], dim=-1).reshape_as(query)
query = query * cos_pos + rotate_half_query * sin_pos
# apply to key, value

# ========= 普通的 dot-product attention ==========
score = torch.matmul(query_layer, key_layer.transpose(-1, -2))
# ...
```

### RMSNorm

以下模型均采用了 RMSNorm:

- T5
- Llama

LayerNorm: 对于一个输入形状为 `(B, L, C)` 的输入, 对最后一维做减均值除方差的操作, 数学表述如下: $\mathbf{x}$ 为一个 $n$ 维向量, LayerNorm 的操作如下:

$$
\begin{align*}
\bar{x}&=\text{Mean}(\mathbf{x}) \\
\text{Var}(\mathbf{x}) &= \frac{\sum_{i=1}^{n}{(x_i - \bar{x})^2} }{n} \\
\hat{x}_i &= \gamma * \frac{x_i-\text{Mean}(x)}{\sqrt{\text{Var}(\mathbf{x})+\epsilon} } + \beta
\end{align*}
$$

RMSNorm 的数学表示如下:

$$
\begin{align*}
\text{RMS}(\mathbf{x}) &= \frac{\sum_{i=1}^{n}{x_i^2} }{n} \\
\hat{x}_i &= \gamma * \frac{x_i}{\sqrt{\text{RMS}(\mathbf{x}) + \epsilon} }
\end{align*}
$$

### Multi-Query Attention

以下模型采用了 Multi-Query Attention

- chatglm2

```python
# x: (B, L, D), D = head_dim * num_heads
# qkv_layer: Linear(in_features=D, out_features=D+2*head_dim)
q, k, v = qkv_layer(x).split([D, head_dim, head_dim], dim=-1)

q = q.reshape(B, L, num_head, head_dim).transpose(1, 2)  # (B, num_head, L, head_dim)
k = k.reshape(B, L, 1, head_dim).transpose(1, 2)         # (B, 1, L, head_dim)
v = v.reshape(B, L, 1, head_dim).transpose(1, 2)         # (B, 1, L, head_dim)

scores = q @ k.transpose(-1, -2)                         # scores: (B, num_head, L, L)
weight = torch.nn.softmax(scores, dim=-1)
y = weight @ v  # (B, num_head, L, L) x (B, 1, L, head_dim) = (B, num_head, L, head_dim)
y = y.transpose(1, 2).reshape(B, L, D)
```

主要的加速在于两点(个人观点, 还需找原论文核对):

- `qkv_layer` 涉及的矩阵运算的计算量减少了
- `scores = q @ k.transpose(-1, -2)` 涉及的运算量不变, 但在推理时, `q` 对应的 `L=1`, 而 `k` 对应的 `L` 会持续增长, 因此内存带宽占用为 $O(B\times D+B\times L \times head\_dim)$, 其中第一项是 `q` 的元素个数, 第二项是 `k` 的元素个数, 而原始的 attention 在此处的内存带宽占用量为 $O(B\times D+B\times L \times D)$
- `y = weight @ v` 同理, 涉及的运算量不变, 在推理时, 内存带宽占用为 $O(B\times num\_head \times L+B\times num\_head \times L)$, 而原始的 attention 在此处的内存带宽占用量为 $O(B\times num\_head \times L+B\times D \times L)$

从原始论文的实验结果看, 相比于原始的 attention 机制, 使用 Multi-Query Attention 可能会对推理速度提升 5-10 倍, 但训练速度提升几乎可以忽略不计.


## GPT-2

```yaml

GPT2Model
  - wte: nn.Embedding
  - wpe: nn.Embedding
  - drop: nn.Dropout
  - h: nn.ModuleList
    - block: GPT2Block
        - ln_1: nn.LayerNorm
        - attn: GPT2Attention
        - ln_2: nn.LayerNorm
        - mlp: nn.ModuleList([linear(C, 4C), gelu, linear(4C, C), dropout])
    - block: GPT2Block
    - ...
  - ln_f: nn.LayerNorm
```

```python
class GPT2Attention:
    def forward(self, hidden_states):
        # (B, L, C) -> (B, L, 3C) -> split
        q, k, v = self.linear(hidden_states).split(C, dim=2)
        # (B, L, C) -> (B, num_head, L, head_dim)
        q, k, v = transpose(q), transpose(k), transpose(v)

        # (B, num_head, L, L)
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
        # mask: (B, 1, 1, L) = (1.0 - attention_mask) * torch.finfo(dtype).min
        attn_weights += mask
        attn_weights = dropout(softmax(attn_weights, dim=-1))
        attn_output = torch.matmul(attn_weights, v)  # (B, num_head, L, head_dim)
        
        hidden_states = transpose(attn_output)  # (B, L, C)
        hidden_states = dropout(self.out_linear(hidden_state))
        k, v = concat(past_k, k), concat(past_v, v)
        return hidden_states, (k, v), attn_weights

GPT2MLP: nn.ModuleList([nn.Linear(C, 4*C), gelu, nn.Linear(4*C, C), dropout])

class GPT2Block:
    def forward(self, hidden_states):
        # hidden_states: (B, L, C)
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)  # (B, L, C) -> (B, L, C)
        # (B, L, C), ((B, num_head, L, head_dim), (B, num_head, L, head_dim)), (B, num_head, L, L)
        (hidden_states, (key, value), attn_weights) = self.attn(hidden_states, ...)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states, (key, value), attn_weights

class GPT2Model:
    def forward(self, input_ids, attention_mask):
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)  # position_ids: range(0, L)
        hidden_states = dropout(inputs_embeds + position_embeds)
        for block in self.h:
            hidden_states, (key, value), attn_weights = block(hidden_states)
        last_hidden_states = self.ln_f(hidden_states)  # layernorm
        return (
            last_hidden_states,
            [(key, value), (key, value), ...],
            all_hidden_states,  # list
            attentions  # list of attn_weights
        )
```

## GPT-J

## GPT-Neo/GPT-NeoX

## LLAMA

## Open-LLAMA

## RWKV

## GLM

- 归一化: RMSNorm
- 激活函数:
- 位置编码: RoPE
- 注意力: 采用 Multi-Query Attention, 实现上还利用了 flashattention-v1

## MOSS

基本上完全就是 GPTJ 的结构?

- 归一化: RMSNorm
- 激活函数:
- 位置编码: RoPE
- 注意力: 普通的自注意力

## internlm

- 归一化: RMSNorm
- 激活函数:
- 位置编码: RoPE(可能有一些修改?)
- 注意力: 普通的自注意力
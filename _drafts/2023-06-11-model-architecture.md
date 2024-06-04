---
layout: post
title: "(P1) Transformers Building Blocks"
date: 2023-06-11 17:20:04 +0800
labels: [transformers]
---

## 动机、参考资料、涉及内容

动机: 梳理 Transformers 的一些关键组件, 例如 layernorm, 位置编码, attention 等. 并以此作为基础从模型结构上看看各类开源模型的模型结构

参考资料: 主要参考 huggingface transformers 的实现

## Building Blocks

### Scaled Dot Product Attention

Attention 机制是为了解决一个这样的问题, 假设现在有很多信息来源, 从人的思维来说, 总是会关注重点, 当然也考虑次要信息, 最终汇总出一个结果. 在 NLP 的场景下, 这里的很多信息来源指的就是现在每个 token 我都有了一个隐层表示, 也就是说 $L$ 个 $D$ 维向量, 而汇总起来的意思就是把这些总结为 1 个 $C$ 维的向量. 那么做到这的自然的想法是什么呢? 最自然的一个想法是平均起来然后再做个线性变换 (其实也等价于先做线性变换再平均):

$$
\mathbf{y}=\frac{1}{L}A\sum_{i}^{L}{\mathbf{x}_l}
$$

其中 $\mathbf{y}\in\mathbb{R}^{C}$, 而 $A\in\mathbb{R}^{C\times D}$. 接下来我们很自然地会想到, 平均可能是不合理的, 因此我们很自然地希望将其更改为

$$
\mathbf{y}=A\sum_{i}^{L}{\alpha_l\mathbf{x}_l}
$$

其中 $\alpha_l$ 是一个标量, 且 $\sum_{i=1}^{L}\alpha_l=1$, 现在有个问题是这些 $\alpha_l$ 怎么确定呢? 回到人的思维角度, 我们通常会带着一个问题来汇总相关的信息, 也就是说会用一个 $\mathbf{q}\in\mathbb{R}^E$ 来审视 $\mathbf{x}_1,\ldots\mathbf{x}_L$, 判断这些信息是否与 $\mathbf{q}$ 相关. 最直接的做法当然是假设 $E=D$, 用内积来衡量相关性, 即:

$$
\alpha_l=\langle\mathbf{q},\mathbf{x}_l\rangle
$$

当然, 这里的小细节是我们无法保证 $\sum_{l=1}^{L}{\alpha_l}=1$, 这实际上可以再对 $\alpha_l$ 做一次 softmax 即可完成.

然后我们看一下 Scaled Dot Product Attention 的定义: 假设有 3 个矩阵 $Q,K,V\in\mathbb{R}^{L\times D}$, 采用注意力机制后得到的表示 $Y\in\mathbb{R}^{L\times D}$ 为:

$$
Y=\text{softmax}(\frac{QK^T}{\sqrt{D}})V
$$

其中 $\text{softmax}$ 是对矩阵的每一行做归一化处理, 我们先忽略掉缩放因子 $\frac{1}{\sqrt{D}}$, 注意到这里似乎与之前的讨论稍有不同: 之前我们是希望用 $L$ 个向量汇总成一个向量, 而这里却仍旧是 $L$ 个向量. 这个区别应当这样理解, 对于每个特定的 $1 \le l \le L$ 来说, 我们希望汇总所有的 $L$ 个向量, 得到一个新的汇总表示, 也就是 $Y$ 的第 $l$ 行, 而如果我们只关注 $Y$ 的第 $l$ 行, 我们会发现, 上面的式子实际上是:

$$
\alpha'_i = \frac{\langle\mathbf{q}_l, \mathbf{k}_i\rangle}{\sqrt{D}}\\
\alpha_i=[\text{softmax}(\alpha'_1,\ldots,\alpha'_L)]_i\\
\mathbf{y}_l=\sum_{i=1}^{L}{\alpha_i\mathbf{v}_i}
$$

而 transformers 中的所谓 self-attention, 其实就只是在此基础上增加一条: $Q,K,V$ 全部来自于上一层的表示 $X\in\mathbb{R}^{L\times D}$, 其实也就是 $Q,K,V$ 是通过 $X$ 线性变换得到的.

最后 transformers 中 attention 的“最终形态” multi-head self-attention, 实际上则更加简单粗暴: 在通过线性变换得到 $Q,K,V$ 后, 本来各自都是 $L\times D$ 的形状, 现在直接将其切分为 $\text{heads}$ 段, 然后每一段各自做 Scaled Dot Product Attention, 这样每一段输出的 $Y$ 也都只有 $D/\text{heads}$ 的长度, 而最终的 $Y$ 就将他们直接拼起来即可. 因此, Scaled Dot Product Attention 用代码实现如下

```python
# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # 备注, 一般地形状如下, 但在 transformer 中, L=S 为序列长度, c=d, num_heads*d 为隐层的维度
    # query: (B, num_heads, L, c)
    # key:   (B, num_heads, S, c)
    # value: (B, num_heads, S, d)
    # output:(B, num_heads, L, d)
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    # ==========================
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    # ===========================
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
```

### RoPE 位置编码 (TODO)

- 公式参考 [作者原文公式](https://spaces.ac.cn/archives/8265)
- 实现参考: [huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py)


笔者按照自己的思路讲述 RoPE 编码的来由, 我们考虑 self-attention 的情形, 假设对于原始序列送入 Embedding 层后, 不加上位置编码, 我们应该怎样把位置信息考虑进去?

```python
# 假设词表长度为 4, 词嵌入的维度为 2
embedding_map = [[0.1, 0.2], [0.1, 0.3], [-0.1, 0.2], [-0.1, 0.1]]
input_ids = [0, 1, 0, 2, 0, 1]
embeddings = [[0.1, 0.2], [0.1, 0.3], [0.1, 0.2], [-0.1, 0.2], [0.1, 0.2], [0.1, 0.3]]

Q = embeddings @ q_trans
K = embeddings @ k_trans
V = embeddings @ v_trans

# 如果使用标准的 Y = softmax(Q@K.T)V, 那么在这个例子中, Y[0] == Y[2] == Y[4], Y[1]==Y[5]
```

我们考虑按这种方式加位置编码, 假设我们通过一个函数 $f(\mathbf{x}, m)\in\mathbb{R}^2$ 来增加位置编码, 其中 $\mathbf{x}\in\mathbb{R}^2$, 而 $m\geq 0$ 且为整数. 我们希望对前面的 $Q$ 和 $K$ 使用这一变换, 满足:

$$
\langle f(\mathbf{q}, m), f(\mathbf{k}, n)\rangle = g(\mathbf{q}, \mathbf{k}, m-n)
$$

首先, 我们为什么希望有一个这样的式子成立呢? 这个式子表明的含义是内积计算结果只与 $\mathbf{q}, \mathbf{k}$ 以及 $m-n$ (相对位置的距离) 相关 (重点在于固定 $\mathbf{q}$ 和 $\mathbf{k}$ 时, 仅与相对距离相关, 而不是关于 $m$ 以及 $n$ 更为复杂的关系).

这里我们先看最终的答案:

$$
f(\mathbf{q}, m)=\begin{bmatrix}
\cos(m\theta)&-\sin(m\theta)\\
\sin(m\theta)&\cos(m\theta)\\
\end{bmatrix}
\begin{bmatrix}
q_1\\
q_2
\end{bmatrix}
$$

那么

$$
\langle f(\mathbf{q}, m), f(\mathbf{k}, n)\rangle=\mathbf{q}^T\begin{bmatrix}
\cos(m\theta)&-\sin(m\theta)\\
\sin(m\theta)&\cos(m\theta)\\
\end{bmatrix}^T
\begin{bmatrix}
\cos(n\theta)&-\sin(n\theta)\\
\sin(n\theta)&\cos(n\theta)\\
\end{bmatrix}\mathbf{k}^T=\mathbf{q}^T\begin{bmatrix}
\cos((m-n)\theta)&\sin((m-n)\theta)\\
-\sin((m-n)\theta)&\cos((m-n)\theta)\\
\end{bmatrix}\mathbf{k}^T
$$

上述计算利用到了三角公式

$$
\sin(\alpha+\beta)=\sin\alpha\cos\beta+\cos\alpha\sin\beta \\
\sin(\alpha-\beta)=\sin\alpha\cos\beta-\cos\alpha\sin\beta \\
\cos(\alpha+\beta)=\cos\alpha\cos\beta-\sin\alpha\sin\beta \\
\cos(\alpha-\beta)=\cos\alpha\cos\beta+\sin\alpha\sin\beta
$$

我们很容易将其推广至高维: [https://spaces.ac.cn/archives/8265](https://spaces.ac.cn/archives/8265) 的第 11 个式子

推导过程 (TODO):

```python
q = (q1, q2)
k = (k1, k2)

<q, k> = (q1*k1+q2*k2)

# 将 q 和 k 看作复数, 那么 q 与 k 内积的实部与它们作为二维向量的内积相同
(q1 + q2 * i) * (k1 - k2 * i) = <q, k> + (- q1 * k2 + q2 * k1) * i
```

#### 实现

从上面可知, 假设 $\mathbf{q}$ 与 $\mathbf{k}$ 都是 $d$ 维向量, 我们可以在对角线上使用 $d/2$ 个不同的 $\theta$ 值, RoFormer 作者采用了

$$
\theta_i=10000^{-\frac{2i}{d}}, \quad i=0,\ldots,\frac{d}{2}-1
$$

由于变换 $f$ 在扩展到 $d$ 维之后变换矩阵只有对角线上有值(将变换矩阵视为 2x2 的分块矩阵, 则只有对角线上的块有值). 因此 $f$ 可以有更简单的形式, 首先回到 2 维情形:

$$
f(\mathbf{q}, m)=\begin{bmatrix}
\cos(m\theta)&-\sin(m\theta)\\
\sin(m\theta)&\cos(m\theta)\\
\end{bmatrix}
\begin{bmatrix}
q_1\\
q_2
\end{bmatrix}
=\begin{bmatrix}
q_1\cdot\cos(m\theta)-q_2\cdot\sin(m\theta)\\
q_1\cdot\sin(m\theta)+q_2\cdot\cos(m\theta)
\end{bmatrix}=
\begin{bmatrix}
\cos(m\theta)\\
\cos(m\theta)\\
\end{bmatrix}
\odot
\begin{bmatrix}
q_1\\
q_2
\end{bmatrix}
+
\begin{bmatrix}
\sin(m\theta)\\
\sin(m\theta)\\
\end{bmatrix}
\odot
\begin{bmatrix}
-q_2\\
q_1
\end{bmatrix}
$$

因此在实现上, 可以先将 $\mathbf{q}$ 保存两份: $\mathbf{q}$ 以及 $\mathbf{q}'$, 其中

$$
\mathbf{q}'=[-q_2, q_1, -q_4, q_3, ..., -q_d, q_{d-1}]^T
$$

也就是最终的计算方式是: 假设 $\mathbf{q}$ 位于第 $m$ 个 token id 的位置上,

$$
\text{rotary}(\mathbf{q}, m)=\begin{bmatrix}
\cos(m\theta_0)\\
\cos(m\theta_0)\\
\vdots\\
\cos(m\theta_{d/2-1})\\
\cos(m\theta_{d/2-1})\\
\end{bmatrix}
\odot
\mathbf{q}
+
\begin{bmatrix}
\sin(m\theta_0)\\
\sin(m\theta_0)\\
\vdots\\
\sin(m\theta_{d/2-1})\\
\sin(m\theta_{d/2-1})\\
\end{bmatrix}
\odot
\mathbf{q}'
$$

其中

$$
\theta_i=10000^{-\frac{2i}{d}}, \quad i=0,\ldots,\frac{d}{2}-1
$$

我们仔细看一下 $\theta_i$ 的定义, 首先 $\theta_i$ 随着 $i$ 的增大单调递减, 且 $0\le\theta_{d/2-q}\le\theta_i\le\theta_0= 1$

$$
\begin{align}
&d=128:\quad&\theta_0=1,&\theta_1\approx0.866,&\theta_2\approx0.750,&\ldots,&\theta_{63}=0.0001\\
&d=64:\quad&\theta_0=1,&\theta_1\approx0.750,&\theta_2\approx0.562,&\ldots,&\theta_{31}=0.0001\\
\end{align}
$$

注意, 这里看 $d=128,64$ 的原因是使用 RoPE 作为位置编码时, 会将原始的隐层维数切分 head 之后再应用 RoPE, 一个例子是 QWen-7B 模型的隐层维数是 4096, 但其 head 数是 32, 因此 head dim 为 128.

(TODO) 然后我们回过头去看完整的 attention 计算, 只考虑一个 head, 第 $m$ 个位置的隐层输出的计算公式为

$$
\text{attention}(\mathbf{x}_m)=\text{normalize}(\langle\mathbf{q}_k,\mathbf{m}_1\rangle\mathbf{v}_1+\cdots+\langle\mathbf{q}_m,\mathbf{k}_d\rangle\mathbf{v}_d)
$$

**代码实现**

$$
\text{positionenc} = \begin{bmatrix}
0\cdot10000^{-\frac{0}{d} }&0\cdot10000^{-\frac{2}{d} }&\cdots&0\cdot10000^{-\frac{d-2}{d} } \\
1\cdot10000^{-\frac{0}{d} }&1\cdot10000^{-\frac{2}{d} }&\cdots&1\cdot10000^{-\frac{d-2}{d} } \\
\vdots & \vdots & \ddots & \vdots\\
(L-1)\cdot10000^{-\frac{0}{d} }&(L-1)\cdot10000^{-\frac{2}{d} }&\cdots&(L-1)\cdot10000^{-\frac{d-2}{d} } \\
\end{bmatrix}\in\mathbb{R}^{L\times\frac{d}{2}}
$$


```python
# 使用逻辑: 每一次 attention 之前都先对 query, key, value 做旋转处理 (value 未必需要)
def apply_rotary(query):
    # ============ 旋转 ===============
    # query, key, value: (B, num_head, L, head_dim)
    head_dim = query.dim(-1)
    position_enc = torch.tensor([[i * 10000 ** (-2*j/head_dim) for j in range(head_dim//2)] for i in range(L)])  # (L, head_dim // 2)
    # sin: (L, head_dim // 2), cos: (L, head_dim // 2)
    sin, cos = torch.sin(position_enc), torch.cos(position_enc)

    # stack 后的形状是 (L, head_dim // 2, 2)
    # (0, 1, ..., d/2-1) -> (0, 0, 1, 1, ..., d/2-1, d/2-1)
    sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)

    # stack 后的形状是 (L, head_dim // 2, 2)
    # (0, 1, ..., d/2-1) -> (0, 0, 1, 1, ..., d/2-1, d/2-1)
    cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)

    # rotate_half_query:
    # stack 后
    # [(-q1, q0), (-q3, q2), ..., (-qd-1. qd-2)]
    # reshape 后, 其实也就是展平
    # [-q1,q0,-q3,q2......,-qd-1,qd-2]
    rotate_half_query = torch.stack([-query[..., 1::2], query[..., ::2]], dim=-1).reshape_as(query)
    query = query * cos_pos + rotate_half_query * sin_pos
    return query

# x: (B, num_heads, L, head_dim) 表示上一层的输入
query = query_layer(x)
key = key_layer(x)
value = value_layer(x)

query = apply_rotary(query)
key = apply_rotary(key)

# ========= 普通的 dot-product attention, 参考前面的小节 ==========
x = scaled_dot_product_attention(query, key, value)
```

#### 变体

RoPE 在实现上有两种风格, 一种是上面介绍的, 称为 GPT-J 风格, 也是原始版本, 也就是使用 $\mathbf{q}$ 的 $q_{2i}, q_{2i+1}$ 应用于 $\theta_i$ 角度的旋转. 另一种风格被称为 GPT-NeoX 风格, 也就是 $q_{i}, q_{d/2+i}$ 应用于 $\theta_i$ 角度的旋转. (这种区分方式可参考 flash-attention 的[注释](https://github.com/Dao-AILab/flash-attention/blob/v2.5.9/flash_attn/layers/rotary.py#L94), huggingface 对 [GPT-J](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/gptj/modeling_gptj.py#L84) 与 [GPT-NeoX](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L610) 的实现)

QWen-1 系列均采用 GPT-NeoX 风格的实现

#### RoPE 的长度外推方法

所谓长度外推, 是指模型训练时总会设置一个最大长度, 譬如说 2048, 而现在希望将其上下文长度扩大到 9192. 最理想的解决方案是不训练, 直接扩大到 9192 上下文做推理 (允许对模型权重或 position embedding 做一些非训练的调整), 次理想的解决方案是加入一些更长的训练数据微调, 但即使训练也应该利用已经训练好的模型参数, 而不是完全重训. 而 Transformer 架构的模型一般的外推策略只涉及到 position embedding 的特殊处理

根据作者博客 [https://kexue.fm/archives/9675](https://kexue.fm/archives/9675) 的描述, 存在多种长度外推方案, 这里只介绍其中的一种:

假设原先的最大长度为 $L$, 需要扩充 $k$ 倍, 即需要的上下文长度为 $kL$. 那么应该将 positionenc 定义为:

$$
\text{positionenc} = \begin{bmatrix}
\frac{0}{k}\cdot10000^{-\frac{0}{d} }&\frac{0}{k}\cdot10000^{-\frac{2}{d} }&\cdots&\frac{0}{k}\cdot10000^{-\frac{d-2}{d} } \\
\frac{1}{k}\cdot10000^{-\frac{0}{d} }&\frac{1}{k}\cdot10000^{-\frac{2}{d} }&\cdots&\frac{1}{k}\cdot10000^{-\frac{d-2}{d} } \\
\vdots & \vdots & \ddots & \vdots\\
\frac{(kL-1)}{k}\cdot10000^{-\frac{0}{d} }&\frac{(kL-1)}{k}\cdot10000^{-\frac{2}{d} }&\cdots&\frac{(kL-1)}{k}\cdot10000^{-\frac{d-2}{d} } \\
\end{bmatrix}\in\mathbb{R}^{kL\times\frac{d}{2}}
$$

然后直接按上面的代码进行实现即可

### Post-LayerNorm (原始 transformer) vs Pre-LayerNorm (推荐)

- 参考博客: [https://sh-tsang.medium.com/review-pre-ln-transformer-on-layer-normalization-in-the-transformer-architecture-b6c91a89e9ab](https://sh-tsang.medium.com/review-pre-ln-transformer-on-layer-normalization-in-the-transformer-architecture-b6c91a89e9ab)

```python
# Post-LayerNorm
x1 = multi_head_attention(x)
x = x + x1
x = layernorm(x)
x1 = ffn(x)
x = x + x1
x = layernorm(x)

# Pre-LayerNorm
x1 = layernorm(x)
x1 = multi_head_attention(x1)
x = x + x1
x1 = layernorm(x)
x1 = ffn(x)
x = x + x1
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


### Grouped Query Attention (GQA)

GQA 是 MQA 的变体, 采用此结构的模型:

- llama3

```python
# x: (B, L, D), D = head_dim * num_heads
# qkv_layer: Linear(in_features=D, out_features=D+2*head_dim*num_key_value_heads)
# num_key_value_groups = num_heads // num_key_value_heads
q, k, v = qkv_layer(x).split([D, head_dim*num_key_value_heads, head_dim*num_key_value_heads], dim=-1)

q = q.reshape(B, L, num_head, head_dim).transpose(1, 2)             # (B, num_head, L, head_dim)
k = k.reshape(B, L, num_key_value_heads, head_dim).transpose(1, 2)  # (B, num_key_value_heads, L, head_dim)
v = v.reshape(B, L, num_key_value_heads, head_dim).transpose(1, 2)  # (B, num_key_value_heads, L, head_dim)

# [a, b, c], repeats=2 -> [a, a, b, b, c, c]
k = torch.repeat_interleave(k, dim=1, repeats=num_key_value_groups) # (B, num_head, L, head_dim)
v = torch.repeat_interleave(v, dim=1, repeats=num_key_value_groups) # (B, num_head, L, head_dim)

scores = q @ k.transpose(-1, -2)  # scores: (B, num_head, L, L)
weight = torch.nn.softmax(scores, dim=-1)
y = weight @ v  # (B, num_head, L, L) x (B, num_head, L, head_dim) = (B, num_head, L, head_dim)
y = y.transpose(1, 2).reshape(B, L, D)
```


## Models

### GPT-2

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

## RWKV

## GLM

- 归一化: RMSNorm
- 激活函数:
- 位置编码: RoPE
- 注意力: 采用 Multi-Query Attention, 实现上还利用了 flashattention-v1

## MOSS (TO REMOVE)

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
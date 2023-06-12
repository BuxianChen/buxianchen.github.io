---
layout: post
title: "(WIP) NLP Model Architecture Examples"
date: 2023-06-11 17:20:04 +0800
labels: [transformers]
---

## 动机、参考资料、涉及内容

动机

- 梳理目前较为经典的 NLP 模型的结构 (在原始 Transformers 结构上的微小改动的部分)

参考资料

- 🤗 Transformer 的实现

涉及内容

- 模型结构
- 具体的超参数, 例如层数, 特征向量维数等
- 各个模型间的“渊源”梳理

## 渊源


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

## GLM

## MOSS

## LLAMA

## Open-LLAMA

## RWKV
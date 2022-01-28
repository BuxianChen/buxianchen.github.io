---
layout: post
title:  "Swin Transformer"
date: 2021-12-15 22:30:04 +0800
---

### 二维相对位置编码

在一个 $W*W$ 的二维区域中，相对位置有多少种呢？答案是 $(2W-1)^2$ 种，这是因为二维区域中的两个像素点的 $x$ 轴的距离的取值为 $[-(W-1), (W-1)]$，同理 $y$ 轴的距离为 $[-(W-1), (W-1)]$。例如：如下图所示，$W=3$，像素点 $A$ 与像素点 $I$ 之间的距离为 $(-2,-2)$，而像素点对 $(B, G)$ 与像素点对 $(C, H)$ 间的距离均为 $(1, -2)$。

{% raw %}
{% drawio path="../assets/figures/swin-transformer/relative-position-embedding.drawio" page_number=0 height=500 %}
{% endraw %}

因此，只需使用一个形状为 `((2W-1)**2,)` 的张量存储所有的偏置项（即 `self.relative_position_bias_table`），另外需要一个下标模板（即 `relative_position_index`）从这个偏置项集合中取出相应的元素即可：

```python
self.relative_position_bias_table = nn.Parameter(
    torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

# get pair-wise relative position index for each token inside the window
coords_h = torch.arange(self.window_size[0])
coords_w = torch.arange(self.window_size[1])
coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
relative_coords[:, :, 1] += self.window_size[1] - 1
relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
self.register_buffer("relative_position_index", relative_position_index)
```

在上图这个例子里，下标模板为：

$$\begin{bmatrix}
12&11&10&7&6&5&2&1&0\\
13&12&11&8&7&6&3&2&1\\
14&13&12&9&8&7&4&3&2\\
17&16&15&12&11&10&7&6&5\\
18&17&16&13&12&11&8&7&6\\
19&18&17&14&13&12&9&8&7\\
22&21&20&17&16&15&12&11&10\\
23&22&21&18&17&16&13&12&11\\
24&23&22&19&18&17&14&13&12\\
\end{bmatrix}$$


### WindowAttention


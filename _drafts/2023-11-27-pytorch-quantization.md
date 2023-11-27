---
layout: post
title: "(WIP) Pytorch Quantization"
date: 2023-11-27 11:10:04 +0800
---

Pytorch 原生量化支持

pytorch 文档中对量化的具体公式没有很清楚的描述, 可以参考这个 [blog](https://leimao.github.io/article/Neural-Networks-Quantization/)

```python
x = torch.tensor([-1.0, 0.0, 1.0, 20])
qx = torch.quantize_per_tensor(x, scale=0.1, zero_point=10, dtype=torch.qint8)
print(qx)
print("int8 represent", torch.int_repr(qx))  # 获取 int8 数据, qx = (x / s + zero) = x / 0.1 + 10
print("dequantized data", qx.dequantize())   # 反量化回 float 类型
qx.q_scale()  # 0.1
qx.q_zero_point()  # 10
qx.qscheme()  # torch.per_tensor_affine
# tensor([-1.0000,  0.0000,  1.0000, 11.7000], size=(4,), dtype=torch.qint8,
#     quantization_scheme=torch.per_tensor_affine, scale=0.1, zero_point=10)
# int8 represent tensor([  0,  10,  20, 127], dtype=torch.int8)
# dequantized data tensor([-1.0000,  0.0000,  1.0000, 11.7000])
```

注意: 浮点数 `0.0`, 总是会被精确的转化为 `zero_point`


dynamic quantization linear 层的计算逻辑

```python
model = torch.nn.Sequential(torch.nn.Linear(3, 10, bias=False))
qmodel = torch.quantization.quantize_dynamic(
    model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
)
model2 = torch.nn.Sequential(torch.nn.Linear(3, 10, bias=False))
model2[0].weight.data = qmodel[0].weight().dequantize().clone()

# x = torch.tensor([[1, 2., 3], [2, 3, 4]])  # (2, 3), w:(10, 3)
x = torch.rand(1, 3)

# 正常用法: 直接利用 torch.nn.quantized.dynamic.modules.linear.Linear 与 float32 的输入做运算
y1 = qmodel(x)

qw = qmodel[0].weight()
# 输入的量化方式存疑
qx = torch.quantize_per_tensor_dynamic(x, dtype=torch.qint8, reduce_range=False)

intw = qw.int_repr().to(torch.int64).T  # (3, 10)
intx = qx.int_repr().to(torch.int64)    # (2, 3)

zw = qw.q_zero_point()
zx = qx.q_zero_point()

sw = qw.q_scale()
sq = qx.q_scale()

# 手动将运算转为 int 计算(输入也做量化)
y2 = sw * sq * (intx @ intw - zx * torch.ones_like(intx) @ intw - intx @ (zw * torch.ones_like(intw)) + zx*zw)

# 将权重替换为量化后的, 输入不做量化, 这样与正常用法会有误差
y3 = model2(x)

print((y1-y2).abs().max(), sw*sq)
print((y1-y3).abs().max())
```
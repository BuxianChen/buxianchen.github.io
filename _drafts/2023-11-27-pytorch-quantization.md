---
layout: post
title: "(WIP) Pytorch Quantization"
date: 2023-11-27 11:10:04 +0800
---

参考资料:

- Pytorch 的一篇指导性的博客（食用指南！）: [https://pytorch.org/blog/quantization-in-practice/](https://pytorch.org/blog/quantization-in-practice/)
- 官方支持量化的博客: [https://pytorch.org/blog/introduction-to-quantization-on-pytorch/](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
- 官方文档: [https://pytorch.org/docs/2.1/quantization.html](https://pytorch.org/docs/2.1/quantization.html)
- 官方API文档: [https://pytorch.org/docs/2.1/quantization-support.html](https://pytorch.org/docs/2.1/quantization-support.html)
- 官方tutorial搜索: [https://pytorch.org/tutorials/search.html?q=quantization&check_keywords=yes&area=default](https://pytorch.org/tutorials/search.html?q=quantization&check_keywords=yes&area=default)
- Huggingface Optimum 的一篇概念介绍: [https://huggingface.co/docs/optimum/v1.16.1/en/concept_guides/quantization](https://huggingface.co/docs/optimum/v1.16.1/en/concept_guides/quantization)

Pytorch Tutorials:

- Static Quantization + QAT: [https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- Pytorch Wiki: [https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor](https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor)

相关内容:

- QLoRA: [https://huggingface.co/blog/4bit-transformers-bitsandbytes](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- FP8 paper: [FP8 formats for deep learning](https://arxiv.org/pdf/2209.05433.pdf): 英伟达 H100 引入, 文中写 E4M3 的最大值是 448, 但笔者按 IEEE 754 算是 240, 其余均吻合. 原因是 E4M3 不完全遵循 IEEE 754, 而 E5M2 遵循 IEEE 754 (参考: [博客](https://lambdalabs.com/blog/nvidia-hopper-h100-and-fp8-support))
- 一篇关于 QAT 的知乎[博客](https://zhuanlan.zhihu.com/p/548174416), 博客中有原论文及原Tensorflow实现的, Pytorch 的实现包含在本文内容中. 如果要分析 QAT 的原始 TensorFlow 实现, 主要看这个端到端的[例子](https://www.tensorflow.org/model_optimization/guide/quantization/training_example), 以及入口[源码](https://github.com/tensorflow/model-optimization/blob/v0.3.0/tensorflow_model_optimization/python/core/quantization/keras/quantize.py#L80), 这些代码与博客中的分析也基本一致.
- 一篇基于Pytorch官方博客的笔记: [博客园笔记](https://www.cnblogs.com/LXP-Never/p/16822727.html)

Pytorch 原生量化支持分为三类:

- Post-Training Dynamic Quantization: 原理上是提前将权重转化为 int8, 在计算时, 每一层的输入先由浮点数转化为 int8 (量化过程的 `max_val` 和 `min_val` 动态决定), 之后用 int8 的输入与 int8 的权重进行矩阵乘法或卷积等运算, 然后将输出转换回浮点数. 因为每一层都需要动态计算出 `max_val` 和 `min_val`, 并且需要不断地对 activation 进行 int8 与浮点数之间的转换, 因此加速并不明显.
- Post-Training Static Quantization: 原理上是模型训练好后, 首先将权重转换为 int8, 然后给模型喂入一批数据, 计算每层输入的分布情况, 由此得到每一层输出的 `min_val` 和 `max_val`, 因此初看上去, 可以节约动态计算 `min_val` 和 `max_val` 的时间, 然而实际上, 这种做法可以允许整个网络每层之间不必要进行 activation 的 int8 与浮点数之间的转换(为什么?), 所以可以获得比较大的加速.
- Quantization Aware Training: 训练过程中就加入量化损失

## 注意事项

Pytorch 的量化功能目前仅支持 CPU, 在量化算法方面, 不同的软件例如 TensorRT 都有着各自的量化策略, 并没有所谓的"正统". 从使用者的角度更多还是了解大致的原理, 用即可. 原理上只需要记住以下几点:

以下是 baseline

- Dynamic/Static 量化至 int8/uint8 的 baseline 是 RTN (round to nearset): 注意变体是 per tensor/channel, 如果需要详细了解更具体的算法细节参考 Pytorch/TensorRT 均可
- QAT baseline 的核心思路待确认(Tensorflow vs Pytorch): 参考 tf/pytorch 的官方集成即可

以下这几个是一些比较出名的量化算法(集成进了 huggingface transformer)

- GPTQ
- AWQ
- LLM.int8 (bnb)

至于平时的各种所谓的 int8/int4 训练, 一般都没有特定的标准算法?

Pytorch 原生支持的量化算法因为只支持 CPU, 所以应该暂时没啥大用, 目前仅供学习, 而且很多实现也只是 Pytorch 的一种选择, 而不像 `torch.amp` 那样成为混合精度训练的事实标准之一.

- 线性层 (`torch.ao.nn.quantized.dynamic.modules.linear.Linear`): 只量化权重, 不量化偏置, 注意这是一种选择, 而不是不能做
- 卷积层, 仅支持静态量化, 动态量化不支持(Pytorch 开发团队认为这个算子做动态量化精度损失太大, 所以干脆不予支持, 注意这是一种选择, 而不是不能做)


pytorch 文档中对量化的具体公式没有很清楚的描述, 可以参考这个 [blog](https://leimao.github.io/article/Neural-Networks-Quantization/)

```python
x = torch.tensor([-1.0, 0.0, 1.0, 20])
qx = torch.quantize_per_tensor(x, scale=0.1, zero_point=10, dtype=torch.qint8)
print(qx)
print("int8 represent", torch.int_repr(qx))  # 获取 int8 数据, qx = (x / s + zero) = x / 0.1 + 10, qx.int_repr() 也 OK
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

这篇文档 [https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor](https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor) 记录了一些关于 Quantized Tensor 的接口

```python
x = torch.randn(2, 3)  # (B, in)
scale, zero_point = 1e-4, 2
dtype = torch.quint8
x = torch.quantize_per_tensor(x, scale, zero_point, dtype)


w = torch.randn(4, 3)  # (out, in)
scale, zero_point = 1e-3, 2
dtype = torch.qint8
w = torch.quantize_per_tensor(w, scale, zero_point, dtype)

torch.ao.nn.quantized.functional.linear(x, w)
```

## Dynamic Quantization

参考[这里](https://pytorch.org/blog/quantization-in-practice/#post-training-dynamicweight-only-quantization), 运行环境: torch==2.0.0, python 3.10

```python
import torch
from torch import nn

m = nn.Sequential(
  nn.Conv2d(2, 64, 8),
  nn.ReLU(),
  nn.Flatten(),
  nn.Linear(16,10),
  nn.LSTM(10, 10))

m.eval()

## EAGER MODE
from torch.quantization import quantize_dynamic
model_quantized = quantize_dynamic(
    model=m, qconfig_spec={nn.LSTM, nn.Linear}, dtype=torch.qint8, inplace=False
)

## FX MODE
from torch.quantization import quantize_fx
qconfig_dict = {"": torch.quantization.default_dynamic_qconfig}  # An empty key denotes the default applied to all modules
model_prepared = quantize_fx.prepare_fx(m, qconfig_dict, torch.rand(32, 2, 8, 8))
model_quantized = quantize_fx.convert_fx(model_prepared)
```

### eager mode

大体上是遍历每一个层, 如果可以动态量化, 则进行一对一转换:

```python
mapping = {
    torch.nn.modules.linear.Linear: torch.ao.nn.quantized.dynamic.modules.linear.Linear,
    torch.nn.modules.rnn.LSTM: torch.ao.nn.dynamic.modules.rnn.LSTM,
}
mod: torch.nn.modules.linear.Linear = torch.nn.Linear(4, 5)
new_mod = torch.ao.nn.quantized.dynamic.modules.linear.Linear.from_float(mod)  # 一对一转换


# torch.ao.nn.quantized.dynamic.modules.linear.Linear.from_float 的具体细节(一个分支)
mod = torch.nn.Linear(64, 10)  # out: 64, in: 10
observer = torch.ao.quantization.observer.MinMaxObserver()
observer(mod.weight)  # from_float 方法中
wt_scale, wt_zp = observer.calculate_qparams()  # 在 torch.ao.nn.quantized.modules.utils._quant_weight 函数中

qweight = torch.quantize_per_tensor(mod.weight.float(), float(wt_scale), int(wt_zp), torch.int8)
qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)  # torch.ao.nn.quantized.modules.utils._clamp_weights
qlinear = torch.ao.nn.quantized.dynamic.modules.linear.Linear(mod.in_feature, mod.out_feature, dtype=torch.int8)
qlinear.set_weight_bias(qwight, mod.bias)
```

### fx mode


## QAT (tensorflow)

参考 [Realease](https://github.com/tensorflow/model-optimization/releases?page=2) 列表, 相关版本: `tensorflow==2.0.0`, `model_optimization==0.3.0`

```python
# tensorflow_model_optimization/python/core/quantization/keras/quantize.py
def quantize_model(to_quantize: tf.keras.Model):
    def _add_quant_wrapper(layer):
        return quantize_annotate_mod.QuantizeAnnotate(layer)
    annotated_model = tf.keras.models.clone_model(to_annotate, input_tensors=None, clone_function=_add_quant_wrapper)
    return quantize_apply(annotated_model)
```
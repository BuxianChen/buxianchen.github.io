---
layout: post
title: "(WIP) Pytorch Quantization"
date: 2023-11-27 11:10:04 +0800
---

## 动机

了解模型量化的基本原理, 以及 Pytorch 对这些量化算法的实现

## 总览

本文主要参考资料:

- [A1] Pytorch 的一篇指导性的博客 (食用指南! 可快速上手使用转化为生产力, 读者如果仅出于使用目的可以只看这篇博客, 本文后续内容均可不看): [https://pytorch.org/blog/quantization-in-practice/](https://pytorch.org/blog/quantization-in-practice/)
- [A2] 官方支持量化的博客 (内含 3 种量化模式的上层 API, 但不是完整可运行示例, 也不包括后续版本增加的 fx mode 量化): [https://pytorch.org/blog/introduction-to-quantization-on-pytorch/](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
- [A3] 官方文档 (需要仔细琢磨): [https://pytorch.org/docs/2.1/quantization.html](https://pytorch.org/docs/2.1/quantization.html)
- [A4] 官方 API 文档 (因为 Pytorch 提供了 3 种量化方式, 以及 eager/fx 模式, 并且 API 分为上层 API 和底层 API, 所以显得比较混乱, 个人还感觉 Pytorch 量化方面暴露的底层接口似乎不算完善): [https://pytorch.org/docs/2.1/quantization-support.html](https://pytorch.org/docs/2.1/quantization-support.html)
- [A5] Huggingface Optimum 的一篇关于模型量化的总览介绍: [https://huggingface.co/docs/optimum/v1.16.1/en/concept_guides/quantization](https://huggingface.co/docs/optimum/v1.16.1/en/concept_guides/quantization)
- [A6] Pytorch wiki (包含了关于底层诸如 `torch.qint8` 数据类型的张量的一些 API): [https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor](https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor)
- [A7] 一篇详细介绍数学公式推导的博客, 非常值得仔细研究: [https://leimao.github.io/article/Neural-Networks-Quantization/](https://leimao.github.io/article/Neural-Networks-Quantization/)

Pytorch Tutorials (一些端到端的例子):

- 官方 tutorial 搜索 (端到端的示例): [https://pytorch.org/tutorials/search.html?q=quantization&check_keywords=yes&area=default](https://pytorch.org/tutorials/search.html?q=quantization&check_keywords=yes&area=default)
- Static Quantization + QAT: [https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)

相关内容 (本文可能不会过多涉及):

- QLoRA: [https://huggingface.co/blog/4bit-transformers-bitsandbytes](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- FP8 paper: [FP8 formats for deep learning](https://arxiv.org/pdf/2209.05433.pdf): 英伟达 H100 引入, 文中写 E4M3 的最大值是 448, 但笔者按 IEEE 754 算是 240, 其余均吻合. 原因是 E4M3 不完全遵循 IEEE 754, 而 E5M2 遵循 IEEE 754 (参考: [博客](https://lambdalabs.com/blog/nvidia-hopper-h100-and-fp8-support))
- 一篇关于 QAT 的知乎[博客](https://zhuanlan.zhihu.com/p/548174416), 博客中有原论文及原Tensorflow实现的, Pytorch 的实现包含在本文内容中. 如果要分析 QAT 的原始 TensorFlow 实现, 主要看这个端到端的[例子](https://www.tensorflow.org/model_optimization/guide/quantization/training_example), 以及入口[源码](https://github.com/tensorflow/model-optimization/blob/v0.3.0/tensorflow_model_optimization/python/core/quantization/keras/quantize.py#L80), 这些代码与博客中的分析也基本一致.
- 一篇基于Pytorch官方博客的笔记: [博客园笔记](https://www.cnblogs.com/LXP-Never/p/16822727.html)

Pytorch 原生量化支持有三类 (关于计算流程的描述也可参见官方文档 [A3](https://pytorch.org/docs/2.1/quantization.html#eager-mode-quantization) 的各个 Diagram):

**Post-Training Dynamic Quantization**

原理上是提前将权重转化为 int8, 在计算时, 每一层的输入 (activation) 先由浮点数转化为 int8 (这一量化过程动态决定, 例如用输入数据的 `max_val` 和 `min_val` 确定量化过程的放缩因子与零点), 之后用 int8 的输入与 int8 的权重进行矩阵乘法或卷积等运算得到结果, 然后将结果转换回浮点数. 因为每一层都需要动态计算出 `max_val` 和 `min_val`, 并且需要不断地对 activation 进行 int8 与浮点数之间的转换, 因此加速并不明显.

```
# original model
# all tensors and computations are in floating point
previous_layer_fp32 -- linear_fp32 -- activation_fp32 -- next_layer_fp32
                 /
linear_weight_fp32

# dynamically quantized model
# linear and LSTM weights are in int8
previous_layer_fp32 -- linear_int8_w_fp32_inp -- activation_fp32 -- next_layer_fp32
                     /
   linear_weight_int8
```

**Post-Training Static Quantization**

原理上是模型训练好后, 首先将权重转换为 int8, 然后给模型喂入一批数据, 计算每层输入 (activation) 的分布情况, 由此得到每一层输入及输出的量化放缩因子与零点 (例如通过统计 `max_val` 和 `min_val` 做到), 模型推理过程如下: 每一层的输入都是 int8 类型, 然后直接与 int8 权重进行 int8 矩阵乘法等运算, 然后根据输出的放缩因子进行少量的取整操作得到 int8 的输出 (可参见 [A7](https://leimao.github.io/article/Neural-Networks-Quantization/#Quantized-Matrix-Multiplication) 的公式). 另外, 输入层需要做一次 float 到 int 的静态量化, 输出层需要做一次反量化得到结果.

```
# original model
# all tensors and computations are in floating point
previous_layer_fp32 -- linear_fp32 -- activation_fp32 -- next_layer_fp32
                    /
    linear_weight_fp32

# statically quantized model
# weights and activations are in int8
previous_layer_int8 -- linear_with_activation_int8 -- next_layer_int8
                    /
  linear_weight_int8
```

**Quantization Aware Training**

训练过程中就加入量化损失, 量化后的模型可以是 dynamic/static/weight-only 量化的 (按英文术语说是: QAT for dynamic/static/weight-only quantization), 以下图例实际上只体现了 static quantization 的情形

```
# original model
# all tensors and computations are in floating point
previous_layer_fp32 -- linear_fp32 -- activation_fp32 -- next_layer_fp32
                      /
    linear_weight_fp32

# fq 代表一次量化加一次反量化过程, 因此 fq 的输出仍然是浮点数
# model with fake_quants for modeling quantization numerics during training
previous_layer_fp32 -- fq -- linear_fp32 -- activation_fp32 -- fq -- next_layer_fp32
                           /
   linear_weight_fp32 -- fq

# quantized model
# weights and activations are in int8 (same as static quantization)
previous_layer_int8 -- linear_with_activation_int8 -- next_layer_int8
                     /
   linear_weight_int8
```

需要注意的是, 还存在着另一个术语:

**Weight-Only Quantization**

原理是只对权重进行量化, 在计算时, 每一层的输入是浮点数, 权重被反量化会浮点数, 然后直接执行浮点数的算子得到结果. 因此这种量化方式在计算时实际上都是浮点运算.

```
# original model
# all tensors and computations are in floating point
previous_layer_fp32 -- linear_fp32 -- activation_fp32 -- next_layer_fp32
                    /
    linear_weight_fp32

# Weight-Only Quantization
# weights are in int8, activations are in float
previous_layer_fp32 -- linear_with_activation_fp32 -- next_layer_fp32
                            /
linear_weight_int8 -- dequantized_linear_weight_fp32
```


源码目录(其余参考[A6](https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor)):

- python 代码: torch/ao/quantization, 早期版本位于 torch/quantization, 为了保持兼容性, 目前在 torch/quantization 目录下的 python 脚本都是一些 import 语句


## 注意事项 (TODO, 可能合并进总览这一节中)

Pytorch 的量化功能目前仅支持 CPU (不确定, 应该支持 GPU, 待确认), 在量化算法方面, 不同的软件例如 TensorRT 都有着各自的量化策略, 并没有所谓的"正统". 从使用者的角度更多还是了解大致的原理, 用即可. 原理上只需要记住以下几点:

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

## 指南 (TODO, 可能直接合并进总览这一节中)

本节主要描述一些总览知识, 主要参考 [A3](https://pytorch.org/docs/2.1/quantization.html), [A4](https://pytorch.org/docs/2.1/quantization-support.html), [A5](https://huggingface.co/docs/optimum/v1.16.1/en/concept_guides/quantization), [A7](https://leimao.github.io/article/Neural-Networks-Quantization/)

## API 总览 (TODO)

本节主要梳理 pytorch 关于量化的源码目录及 API 接口的层次关系, 尤其关注上层接口, 主要是梳理 [A4](https://pytorch.org/docs/2.1/quantization-support.html), 但绝非完整介绍

最上层的 API:

- torch.ao.quantization.quantize: static quantization
- torch.ao.quantization.quantize_dynamic: dynamic quantization
- torch.ao.quantization.quantize_qat: QAT
- torch.ao.quantization.prepare: static quantization
- torch.ao.quantization.prepare_qat: QAT
- torch.ao.quantization.convert: static quantization/QAT

这些接口的关系可以用下面的伪代码来描述 (注意省略了一些使用细节, 仅仅是框架):

```python
# dynamic quantization
qmodel = quantize_dynamic(model, ...)

# static quantization
# 方法一: 实质上与方法二等价
qmodel = quantize(model, fn, ...)
# 方法二:
fp_model = prepare(model, fn, ...)  # fp_model 依然是 float 类型的模型
fn(fp_model, ...)                   # 校准数据喂入模型
qmodel = convert(fp_model)

# QAT
# 方法一: 实质上与方法二等价
qmodel = quantize_qat(model, fn, ...)
# 方法二:
fp_model = prepare_qat(model, fn, ...)  # fp_model 依然是 float 类型的模型
fn(fp_model, ...)                       # 校准数据喂入模型
qmodel = convert(fp_model)
```

底层 API (省略 `torch.ao` 前缀):

TODO: 这部分 API 还需理清, 很容易错乱, 目前感觉是分为 3 种量化的公共底层, 然后是 3 种量化各自的底层接口, 而 static quantization 和 QAT 有一部分接口也是共用的. 从 layer 的角度看, 大体上可能有这些:

- dynamic quantization
  - 量化前对 float layer 进行 layer 的融合
  - 量化后的 int layer
  - 量化后的 int fused layer
- static quantization
  - 量化前的准备 (fused) float layer (前向过程增加 observer 的调用)
  - 量化后的 (fused) int layer
- QAT
  - 伪量化层
  - 量化前的准备 (fused) float layer (前向过程增加 observer 及伪量化层的调用)
  - 量化后的 (fused) int layer

以下是具体的 API

- 公共部分:
    - quantizad tensor (感觉暴露的接口/文档并不完善): Tensor.dequantize, Tensor.q_scale, Tensor.q_zero_point, Tensor.int_repr, torch.quantize_per_tensor, torch.dequantize, torch.quantize_per_tensor_dynamic
    - observer (观测输入输出浮点值的分布): quantization.observer.MinMaxObserver, quantization.observer.default_observer
    - QConfig (量化配置, 例如 int8/uint8, 对称量化/非对称量化): torch.per_tensor_affine, quantization.qconfig.QConfig, quantization.qconfig.default_dynamic_qconfig
- dynamic quantization:
    - 量化后的 layer: nn.quantized.dynamic.Linear
- static quantization/QAT:
    - prepare model 的底层工具: fuse_modules, QuantStub
    - 量化后的 fused layer: nn.intrinsic.quantized.ConvReLU2d
    - 量化后的 layer: nn.quantized.Linear
    - 量化后的 layer 的函数式接口: nn.quantized.functional.conv2d
    - 量化算子 (非公开接口): torch.ops.quantized.conv2d
- static quantization:
    - prepare model (float model): nn.intrinsic.ConvReLU2d
- QAT:
    - prepare model: quantization.fake_quantize.FakeQuantize
    - prepare model (float model): nn.intrinsic.qat.ConvReLU2d

不确定的
- nn.intrinsic.quantized.dynamic.LinearReLU, nn.qat.dynamic.Linear, nn.qat.Linear
- nn.quantizable.LSTM


这些上层 API 与底层 API 之间的联系在官方文档中解释和代码示例中里解释的比较清楚

- [A3-post-training-dynamic-quantization](https://pytorch.org/docs/2.1/quantization.html#post-training-dynamic-quantization): dynamic quantization 原理
- [A3-post-training-static-quantization](https://pytorch.org/docs/2.1/quantization.html#post-training-static-quantization): static quantization 原理
- [A3-quantization-aware-training-for-static-quantization](https://pytorch.org/docs/2.1/quantization.html#quantization-aware-training-for-static-quantization): QAT 原理
- [A3-quantization-custom-module-api](https://pytorch.org/docs/2.1/quantization.html#quantization-custom-module-api): 自定义量化教程, 解释了上层 API 与底层 API 间的联系

static quantization/QAT 使用注意事项 (即需要对原模型的代码进行变动, 例如增加 `QuantStub` 和 `DeQuantStub` 层, 使用 `FloatFunctional` 替换一些 `add`, `cat` 操作): [A3-model-preparation-for-eager-mode-static-quantization](https://pytorch.org/docs/2.1/quantization.html#model-preparation-for-eager-mode-static-quantization)

基于 torch.fx 的量化方式: 以笔者目前的观点看, 正是因为 eager mode 的使用有上述注意事项, 使得使用者需要对原模型的代码进行小变动或重构, 而更理想的方式是不对原模型的代码进行变动, torch.fx 量化方式就是为了做到这一点.

下面会先具体介绍一下底层接口, 之后再分章节从上层接口的官方使用示例作为源码分析目标, 介绍各个量化算法的具体流程及相应的实现方式.

## 底层接口

本节只介绍一部分底层接口, 其余底层接口与具体的量化算法结合起来在后续章节介绍.

### quantized tensor

(TODO: 这句话需要调整一下) pytorch 文档中对量化的具体数学公式及针对量化张量的算子没有十分仔细的描述, 对公式感兴趣的读者仔细研究这个博客 [A7](https://leimao.github.io/article/Neural-Networks-Quantization/)

Pytorch 的核心量化公式是:

$$
\begin{align*}
Xq &= round(\frac{x}{s}) + Z \quad (quantization)\\
\tilde{x}&=(Xq - Z) * s \quad (dequantization)
\end{align*}
$$

其中 $x$ 是原始的浮点数值, $Xq$ 是量化后的整数值, $\tilde{x}$ 是量化-反量化后的浮点数值, $Z$ 是浮点数 $0.0$ 量化后的整数值 (从量化公式上看, 浮点数 $0.0$ 经过量化-反量化后会是无损的), $s$ 是浮点数放缩因子

接下来简单看下一些关于 quantized tensor 的底层 API, 主要参考资料: [A6](https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor), [A2](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)

#### quantized tensor 的创建

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

#### quantized tensor 算子 (TODO)

- 算子: `torch.ops.quantized`, 例如: `torch.ops.quantized.linear_dynamic`
- nn.Module:
- functional:
- quantized tensor 方法:

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


### observer

observer 的作用主要是确定原始浮点数据的 `min_val` 和 `max_val`, 并依据量化后整数数值范围计算好放缩系数及“整数零”, 代码主要位于 `torch/ao/quantization/observer.py` 下, 以下示例参考[A1](https://pytorch.org/blog/quantization-in-practice/#calibration)

```python
# 备注: 后续版本 pytorch 的源码计划由 torch/quantization -> torch/ao/quantization
from torch.ao.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver, HistogramObserver
C, L = 3, 4
normal = torch.distributions.normal.Normal(0,1)
inputs = [normal.sample((C, L)), normal.sample((C, L))]

observers = [MinMaxObserver(), MovingAverageMinMaxObserver(), HistogramObserver()]
for obs in observers:
    for x in inputs:
        obs(x)  # MinMaxObserver 这些类实际上继承自 nn.Module
        print(obs.__class__.__name__, obs.calculate_qparams())
```

`torch/ao/quantization/observer.py` 还包含了许多内部使用的 `default_*_observer` 的定义, 例如:

```python
# Default weight observer.
default_weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)

# Symmetric weight observer with the 8-bit values restricted to [-127, +127], excluding -128.
weight_observer_range_neg_127_to_127 = MinMaxObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_tensor_symmetric,
    quant_min=-127, quant_max=127, eps=2 ** -12)

# Default observer for dynamic quantization.
default_dynamic_quant_observer = PlaceholderObserver.with_args(
    dtype=torch.quint8, quant_min=0, quant_max=255, is_dynamic=True,
)

# ...
```

### qschema, QConfig, observer

伪代码如下

```python
# 一般情况下 reduce_range 是 False, quant_min 和 quant_max 由 dtype 及 reduce_range 确定
observer = Observer(
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,  # 枚举类型
    reduce_range=False,  # 含义参见 https://discuss.pytorch.org/t/what-is-the-purpose-of-reduce-range/163291
    quant_min=None,  # -128
    quant_max=None,  # 127
)

# QConfig 通常用于高阶 API (即直接指定带量化的完整模型的量化选项)
# torch/ao/quantization/qconfig.py
default_dynamic_qconfig = QConfig(
    activation=default_dynamic_quant_observer,
    weight=default_weight_observer
)
```

关于 `reduce_range` 参数, 可参见[这里](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/ao/quantization/utils.py#L367), 简单来说:

```python
# torch.qint8
quant_min, quant_max = -128, 127  # reduce_range=False
quant_min, quant_max = -64, 63    # reduce_range=True
# torch.quint8
quant_min, quant_max = 0, 255     # reduce_range=False
quant_min, quant_max = 0, 127     # reduce_range=True
# torch.qint32
quant_min, quant_max = -1 * (2 ** 31), (2 ** 31) - 1  # reduce_range=False
quant_min, quant_max = 0, 15                          # reduce_range=True
```

## Dynamic Quantization

### 使用方式

涉及如下 3 个高级 API:

- eager mode: `torch.quantization.quantize_dynamic`
- fx mode: `torch.quantization.quantize_fx.prepare_fx`, `torch.quantization.quantize_fx.convert_fx`

以下是一个完整的示例, 参考自[A1](https://pytorch.org/blog/quantization-in-practice/#post-training-dynamicweight-only-quantization), 运行环境: torch==2.0.0, python 3.10

```python
import torch
from torch import nn

m = nn.Sequential(
  nn.Conv2d(2, 64, 8),
  nn.ReLU(),
  nn.Flatten(),
  nn.Linear(64, 10),
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

接下来具体分析上面的高级接口实际是怎么运作的

### eager mode

`quantize_dynamic` 函数的执行逻辑大体上是遍历每一个层, 如果可以动态量化, 则进行一对一转换, 伪代码如下:

```python
mapping = {
    torch.nn.modules.linear.Linear: torch.ao.nn.quantized.dynamic.modules.linear.Linear,
    torch.nn.modules.rnn.LSTM: torch.ao.nn.dynamic.modules.rnn.LSTM,
}
mod: torch.nn.modules.linear.Linear = torch.nn.Linear(4, 5)
new_mod = torch.ao.nn.quantized.dynamic.modules.linear.Linear.from_float(mod)  # 一对一转换

# 因此主要就关注 from_float 方法及 forward 方法即可
# torch.ao.nn.quantized.dynamic.modules.linear.Linear.from_float 的具体细节(一个分支)
mod = torch.nn.Linear(64, 10)  # out: 64, in: 10
observer = torch.ao.quantization.observer.MinMaxObserver()
observer(mod.weight)
wt_scale, wt_zp = observer.calculate_qparams()  # 在 torch.ao.nn.quantized.modules.utils._quant_weight 函数中

qweight = torch.quantize_per_tensor(mod.weight.float(), float(wt_scale), int(wt_zp), torch.int8)
qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)  # torch.ao.nn.quantized.modules.utils._clamp_weights
qlinear = torch.ao.nn.quantized.dynamic.modules.linear.Linear(mod.in_feature, mod.out_feature, dtype=torch.int8)
qlinear.set_weight_bias(qwight, mod.bias)

# torch.ao.nn.quantized.dynamic.modules.linear.Linear.forward 见下面的分析
```

#### `torch.ao.nn.quantized.dynamic.modules.linear.Linear` 深入分析

以线性层为例, 来分析一下底层实现, `torch==2.0.0` 源代码:

```python
# torch/ao/nn/quantized/dynamic/modules/linear.py
class Linear(nnq.Linear):  # 父类 nnq.Linear 是 torch.ao.nn.quantized.modules.linear.Linear, 这里不再过多描述
    def __init__(self, ...): ...
    
    def forward(self, x):  # 我们只看 qint8 的实现: 只支持fp16与qint8
        # self._packed_params 是 torch.ao.nn.quantized.modules.linear.LinearPackedParams 对象
        # self._packed_params._packed_params 是 torch._C.ScriptObject 对象, 包含了量化后的权重(torch.qint8类型)与偏置(fp32类型)
        # 输入: x(float32), 输出: Y(float32)
        Y = torch.ops.quantized.linear_dynamic(x, self._packed_params._packed_params, reduce_range=True)  # 此算子是 C++ 实现
        return Y.to(x.dtype)  # 这里是 float -> float 的转换 (也许会出现 fp16 与 fp32 之间的转换?)
    
    # 伪代码刚刚已经分析过, 这里贴上源码
    @classmethod
    def from_float(cls, mod):
        # mod (nn.Module): a float module, either produced by torch.ao.quantization utilities or provided by the user
        float_modules = [torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
                         torch.ao.nn.intrinsic.modules.fused.LinearReLU, torch.ao.nn.qat.dynamic.Linear]

        assert type(mod) in float_modules, 'nn.quantized.dynamic.Linear.from_float only works for one of' + str([float_mod.__name__ for float_mod in float_modules])
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        if type(mod) == nni.LinearReLU:
            mod = mod[0]
        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer = mod.qconfig.weight()
        else:
            from torch.ao.quantization.qconfig import default_dynamic_qconfig
            weight_observer = default_dynamic_qconfig.weight()
        dtype = weight_observer.dtype
        assert dtype in [torch.qint8, torch.float16], "The only supported dtypes for dynamic quantized linear are qint8 and float16 got: {}".format(dtype)
        weight_observer(mod.weight)
        if dtype == torch.qint8:
            qweight = _quantize_weight(mod.weight.float(), weight_observer)
        elif dtype == torch.float16:
            qweight = mod.weight.float()
        else:
            raise RuntimeError('Unsupported dtype specified for dynamic quantized Linear!')
        qlinear = cls(mod.in_features, mod.out_features, dtype=dtype)
        qlinear.set_weight_bias(qweight, mod.bias)
        return qlinear
```

我们只需要重点关注 `from_float` (已用伪代码描述过) 和 `forward` 方法即可, 在此之前先简单看一个小细节: 注意到上面的 `self._packed_params` 是 `LinearPackedParams` 对象, 而 `self._packed_params._packed_params` 是 `torch.ScriptObject` 对象 (`torch.fx` 源码中也有出现)

```python
# torch/ao/nn/quantized/modules/linear.py
class LinearPackedParams(torch.nn.Module):
    _version = 3

    def __init__(self, dtype=torch.qint8):
        super().__init__()
        self.dtype = dtype
        if self.dtype == torch.qint8:
            wq = torch._empty_affine_quantized([1, 1], scale=1.0, zero_point=0, dtype=torch.qint8)
        elif self.dtype == torch.float16:
            wq = torch.zeros([1, 1], dtype=torch.float)
        self.set_weight_bias(wq, None)

    @torch.jit.export
    def set_weight_bias(self, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        if self.dtype == torch.qint8:
            self._packed_params = torch.ops.quantized.linear_prepack(weight, bias)
        elif self.dtype == torch.float16:
            self._packed_params = torch.ops.quantized.linear_prepack_fp16(weight, bias)
        else:
            raise RuntimeError('Unsupported dtype on dynamic quantized linear!')
    
    def forward(self, x):
        return x

    # Version 1
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #
    # Version 2
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- dtype : torch.dtype
    #
    # Version 3
    #   self
    #   |--- _packed_params : (Tensor, Tensor) representing (weight, bias)
    #                         of LinearPackedParams
    #   |--- dtype : torch.dtype
    ...
```

我们现在重点搞清 `forward` 方法中使用到的 `torch.ops.quantized.linear_dynamic` 的具体算法细节, 它是 C++ 实现的, 首先是怎么找到它的 C++ 源码呢? 根据[README.md](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/quantized/README.md) 的指引, 注意到这两个文件:

```C++
// aten/src/ATen/native/quantized/library.cpp
TORCH_LIBRARY(quantized, m) {
    // ...
    m.def(TORCH_SELECTIVE_SCHEMA("quantized::linear_dynamic(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, bool reduce_range=False) -> Tensor Y"));
    m.def(TORCH_SELECTIVE_SCHEMA("_quantized::linear_dynamic(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, bool reduce_range=False) -> Tensor Y"));
    // ...
}

// aten/src/ATen/native/quantized/cpu/qlinear_dynamic.cpp
TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  // ...
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_dynamic"), TORCH_FN(QLinearDynamicInt8<false>::run));
  // ...
}

TORCH_LIBRARY_IMPL(_quantized, CPU, m) {
  // ...
  m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_dynamic"), TORCH_FN(QLinearDynamicInt8<false>::run));
} 
```

而对应的最终实现 (仅关注 `fbgemm` 后端: FaceBook GEneral Matrix Multiplication) 在 `aten/src/ATen/native/quantized/cpu/qlinear_dynamic.cpp` 中, [源码](https://github.com/pytorch/pytorch/blob/v2.0.0/aten/src/ATen/native/quantized/cpu/qlinear_dynamic.cpp#L31), 而它最终会调用 fbgemm 的 [fbgemm::fbgemmPacked](https://github.com/pytorch/FBGEMM/blob/v0.5.0/src/Fbgemm.cc#L29) 函数

**Highlight**

`torch.ops.quantized.linear_dynamic` 的执行逻辑如下 (以下源码在[这里]((https://github.com/pytorch/pytorch/blob/v2.0.0/aten/src/ATen/native/quantized/cpu/qlinear_dynamic.cpp#L31)): `fbgemm` 后端):

- 首先对输入数据进行量化, 使用最大最小非对称量化, 量化后的数据类型为 `uint8`
- 分配输出结果的内存空间 `output` (float32 数组, 源码中可以见到 `at::kFloat` 这样的代码) 和计算缓冲空间 `buffer` (int32 数组, 源码中可以见到诸如 `at::kInt`, `buffer.data_ptr<int32_t>` 这样的代码)
    ```c++
    auto output = at::empty(out_sizes, input.options().dtype(at::kFloat));
    auto buffer = at::empty_like(output, output.options().dtype(at::kInt), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    // ...
    return output
    ```
- 然后调用 `fbgemm::fbgemmPacked` 进行计算: 此算子输入时量化后的输入 (uint8) 与量化权重 (int8) 及偏置 (float32), 输出为 float32 类型, 从以下摘录的源码及注释可以看出, 实际的计算过程是先执行 uint8 与 int8 的矩阵乘法, 计算结果累积在 int32 的 `buffer` 上, 然后转换回 float32 到 `output` 上, 最后加上 float32 的偏置
    ```C++
    // C(output) = A(input) x B(weight), where C, A, B are M x N, M x K, K x N matrices, respectively.
    
    // Process the per tensor quantization.
    //
    // After the uint8 * int8 matrix multiplication is performed, this
    // operation does:
    //  1) Add in row and column offsets to the rows and columns,
    //  respectively.
    //  2) Dequantize the results into floating point.
    //  3) Add in the bias term.
    fbgemm::ReQuantizeForFloat<ReluFused> outputProcObj(
        /*nextop=*/doNothingObj,
        /*Aq_scale=*/q_params.scale,
        /*Bq_scale=*/w_scale.data(),
        /*Aq_zero_point=*/q_params.zero_point,
        /*Bq_zero_point=*/w_zp.data(),
        /*row_offsets=*/packA.getRowOffsetBuffer(),
        /*col_offsets=*/col_offsets.data(),
        /*bias=*/bias_ptr,
        /*nCol=*/N);

    // Do the GEMM
    fbgemm::fbgemmPacked(
        /*packA=*/packA,
        /*packB=*/*packB,
        /*C=*/output.data_ptr<float>(),
        /*C_buffer=*/buffer.data_ptr<int32_t>(),
        /*ldc=*/N,
        /*outProcess=*/outputProcObj,
        /*thread_id=*/task_id,
        /*num_threads=*/num_tasks);
    ```

由于继续深入 `fbgemm::fbgemmPacked` 有些过于琐碎(没能力看懂), 因此这里给出其[源码位置](https://github.com/pytorch/FBGEMM/blob/v0.5.0/src/Fbgemm.cc#L29)与之等价的 python 实现:

**实现一: 使用 pytorch quantization 的低阶 API 实现 (完全对齐高阶API)**

```python
# from torch.ao.quantization.qconfig import default_dynamic_qconfig
# from torch.ao.nn.quantized.modules.linear import LinearPackedParams
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.nn.quantized.modules.utils import _quantize_weight
from torch.quantization import quantize_dynamic

layer = torch.nn.Linear(4, 10)
x = torch.rand(1, 4)
dtype = torch.qint8

# observer = default_dynamic_qconfig.weight()
observer = MinMaxObserver(dtype=dtype, qscheme=torch.per_tensor_symmetric)
observer(layer.weight)
qweight = _quantize_weight(layer.weight.float(), observer)
# packed_params = LinearPackedParams(dtype)
# packed_params._packed_params = torch.ops.quantized.linear_prepack(qweight, layer.bias)
packed_params = torch.ops.quantized.linear_prepack(qweight, layer.bias)  # torch.ScriptObject 对象
# packed_params 有一个 unpack 方法: qweight, bias = packed_params.unpack()
manual_res = torch.ops.quantized.linear_dynamic(x, packed_params, reduce_range=True)

model_quantized = quantize_dynamic(model=torch.nn.Sequential(layer), qconfig_spec={nn.LSTM, nn.Linear}, dtype=torch.qint8, inplace=False)
torch_res = model_quantized(x)

print("高阶API与低阶API的实现差异:", (y1-y2).abs().max().item())  # 0.0
```

**实现二: 使用 pytorch quantization 的部分低阶 API 实现量化张量算子**

```python
batch_size, in_features, out_features = 20, 30, 40
model = torch.nn.Sequential(torch.nn.Linear(in_features, out_features))
qmodel = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8, inplace=False)
x = torch.rand(batch_size, in_features)

# 方法一: 利用高阶 API 计算
y1 = qmodel(x)

# 方法二: 利用低阶 API 计算
qw = qmodel[0].weight()  # symmetric=True, torch.qint8, reduce_range=False
# 与高阶API的一致性: 
# qlinear_dynamic 的源码(https://github.com/pytorch/pytorch/blob/v2.0.0/aten/src/ATen/native/quantized/cpu/qlinear_dynamic.cpp#L31) 中使用 quant_utils::ChooseQuantizationParams(...) 来计算输入数据的量化参数
# 而 quantize_per_tensor_dynamic 的源码(https://github.com/pytorch/pytorch/blob/v2.0.0/aten/src/ATen/native/quantized/QTensor.cpp#L17)也是用同样的方式计算
qx = torch.quantize_per_tensor_dynamic(x, dtype=torch.quint8, reduce_range=True)  # symmetric=False

intw = qw.int_repr().to(torch.int64).T
intx = qx.int_repr().to(torch.int64)

zw = qw.q_zero_point()
zx = qx.q_zero_point()

sw = qw.q_scale()
sq = qx.q_scale()

y2 = model[0].bias + sw * sq * (intx @ intw - zx * torch.ones_like(intx) @ intw - intx @ (zw * torch.ones_like(intw)) + zx*zw)

# 原始模型(未经量化)的输出
y3 = model(x)

print("高阶API与低阶API的实现差异:", (y1-y2).abs().max().item())
print("量化前与量化后的计算误差:", (y1-y3).abs().max().item())
```

**实现三: 不使用任何 pytorch quantization API 实现张量量化及算子**

```python
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic


# =================== linear weight 量化 =========================
# weight: dtype=torch.qint8, symmetric=True, reduce_range=False
# input: 约为: dtype=torch.quint8, symmetric=False, reduce_range=True
def calculate_qparams_linear_weight(weight, dtype=torch.qint8, symmetric=True, reduce_range=False):
    # dtype: qint8/quint8, symmetric: False/True
    # only support per tensor
    min_val, max_val = weight.min(), weight.max()
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))  # 用了 0.0 截断
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))  # 用了 0.0 截断

    device = min_val_neg.device
    scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
    zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    if reduce_range:
        if dtype == torch.qint8:
            quant_min, quant_max = -64, 63
        elif dtype == torch.quint8:
            quant_min, quant_max = 0, 127
        # else: raise ValueError(f"{dtype} not support")
    else:
        if dtype == torch.qint8:
            quant_min, quant_max = -128, 127
        elif dtype == torch.quint8:
            quant_min, quant_max = 0, 255
        # else: raise ValueError(f"{dtype} not support")

    if symmetric:
        max_val_pos = torch.max(-min_val_neg, max_val_pos)
        scale = max_val_pos / (float(quant_max - quant_min) / 2)
        # symmetric 情况下, 0.0 的量化表示是 0/128
        if dtype == torch.quint8:
            zero_point = zero_point.new_full(zero_point.size(), 128)
        # elif dtype == torch.qint8:
        #     zero_point = zero_point.new_full(zero_point.size(), 0)
        # else: raise ValueError(f"{dtype} not support")
    else:
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        # 注意到这可能会使得 min_val_neg 或 max_val_neg 对应的量化值可能不是 quant_min 或 quant_max
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int64)  # 必须用 round 取整, 如果直接用 to(torch.int) 会是向 0 取整
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
    return scale, zero_point, quant_min, quant_max

def quantize_linear_weight(weight, dtype=torch.qint8, symmetric=True, reduce_range=False):
    scale, zero_point, quant_min, quant_max = calculate_qparams_linear_weight(
        weight, dtype=dtype, symmetric=symmetric, reduce_range=reduce_range)
    qweight = torch.round(weight / scale).to(torch.int64) + zero_point
    qweight = torch.clamp(qweight, quant_min, quant_max)
    return qweight

def dequantize_linear_weight(qweight, scale, zero_point, quant_min, quant_max):
    deqweight = (qweight - zero_point).float() * scale
    return deqweight

# torch 默认用 symmetric 的方式量化权重 
def torch_dynamic_quantize_linear_weight(layer: torch.nn.Linear):
    qmodel = quantize_dynamic(model=torch.nn.Sequential(layer), qconfig_spec={nn.Linear}, dtype=torch.qint8, inplace=False)
    weight = qmodel[0].weight()
    
    zero_point = weight.q_zero_point()  # int scaler
    scale = weight.q_scale()            # float scalar
    qweight = weight.int_repr().long()  # int64 tensor
    return qweight, scale, zero_point

# 单元测试
def test_linear_weight_quantize(layer: torch.nn.Linear):
    dtype = torch.qint8
    symmetric = True
    weight = layer.weight
    scale_m, zero_point_m, _, _ = calculate_qparams_linear_weight(weight, dtype=dtype, symmetric=symmetric)
    qweight_m = quantize_linear_weight(weight, dtype=dtype, symmetric=symmetric)

    qweight_t, scale_t, zero_point_t = torch_dynamic_quantize_linear_weight(layer)

    assert int(zero_point_m) == int(zero_point_t)
    assert float(scale_m) - float(scale_t) < 1e-7
    assert (qweight_m - qweight_t).abs().sum().item() == 0

# ========================= linear input 量化 ============================
# https://github.com/pytorch/pytorch/blob/v2.0.0/aten/src/ATen/native/quantized/cpu/qlinear_dynamic.cpp#L31
# 基本上类似于 quantize_linear_weight: symmetric = False, dtype=quint8, reduce_range=True
# 但 zero_point 和 scale 的计算与weight有所区别 (主要是zero_point会在min_val或max_val里选择进行计算)
# TODO: 为简单起见, 这里忽略 zero_point 的差异, 精确实现要参照上面链接里的源码: auto q_params = quant_utils::ChooseQuantizationParams(...)
def quantize_linear_input(x, dtype=torch.quint8, symmetric=False, reduce_range=True):
    return quantize_linear_weight(x, dtype=dtype, symmetric=symmetric, reduce_range=reduce_range)

# ======================== 量化数据的矩阵乘法实现(仅示意) ===========================
def gemm(qweight, qinput, bias, weight_qparams, input_qparams):
    # input @ weight.T + bias, qweight(n*k), qinput(m*k), bias(n)
    # 计算公式如下:
    # deq_x = (q_x - z_x) * s_x, deq_w.T = (q_w.T - z_w.T) * s_w (note: z_x, z_w is matrix)
    # output = deq_x @ deq_w.T + bias = s_w * s_x * (q_x @ q_w.T - q_x @ z_w .T- z_x @ q_w.T + z_x@z_w.T) + bias
    # 注意在真实的实现里: s_w * s_x, z_x @ q_w.T, z_x@z_w 其实可以提前算好, 只有 q_x @ q_w.T 和 q_x @ z_w 需要计算
    m, k = qinput.shape
    n = qweight.shape[0]

    s_w, z_w = weight_qparams
    s_x, z_x = input_qparams

    # 这里比较低效, 直接创建了常量张量
    z_w = torch.ones_like(qweight) * z_w
    z_x = torch.ones_like(qinput) * z_x

    # 真正的实现是用 torch.int32 进行累加, 此处为了简单 qweight, qinput, buffer 其实都用了 torch.long 类型
    buffer = torch.zeros((m, n), dtype=torch.int64)
    output = torch.zeros((m, n), dtype=torch.float32)  # 真实实现(fbgemm)用float32类型

    buffer += qinput @ qweight.T
    buffer -= qinput @ z_w.T   # 因为权重采用的是对称量化, 所以 z_w 实际上是 0, 因此可跳过这一部分
    buffer -= z_x @ qweight.T  # 可提前算好
    buffer += z_x @ z_w.T      # 可提前算好

    output = buffer.double() * s_w * s_x + bias
    output = output.to(torch.float32)  # 真实实现里, 这里是回到 input.dtype
    return output

# 使用 pytorch quantization API 的高级接口
def torch_dynamic_quantization_forward(layer, inp):
    # fbgemm: per tensor quantize weight (qint8) and input (quint8)
    qmodel = quantize_dynamic(model=torch.nn.Sequential(layer), qconfig_spec={nn.Linear}, dtype=torch.qint8, inplace=False)
    output = qmodel(inp)
    return output


# 不使用任何 pytorch quantization API 的实现
def manual_dynamic_quantization_forward(layer, inp):
    weight = layer.weight
    bias = layer.bias
    # step 1 权重量化, 参考: torch.ao.quantization.observer.MinMaxObserver, 注意取整的方式(四舍五入 vs 向零取整)
    qweight = quantize_linear_weight(weight, symmetric=True)
    scale, zero_point, quant_min, quant_max = calculate_qparams_linear_weight(weight, symmetric=True)
    weight_qparams = (scale, zero_point)
    # deqweight = dequantize_linear_weight(qweight, scale, zero_point, quant_min, quant_max)
    
    # step 2 输入量化, 参考: C++ 源码: https://github.com/pytorch/pytorch/blob/v2.0.0/aten/src/ATen/native/quantized/cpu/qlinear_dynamic.cpp#L31
    qinput = quantize_linear_weight(inp, dtype=torch.quint8, symmetric=False, reduce_range=True)
    scale, zero_point, quant_min, quant_max = calculate_qparams_linear_weight(inp, dtype=torch.quint8, symmetric=False, reduce_range=True)
    input_qparams = (scale, zero_point)
    # deqinput = dequantize_linear_weight(qinput, scale, zero_point, quant_min, quant_max)

    # step 3 量化后的输入与量化后权重进行计算并还原为 float32, 并加上偏置项
    output = gemm(qweight, qinput, bias, weight_qparams, input_qparams)
    return output

if __name__ == "__main__":
    batch_size, in_features, out_features = 20, 30, 40
    layer = torch.nn.Linear(in_features, out_features)
    inp = torch.distributions.normal.Normal(0, 1).sample((batch_size, in_features))
    # 测试权重量化的正确性
    test_linear_weight_quantize(layer)

    # 测试手工实现与pytorch的差异
    torch_output = torch_dynamic_quantization_forward(layer, inp)
    manual_output = manual_dynamic_quantization_forward(layer, inp)
    print("测试手工实现与pytorch的差异:", (torch_output - manual_output).abs().max().item())  # 1.1920928955078125e-07
```


### fx mode (TODO)


## Static Quantization(TODO)


## QAT (tensorflow)

参考 [Release](https://github.com/tensorflow/model-optimization/releases?page=2) 列表, 相关版本: `tensorflow==2.0.0`, `model_optimization==0.3.0`

```python
# tensorflow_model_optimization/python/core/quantization/keras/quantize.py
def quantize_model(to_quantize: tf.keras.Model):
    def _add_quant_wrapper(layer):
        return quantize_annotate_mod.QuantizeAnnotate(layer)
    annotated_model = tf.keras.models.clone_model(to_annotate, input_tensors=None, clone_function=_add_quant_wrapper)
    return quantize_apply(annotated_model)
```

## QAT (pytorch)

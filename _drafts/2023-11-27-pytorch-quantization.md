---
layout: post
title: "(WIP) Pytorch Quantization"
date: 2023-11-27 11:10:04 +0800
---

## 总览

本文主要参考资料:

- Pytorch 的一篇指导性的博客 (食用指南! 可快速上手使用转化为生产力, 读者如果仅出于使用目的可以只看这篇博客, 本文后续内容均可不看): [https://pytorch.org/blog/quantization-in-practice/](https://pytorch.org/blog/quantization-in-practice/)
- 官方支持量化的博客 (内含 3 种量化模式的上层 API, 但不是完整可运行示例, 也不包括后续版本增加的 fx mode 量化): [https://pytorch.org/blog/introduction-to-quantization-on-pytorch/](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
- 官方文档 (需要仔细琢磨): [https://pytorch.org/docs/2.1/quantization.html](https://pytorch.org/docs/2.1/quantization.html)
- 官方 API 文档 (因为 Pytorch 提供了 3 种量化方式, 以及 eager/fx 模式, 并且 API 分为上层 API 和底层 API, 所以显得比较混乱, 个人还感觉 Pytorch 量化方面暴露的底层接口似乎不算完善): [https://pytorch.org/docs/2.1/quantization-support.html](https://pytorch.org/docs/2.1/quantization-support.html)
- Huggingface Optimum 的一篇关于模型量化的总览介绍: [https://huggingface.co/docs/optimum/v1.16.1/en/concept_guides/quantization](https://huggingface.co/docs/optimum/v1.16.1/en/concept_guides/quantization)
- Pytorch wiki (包含了关于底层诸如 `torch.qint8` 数据类型的张量的一些 API): [https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor](https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor)
- 一篇详细介绍数学公式推导的博客, 非常值得仔细研究: [https://leimao.github.io/article/Neural-Networks-Quantization/](https://leimao.github.io/article/Neural-Networks-Quantization/)

Pytorch Tutorials (一些端到端的例子):

- 官方 tutorial 搜索 (端到端的示例): [https://pytorch.org/tutorials/search.html?q=quantization&check_keywords=yes&area=default](https://pytorch.org/tutorials/search.html?q=quantization&check_keywords=yes&area=default)
- Static Quantization + QAT: [https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)

相关内容 (本文可能不会过多涉及):

- QLoRA: [https://huggingface.co/blog/4bit-transformers-bitsandbytes](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- FP8 paper: [FP8 formats for deep learning](https://arxiv.org/pdf/2209.05433.pdf): 英伟达 H100 引入, 文中写 E4M3 的最大值是 448, 但笔者按 IEEE 754 算是 240, 其余均吻合. 原因是 E4M3 不完全遵循 IEEE 754, 而 E5M2 遵循 IEEE 754 (参考: [博客](https://lambdalabs.com/blog/nvidia-hopper-h100-and-fp8-support))
- 一篇关于 QAT 的知乎[博客](https://zhuanlan.zhihu.com/p/548174416), 博客中有原论文及原Tensorflow实现的, Pytorch 的实现包含在本文内容中. 如果要分析 QAT 的原始 TensorFlow 实现, 主要看这个端到端的[例子](https://www.tensorflow.org/model_optimization/guide/quantization/training_example), 以及入口[源码](https://github.com/tensorflow/model-optimization/blob/v0.3.0/tensorflow_model_optimization/python/core/quantization/keras/quantize.py#L80), 这些代码与博客中的分析也基本一致.
- 一篇基于Pytorch官方博客的笔记: [博客园笔记](https://www.cnblogs.com/LXP-Never/p/16822727.html)

Pytorch 原生量化支持有三类:

- Post-Training Dynamic Quantization: 原理上是提前将权重转化为 int8, 在计算时, 每一层的输入先由浮点数转化为 int8 (量化过程的 `max_val` 和 `min_val` 动态决定), 之后用 int8 的输入与 int8 的权重进行矩阵乘法或卷积等运算, 然后将输出转换回浮点数. 因为每一层都需要动态计算出 `max_val` 和 `min_val`, 并且需要不断地对 activation 进行 int8 与浮点数之间的转换, 因此加速并不明显.
- Post-Training Static Quantization: 原理上是模型训练好后, 首先将权重转换为 int8, 然后给模型喂入一批数据, 计算每层输入的分布情况, 由此得到每一层输出的 `min_val` 和 `max_val`, 因此初看上去, 可以节约动态计算 `min_val` 和 `max_val` 的时间, 然而实际上, 这种做法可以允许整个网络每层之间不必要进行 activation 的 int8 与浮点数之间的转换(为什么?), 所以可以获得比较大的加速.
- Quantization Aware Training: 训练过程中就加入量化损失

## 注意事项

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


## 底层接口

本节只介绍一部分底层接口, 其余底层接口与具体的量化算法结合起来在后续章节介绍.

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

### 使用方式

涉及如下 3 个高级 API:

- eager mode: `torch.quantization.quantize_dynamic`
- fx mode: `torch.quantization.quantize_fx.prepare_fx`, `torch.quantization.quantize_fx.convert_fx`

以下是一个完整的示例, 参考自[这里](https://pytorch.org/blog/quantization-in-practice/#post-training-dynamicweight-only-quantization), 运行环境: torch==2.0.0, python 3.10

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

我们回过头来搞清 `forward` 方法中使用到的 `torch.ops.quantized.linear_dynamic` 的具体算法细节, 怎么找 C 源码呢? 根据[README.md](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/quantized/README.md) 的指引, 注意到这两个文件:

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
- 分配输出结果的内存空间 `output` (float32 类型, 源码中可以见到 `at::kFloat` 这样的代码) 和计算缓冲空间 `buffer` (int32 类型, 源码中可以见到诸如 `at::kInt`, `buffer.data_ptr<int32_t>` 这样的代码)
    ```c++
    auto output = at::empty(out_sizes, input.options().dtype(at::kFloat));
    auto buffer = at::empty_like(output, output.options().dtype(at::kInt), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    // ...
    return output
    ```
- 然后调用 `fbgemm::fbgemmPacked` 进行计算: 此算子输入时量化后的输入 (uint8) 与量化权重 (int8) 及偏置 (float32), 输出为 float32 类型, 从以下摘录的源码及注释可以看出, 实际的计算过程是先执行 uint8 与 int8 的矩阵乘法, 计算结果累积在 int32 的 `buffer` 上, 然后转换会 float32, 最后加上 float32 的偏置
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

由于继续深入 `fbgemm::fbgemmPacked` 有些过于琐碎(没能力看懂), 因此这里给出其[源码位置](https://github.com/pytorch/FBGEMM/blob/v0.5.0/src/Fbgemm.cc#L29)与之等价的 python 实现【TODO】

```python
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic

batch_size, in_features, out_features = 2, 3, 4
layer = torch.nn.Linear(in_features, out_features)
inp = torch.distributions.normal.Normal(0, 1).sample((batch_size, in_features))

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
    observer = torch.ao.quantization.observer.MinMaxObserver()
    observer(weight)
    # step 2 输入量化, 参考: C++ 源码
    # TODO
    # step 3 输入与权重进行矩阵乘法:
    # TODO
    # step 4 矩阵乘法结果还原为 float32, 并加上偏置项
    # TODO
    return output
```


### fx mode (TODO)


## Static Quantization(TODO)


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

## QAT (pytorch)

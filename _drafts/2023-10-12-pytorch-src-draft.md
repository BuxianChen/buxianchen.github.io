---
layout: post
title: "(WIP) Pytorch 源码学习 (草稿杂录)"
date: 2023-10-12 10:20:24 +0800
labels: [pytorch]
---

## 动机、参考资料、涉及内容

### 动机

本篇博客主要是记录一些与 Pytorch 相关的但**不成体系**的内容, 目标是学习 **Pytorch 2.1.0** 版本的底层代码, 后续再考虑将相关内容成体系地整理.

### 参考资料

在此处列举的参考资料默认包含其本身对其他参考资料的引用, 此列表列举的参考资料尽量完整.

### 涉及内容



### 小目标

预估以目前的更新速度, Pytorch 会在 2024 年底更新至 2.5 版本左右, 但本篇博客及未来成体系整理的相关博客只力图研究 Pytorch 2.1.0 版本保底, 后续版本看实际情况


## 正文

记录内容直接用横线隔开, 不做过多的结构体系整理

<hr/>

环境配置: 

CUDA Driver 升级到: 使用 `nvidia-smi` 命令查看

```
NVIDIA-SMI 535.54.04
Driver Version: 536.23
CUDA Version: 12.2
```

```python
conda create --name torch2.1 python=3.10
pip install torch==2.1.0
```

<hr/>

源码安装: 浅克隆一个 tag

```bash
git clone --branch v2.1.0 --depth=1 --recurse-submodules --shallow-submodules https://github.com/pytorch/pytorch
```

<hr/>

`aten/src/README.md`

```
* TH = TorcH
* THC = TorcH Cuda
* THCS = TorcH Cuda Sparse (now defunct)
* THNN = TorcH Neural Network (now defunct)
* THS = TorcH Sparse (now defunct)
```

defunct 是指已经不存在了, 备注有些网上的资料会说有 TH, THC, THNN, THCNN 这四个文件夹

<hr/>

`torch/csrc/autograd/variable.cpp` 中:

```cpp
namespace torch {
namespace autograd {

/// `Variable` is exactly the same as `Tensor` (i.e. we have `using Variable =
/// at::Tensor`). This means you can perform all the usual mathematical and
/// other operations you can perform on `Tensor`s also on `Variable`s.
///
/// The only reason we are keeping the `Variable` class is backward
/// compatibility with external user's legacy C++ frontend code. Our intention
/// is to eliminate the `Variable` class in the near future.
using Variable = at::Tensor;

} // namespace autograd
} // namespace torch
```

这似乎说明: `Variable` 和 `Tensor` 是完全一样的东西了, 在 python 中进行验证

```python
import torch
torch.autograd.Variable is torch.Tensor  # 
```

<hr/>

找 `torch.autograd.Function.apply` (python API) 方法的定义

```
torch.autograd.Function 的一个元类: torch._C._FunctionBase, 源码上有一行注释显示源代码在 torch/csrc/autograd/python_function.cpp
然后找到上面这份代码底部: 这里主要是 pybind11 的用法
bool THPFunction_initModule(PyObject* module) {
  if (PyType_Ready(&THPFunctionType) < 0)
    return false;
  Py_INCREF(&THPFunctionType);
  PyModule_AddObject(module, "_FunctionBase", (PyObject*)&THPFunctionType);
  return true;
}
于是找到 THPFunctionType, 它是一个数据结构, 数据项定义中包含一行:
THPFunction_methods;
于是在找到它的定义, 其中有一行:
{(char*)"apply", THPFunction_apply, METH_CLASS | METH_VARARGS, nullptr}
所以代码在 torch/csrc/autograd/python_function.cpp 的 THPFunction_apply 里
```

总结一下:

- `torch._C._FunctionBase` (python API) 就是 `torch/csrc/autograd/python_function.cpp:THPFunctionType` (C++ API)
- `torch.autograd.Function.apply` (python API) 就是 `torch/csrc/autograd/python_function.cpp:THPFunction_apply` (C++ API)

<hr/>

Pytorch 对 tensor 进行 index 操作, 行为是一致的

> When accessing the contents of a tensor via indexing, PyTorch follows Numpy behaviors that basic indexing returns views, while advanced indexing returns a copy. Assignment via either basic or advanced indexing is in-place. See more examples in [Numpy indexing documentation](https://numpy.org/doc/stable/reference/arrays.indexing.html).

```python
import torch
x = torch.arange(4)
y = x[:3]
z = x[[0, 1, 2]]
w = x[1:3]

# 这三者相同
x.storage().data_ptr()
y.storage().data_ptr()
w.storage().data_ptr()

# 这个跟前面的不一样
z.storage().data_ptr()

# 注意: 这种看上去是copy也会影响原始的 x, 估计是 Tensor.__getitem__ 与 Tensor.__setitem__ 语义有区别
x[[1, 2]] = 3
x  # [0, 3, 3]
```

<hr/>

一个 tensor, 内部最重要的属性包含:

- `size`: tuple of int
- `strides`: tuple of int
- `offset`: int, 通常是 0
- `array` 或者说 `data_ptr`: 实际数据的指针

备注: 由于 `strides` 属性, 所以 advanced indexing 时一定是做成 copy 返回

```python
import torch
x = torch.arange(48).view((6, 8)).contiguous()
y = x[[1, 0, 4, 4]]  # 如果 y 是 view 的话, 就没法定义 y 的 strides 了
```

一个具体的例子:

```python
import torch
x = torch.arange(48).view((6, 8)).contiguous()
y = x[1:5:2, 1:4]
i1, j1, k1 = 1, 5, 2
i2, j2, k2 = 1, 4, 1

x_stride_1, x_stride_2 = x.stride()  # (8, 1)
x.storage_offset()                   # 0

y_stride_1, y_stride_2 = y.stride()  # (16, 1)
y.storage_offset()                   # 9
y.size()                             # (2, 3)

x.storage().data_ptr() == y.storage().data_ptr()
y_stride_1 == x_stride_1 * k1
y_stride_2 == x_stride_2 * k2
y.storage_offset() ==  x_stride_1 * i1 + x_stride_2 * i2
```

[https://ezyang.github.io/stride-visualizer/index.html](https://ezyang.github.io/stride-visualizer/index.html): broadcast 的实现是 stride 为 0?
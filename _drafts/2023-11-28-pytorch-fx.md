---
layout: post
title: "(WIP) torch.fx"
date: 2023-11-28 11:10:04 +0800
labels: [pytorch]
---

## 概述

**参考资料**

- 官方文档: [https://pytorch.org/docs/stable/fx.html](https://pytorch.org/docs/stable/fx.html)
- 论文: [torch.fx 论文](https://arxiv.org/abs/2112.08429)
- 知乎源码分析: [博客](https://zhuanlan.zhihu.com/p/625690498), 笔者结合这篇博客并根据自己阅读源码反复琢磨才自觉基本理解了这篇博客
- 知乎的一篇简单分析 torch 的三种计算图捕获技术 (`torch.fx`, `torch.jit`, `torch.compiler`) 的博客: [博客](https://zhuanlan.zhihu.com/p/644590863), 此篇博客的作者还写了一些关于 Pytorch 2.0 的一些官方文档, 知乎上也有一些相关文章

**torch.fx 应用例子**

- [pytorch Conv+BN fuse](https://github.com/pytorch/pytorch/blob/main/torch/fx/experimental/optimization.py#L50)
- [fx graph mode quantization](https://pytorch.org/docs/master/quantization.html#prototype-fx-graph-mode-quantization)
- [huggingface optimum](https://huggingface.co/docs/optimum/main/en/torch_fx/usage_guides/optimization): 但似乎没有具体的用例

**简介**

AI 编译器的目标是希望把现成的模型提升运行速度或吞吐量, 以笔者目前拙见一般有几种方式:

- 方式1: 将原始用 pytorch 写的 python 代码完全"重写".
- 方式2: 将原始用 pytorch 写的 python 代码适当改写(改写后仍然是 Python 代码), 使得编译器能直接处理改写后的代码. 因此, 做 AI 编译器的人总是希望能尽量少地该原始代码, 或者说尽量扩大编译器能处理的东西.

由于 pytorch 是 python 优先的, 为了做AI模型的静态加速, 首先是将"杂乱无章"的 python 代码转换为结构化的计算图(中间表示), 这一过程也就是所谓的"计算图捕获", 然后再执行相应的编译过程. 至今为止, pytorch 自身在这方面做了 `torch.jit` (pytorch1.0), `torch.fx`(pytorch1.8), `torch.compiler`(pytorch2.0) 的探索.

`torch.fx` 模块的主要用途是使用一种方式 (`torch.fx.Tracer`, 不同于 `torchscript`) 分析源代码, 得到中间表示 `torch.fx.Graph`, 然后程序员可以修改这个中间表示, 最后重新转换回转换后的 python 代码. 也就是说可以完成这个 pipeline: `symbolic tracing -> intermediate representation -> transforms -> Python code generation`, 作用如下:

- 假设现在有很多包含 `nn.Module` 的代码(例如像 huggingface transformers, timm, mmdetection 这种代码库), 现在希望对这些代码做统一的转换, 则可以走上面的完整 pipeline 来达到目的
- 可以直接写一个中间表示的配置文件, 然后直接从配置文件转换为 python 代码

总的来说, 就是可以用比较 hack 的方式修改模型, 下面是一个例子:

这里对于一个已经写好的 `M`, 我希望将 `torch.add` 操作全部替换为 `torch.mul`, 但是又不希望修改 `M` 的源代码

```python
import torch
import torch.fx

class M(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)

def transform(m: torch.nn.Module,
              tracer_class: type = fx.Tracer) -> torch.nn.Module:
    graph : fx.Graph = tracer_class().trace(m)
    for node in graph.nodes:
        if node.op == 'call_function':
            if node.target == torch.add:
                node.target = torch.mul
    graph.lint()

    return fx.GraphModule(m, graph)

# GraphModule 也是 nn.Module
m: torch.fx.GraphModule = transform(M())
```

更多例子请参考官方文档: [https://pytorch.org/docs/stable/fx.html](https://pytorch.org/docs/stable/fx.html)

本篇博客的写作目的如下:

- 笔者最初只是好奇 torch 的量化算法, 其文档中提到了一种自动的用法, 其基于 `torch.fx`, 因此笔者转而研究 `torch.fx`
- 仅仅出于好奇, `torch.fx` 是怎么捕获计算图的
- 笔者 `torch.fx` 的官方文档时, 还是感到很多地方无法理解, 因此可能需要源码理解
- `torch.fx` 的局限性有哪些, 哪些可以通过它的高级特性突破这些局限性, 因此也不得不研究源码

## 实现原理概览及 minimal `torch.fx`

TODO: `Graph.python_code` 的实现尚不完善, 整体代码行数也稍多(目前200多行, 最好能减到200行), 有些地方还需优化

[https://github.com/BuxianChen/snippet/blob/master/fx/simple_fx.py](https://github.com/BuxianChen/snippet/blob/master/fx/simple_fx.py)

## 源码阅读预备知识

### `__torch_function__` 协议: TODO

此协议大致如下: 当调用 `torch.max`, `torch.nn.functional.softmax` 时, 会优先根据输入参数检查是否带有 `__torch_function__`, 如果带有则会优先将实参进行适当的"转换", 触发对 `__torch_function__` 的调用.

`torch.nn.functional.softmax` 源码(`torch==1.8.0`)如下, 以供参考 (`torch.max` 似乎在 C 代码中实现, 暂时不深究)

```python
# torch/nn/functional.py
def softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    # 此分支适用于包含了__torch_function__的自定义类型
    if has_torch_function_unary(input):  # torch._C.has_torch_function_unary
        return handle_torch_function(softmax, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    if dim is None:
        dim = _get_softmax_dim("softmax", input.dim(), _stacklevel)
    # 当 input 是 Tensor 类型时, 会执行下面的逻辑
    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    return ret

# torch/overrides.py
def handle_torch_function(public_api: Callable, relevant_args: Iterable[Any], *args, **kwargs) -> Any:
    overloaded_args = _get_overloaded_args(relevant_args)

    types = tuple(map(type, overloaded_args))
    for overloaded_arg in overloaded_args:
        result = overloaded_arg.__torch_function__(public_api, types, args, kwargs)  # 此处进入 __torch_function__
        if result is not NotImplemented:
            return result

    func_name = '{}.{}'.format(public_api.__module__, public_api.__name__)
    raise TypeError("no implementation found for '{}' on types that implement __torch_function__: {}".format(func_name, [type(arg) for arg in overloaded_args]))
```


### `__getattr__`, `__getattribute__` 相关

```python
class M:
    def __init__(self):
        self.a = 1
    # 第一优先级, 无论是取属性还是调用方法都会被触发
    def __getattribute__(self, name):
        print("__getattribute__")
        return object.__getattribute__(self, name)  # 不能用super,会造成无限递归
    # __getattribute__ 找不到时会被触发
    def __getattr__(self, name):
        print("__getattr__")
        return None
    def __call__(self, x):
        print("__call__")
    def foo(self):
        print("foo method")

m = M()
print("="*20)
print("trace: m(1)")
m(1)

print("="*20)
print("trace: m.__call__(1)")
m.__call__(1)

print("="*20)
print("trace: m.foo()")
m.foo()

print("="*20)
print("trace: m.foo")
m.foo

print("="*20)
print("trace: m.a")
m.a

print("="*20)
print("trace: m.b")
m.b
```

输出

```
====================
trace: m(1)
__call__
====================
trace: m.__call__(1)
__getattribute__
__call__
====================
trace: m.foo()
__getattribute__
foo method
====================
trace: m.foo
__getattribute__
====================
trace: m.a
__getattribute__
====================
trace: m.b
__getattribute__
__getattr__
```

**`torch.nn.Module.__getattr__`**

`Module` 重载了 `__getattr__` 方法: 其主要逻辑是按 `_parameters`, `_buffers`, `_modules` 的顺序搜索属性

```python
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_a = torch.nn.Linear(3, 3)  # 实际上会被存储在 self._modules 中, 而不是 self.__dict__['layer_a']
    def __getattr__(self, name):
        print("Module.__getattr__ called")
        return super().__getattr__(name)
model = MyModule()
layer = model.layer_a  # Module.__getattr__ called
```

### Code Object

Python 程序的运行是由 Python 解释器来执行的, 尽管 Python 通常被认为是解释型语言, 但实际上 Python 代码 (`.py` 文件) 会在执行前被编译为字节码(即那些 `.pyc` 文件, 并非可执行代码, 但 Python 解释器能读懂), 然后由 Python 虚拟机解释执行. 这一点部分类似于 Java 语言的 Java 虚拟机 (JVM). 出于本文的目的, 此话题暂时不过多深究.

从 Python 使用者的角度来看, 我们时常与函数打交道, 而实际上, 函数也是对象, 因此它本身也有类型, 其类型是 `types.FunctionType`. 通常来说, 我们使用 `def` 语句来创建函数, 也就是创建函数对象, 或者说是 `types.FunctionType` 的实例, 例如:

```python
import types
def foo(): pass
type(foo) is types.FunctionType       # True, 函数对象
type(foo.__code__) is types.CodeType  # True, Code 对象
```

因此, 我们可以想象, 应该存在着直接使用 `types.FunctionType` 的构造方法得到函数对象的方式 (简单提及一下, 实际上也可以使用 `type` 函数来定义一个我们平时用 `class` 关键字的方式来定义类), 事实上, 在 `Tracer.trace` 的实现中包含了 `inspect`, `fn.__code__` 的综合应用, 例如:

- 使用 `type(fn.__code__)(...)`, 即 `types.CodeType(...)` 的方式手动构建代码对象
- 使用 `type(fn)(fn.__code__, ...)`, 即 `types.FunctionType(...)` 的方式构建函数 (我们通常是使用 `def` 语句来完成这两个过程)

这里给出 Python 3.9 版本的 `CodeType`, `FunctionType` 的构造函数信息

```python
import types
print(types.CodeType.__doc__)
print(types.FunctionType.__doc__)
```

输出结果

```
code(argcount, posonlyargcount, kwonlyargcount, nlocals, stacksize,
      flags, codestring, constants, names, varnames, filename, name,
      firstlineno, lnotab[, freevars[, cellvars]])

Create a code object.  Not for the faint of heart.


Create a function object.

  code
    a code object
  globals
    the globals dictionary
  name
    a string that overrides the name from the code object
  argdefs
    a tuple that specifies the default argument values
  closure
    a tuple that supplies the bindings for free variables
```

系统化地研究 Python AST 以及字节码相关的内容, 主要涉及的内容包括 `ast`, `dis` 模块, 以下是一些参考资料

- python 官方文档: Language Reference?
- [ast 模块官方文档](https://docs.python.org/3/library/dis.html)
- [ast 详解文章](https://greentreesnakes.readthedocs.io/en/latest/index.html)

其他:
- [https://towardsdatascience.com/understanding-python-bytecode-e7edaae8734d](https://towardsdatascience.com/understanding-python-bytecode-e7edaae8734d)
- [https://medium.com/@noransaber685/demystifying-python-bytecode-a-guide-to-understanding-and-analyzing-code-execution-6a163cb83bd1](https://medium.com/@noransaber685/demystifying-python-bytecode-a-guide-to-understanding-and-analyzing-code-execution-6a163cb83bd1)
- [https://stackoverflow.com/questions/16064409/how-to-create-a-code-object-in-python](https://stackoverflow.com/questions/16064409/how-to-create-a-code-object-in-python)
- [https://github.com/python/cpython/blob/3.8/Include/code.h](https://github.com/python/cpython/blob/3.8/Include/code.h)

**浅尝内置函数 `compile`, `dis.dis`, `fn.__code__` 的基本用法**

示例代码 1:

```python
import dis
code_string = """
def countdown(n):
    while n > 0:
        print('T-minus', n)
        n -= 1
    print('Blastoff!')
"""
# 这种用法在 GraphModule.recompile 方法中有用到:
# 根据 graph 属性生成 python 代码, 然后由这份 python 代码编译出字节码, 最后由字节码及其他信息得到 forward 函数
code = compile(code_string, "test", "exec")
dis.dis(code)
```

输出:

```
  2           0 LOAD_CONST               0 (<code object countdown at 0x7f8b55a74450, file "test", line 2>)
              2 LOAD_CONST               1 ('countdown')
              4 MAKE_FUNCTION            0
              6 STORE_NAME               0 (countdown)
              8 LOAD_CONST               2 (None)
             10 RETURN_VALUE

Disassembly of <code object countdown at 0x7f8b55a74450, file "test", line 2>:
  3     >>    0 LOAD_FAST                0 (n)
              2 LOAD_CONST               1 (0)
              4 COMPARE_OP               4 (>)
              6 POP_JUMP_IF_FALSE       28

  4           8 LOAD_GLOBAL              0 (print)
             10 LOAD_CONST               2 ('T-minus')
             12 LOAD_FAST                0 (n)
             14 CALL_FUNCTION            2
             16 POP_TOP

  5          18 LOAD_FAST                0 (n)
             20 LOAD_CONST               3 (1)
             22 INPLACE_SUBTRACT
             24 STORE_FAST               0 (n)
             26 JUMP_ABSOLUTE            0

  6     >>   28 LOAD_GLOBAL              0 (print)
             30 LOAD_CONST               4 ('Blastoff!')
             32 CALL_FUNCTION            1
             34 POP_TOP
             36 LOAD_CONST               0 (None)
             38 RETURN_VALUE
```

示例代码 2:

```python
import dis
def countdown(n):
    while n > 0:
        print('T-minus', n)
        n -= 1
    print('Blastoff!')
dis.dis(countdown)

code_obj = countdown.__code__
names = [name for name in code_obj.__dir__() if name.startswith("co_")]

for name in names:
    print(name)
    print(getattr(code_obj, name))
    print("-"*40)
```

输出:

```
  2     >>    0 LOAD_FAST                0 (n)
              2 LOAD_CONST               1 (0)
              4 COMPARE_OP               4 (>)
              6 POP_JUMP_IF_FALSE       28

  3           8 LOAD_GLOBAL              0 (print)
             10 LOAD_CONST               2 ('T-minus')
             12 LOAD_FAST                0 (n)
             14 CALL_FUNCTION            2
             16 POP_TOP

  4          18 LOAD_FAST                0 (n)
             20 LOAD_CONST               3 (1)
             22 INPLACE_SUBTRACT
             24 STORE_FAST               0 (n)
             26 JUMP_ABSOLUTE            0

  5     >>   28 LOAD_GLOBAL              0 (print)
             30 LOAD_CONST               4 ('Blastoff!')
             32 CALL_FUNCTION            1
             34 POP_TOP
             36 LOAD_CONST               0 (None)
             38 RETURN_VALUE

co_argcount
1
----------------------------------------
co_posonlyargcount
0
----------------------------------------
co_kwonlyargcount
0
----------------------------------------
co_nlocals
1
----------------------------------------
co_stacksize
3
----------------------------------------
co_flags
67
----------------------------------------
co_code
b'|\x00d\x01k\x04r\x1ct\x00d\x02|\x00\x83\x02\x01\x00|\x00d\x038\x00}\x00q\x00t\x00d\x04\x83\x01\x01\x00d\x00S\x00'
----------------------------------------
co_consts
(None, 0, 'T-minus', 1, 'Blastoff!')
----------------------------------------
co_names
('print',)
----------------------------------------
co_varnames
('n',)
----------------------------------------
co_freevars
()
----------------------------------------
co_cellvars
()
----------------------------------------
co_filename
/tmp/ipykernel_2353/2861170246.py
----------------------------------------
co_name
countdown
----------------------------------------
co_firstlineno
1
----------------------------------------
co_lnotab
b'\x00\x01\x08\x01\n\x01\n\x01'
```

**co_flag**

注意到上例中 `co_flags` 为 `67`, 应该转为 2 进制进行理解, 参照[Python官方文档](https://docs.python.org/3/library/inspect.html#code-objects-bit-flags)和[CPython源码](https://github.com/python/cpython/blob/3.8/Include/code.h)

```python
import inspect

# inspect.CO_OPTIMIZED:   1, 一般都为1
# inspect.CO_NEWLOCALS:   2
# inspect.CO_VARARGS:     4, 代表是否有可变参数 *args
# inspect.CO_VARKEYWORDS: 8, 代表是否有可变参数 **kwargs
# inspect.CO_NESTED:     16,
# inspect.CO_GENERATOR   32,
# inspect.CO_NOFREE      64, inspect 文档中没有记录, 可参考: https://github.com/python/cpython/blob/3.8/Include/code.h#L83
# inspect.CO_COROUTINE   128,
# inspect.CO_ITERABLE_COROUTINE 256,
# inspect.CO_ASYNC_GENERATOR 512
# ...

bin(67)
'0b1000011'
```

**co_code 与 dis.Bytecode 的关系**

从下面的例子可以看出对应关系如下: **在 Python 3.7 版本之后, `co_code` 每两个字节代表一条指令, 其中第一个字节代表 opname, 第二个字节代表操作数 arg**, 全部的指令集为: `dis.opname`

```python
import dis
print(list(enumerate(dis.opname)))
[(0, '<0>'), (1, 'POP_TOP'), (2, 'ROT_TWO'), (3, 'ROT_THREE'), (4, 'DUP_TOP'), (5, 'DUP_TOP_TWO'), (6, 'ROT_FOUR'), (7, '<7>'), ...]
```

验证 `__code__.co_code` 与指令的一一对应关系:

```python
print(countdown.__code__.co_code)
# b'|\x00d\x01k\x04r\x1ct\x00d\x02|\x00\x83\x02\x01\x00|\x00d\x038\x00}\x00q\x00t\x00d\x04\x83\x01\x01\x00d\x00S\x00'
bytecode = dis.Bytecode(countdown)
for instuction in bytecode:
    print(instuction)

for i, x in enumerate(countdown.__code__.co_code):
    if i % 2 == 0:
        print(x, end="\t")
    else:
        print(x, end="\n")
```

输出:

```
Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='n', argrepr='n', offset=0, starts_line=2, is_jump_target=True)
Instruction(opname='LOAD_CONST', opcode=100, arg=1, argval=0, argrepr='0', offset=2, starts_line=None, is_jump_target=False)
Instruction(opname='COMPARE_OP', opcode=107, arg=4, argval='>', argrepr='>', offset=4, starts_line=None, is_jump_target=False)
Instruction(opname='POP_JUMP_IF_FALSE', opcode=114, arg=28, argval=28, argrepr='', offset=6, starts_line=None, is_jump_target=False)
Instruction(opname='LOAD_GLOBAL', opcode=116, arg=0, argval='print', argrepr='print', offset=8, starts_line=3, is_jump_target=False)
Instruction(opname='LOAD_CONST', opcode=100, arg=2, argval='T-minus', argrepr="'T-minus'", offset=10, starts_line=None, is_jump_target=False)
Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='n', argrepr='n', offset=12, starts_line=None, is_jump_target=False)
Instruction(opname='CALL_FUNCTION', opcode=131, arg=2, argval=2, argrepr='', offset=14, starts_line=None, is_jump_target=False)
Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=16, starts_line=None, is_jump_target=False)
Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='n', argrepr='n', offset=18, starts_line=4, is_jump_target=False)
Instruction(opname='LOAD_CONST', opcode=100, arg=3, argval=1, argrepr='1', offset=20, starts_line=None, is_jump_target=False)
Instruction(opname='INPLACE_SUBTRACT', opcode=56, arg=None, argval=None, argrepr='', offset=22, starts_line=None, is_jump_target=False)
Instruction(opname='STORE_FAST', opcode=125, arg=0, argval='n', argrepr='n', offset=24, starts_line=None, is_jump_target=False)
Instruction(opname='JUMP_ABSOLUTE', opcode=113, arg=0, argval=0, argrepr='', offset=26, starts_line=None, is_jump_target=False)
Instruction(opname='LOAD_GLOBAL', opcode=116, arg=0, argval='print', argrepr='print', offset=28, starts_line=5, is_jump_target=True)
Instruction(opname='LOAD_CONST', opcode=100, arg=4, argval='Blastoff!', argrepr="'Blastoff!'", offset=30, starts_line=None, is_jump_target=False)
Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='', offset=32, starts_line=None, is_jump_target=False)
Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=34, starts_line=None, is_jump_target=False)
Instruction(opname='LOAD_CONST', opcode=100, arg=0, argval=None, argrepr='None', offset=36, starts_line=None, is_jump_target=False)
Instruction(opname='RETURN_VALUE', opcode=83, arg=None, argval=None, argrepr='', offset=38, starts_line=None, is_jump_target=False)

124	0
100	1
107	4
114	28
116	0
100	2
124	0
131	2
1	0
124	0
100	3
56	0
125	0
113	0
116	0
100	4
131	1
1	0
100	0
83	0
```

**函数字节码变换例子: 将加法改为乘法**

```python
import types
import dis
import struct

# 原始函数
def original_function(a, b):
    return a + b - a

# 获取原始函数的字节码对象
original_bytecode = original_function.__code__

# 输出原始函数的字节码指令
print("Original bytecode:")
dis.dis(original_bytecode)

# 测试修改前的函数
result = original_function(2, 3)
print("Result of original function:", result)

idx2op = {idx: op for idx, op in enumerate(dis.opname)}
op2idx = {op: idx for idx, op in enumerate(dis.opname)}

# 修改字节码
new_code = []
for i, x in enumerate(original_bytecode.co_code):
    c = x
    if i % 2 == 0:
        if idx2op[x] == "BINARY_ADD":
            c = op2idx["BINARY_MULTIPLY"]
    new_code.append(c)
n = len(new_code)
new_code = struct.pack(f"{n}B", *new_code)


# 构造修改后的代码对象
modified_code = types.CodeType(
    original_bytecode.co_argcount,       # 参数数量
    original_bytecode.co_posonlyargcount,# 限定位置参数
    original_bytecode.co_kwonlyargcount, # 关键字参数数量
    original_bytecode.co_nlocals,        # 局部变量数量
    original_bytecode.co_stacksize,      # 栈大小
    original_bytecode.co_flags,          # 标志
    new_code,                            # 修改后的字节码
    original_bytecode.co_consts,         # 常量
    original_bytecode.co_names,          # 名称
    original_bytecode.co_varnames,       # 变量名
    original_bytecode.co_filename,       # 文件名
    original_bytecode.co_name,           # 函数名
    original_bytecode.co_firstlineno,    # 第一行行号
    original_bytecode.co_lnotab,         # 行号表
    original_bytecode.co_freevars,       # 自由变量
    original_bytecode.co_cellvars        # Cell 变量
)

# 构造修改后的函数
modified_function = types.FunctionType(
    modified_code,
    original_function.__globals__,
    original_function.__name__,
    original_function.__defaults__,
    original_function.__closure__
)


print("Modified bytecode:")
dis.dis(modified_function.__code__)

# 测试修改后的函数
result = modified_function(2, 3)
print("Result of modified function:", result)
```

输出结果

```
Original bytecode:
  7           0 LOAD_FAST                0 (a)
              2 LOAD_FAST                1 (b)
              4 BINARY_ADD
              6 LOAD_FAST                0 (a)
              8 BINARY_SUBTRACT
             10 RETURN_VALUE
Result of original function: 3
Modified bytecode:
  7           0 LOAD_FAST                0 (a)
              2 LOAD_FAST                1 (b)
              4 BINARY_MULTIPLY
              6 LOAD_FAST                0 (a)
              8 BINARY_SUBTRACT
             10 RETURN_VALUE
Result of modified function: 4
```

### frame 对象

python (可能其他语言也类似) 在运行时, 函数的层层调用是一个压栈的过程, 而每一层都有一个上下文环境, 称为栈帧 (frame). Python 语言有着“自省”的特性, 主要是由 `inspect` 模块来提供这种功能, 在栈帧方面, 可以利用 `inspect` 模块的功能获取当前栈帧甚至于更底层的栈帧(即先入栈的那些, 也就是上层函数调用)的信息

`help(type(inspect.currentframe()))` 的输出

```
Help on class frame in module builtins:

class frame(object)
 |  Methods defined here:
 |  
 |  __delattr__(self, name, /)
 |      Implement delattr(self, name).
 |  
 |  __getattribute__(self, name, /)
 |      Return getattr(self, name).
 |  
 |  __repr__(self, /)
 |      Return repr(self).
 |  
 |  __setattr__(self, name, value, /)
 |      Implement setattr(self, name, value).
 |  
 |  __sizeof__(...)
 |      F.__sizeof__() -> size of F in memory, in bytes
 |  
 |  clear(...)
 |      F.clear(): clear most references held by the frame
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  f_back
 |  
 |  f_builtins
 |  
 |  f_code
 |  
 |  f_globals
 |  
 |  f_lasti
 |  
 |  f_lineno
 |  
 |  f_locals
 |  
 |  f_trace
 |  
 |  f_trace_lines
 |  
 |  f_trace_opcodes
```

- `f_back`: 指向调用栈中的上一帧。
- `f_code`: 当前执行帧所对应的代码对象(即 `__code__`)。
- `f_locals`: 当前帧的局部变量字典。
- `f_globals`: 当前帧的全局变量字典。
- `f_lineno`: 当前执行的行号。
- `f_lasti`: 最后执行的指令在字节码中的索引。
- `f_builtins`: 当前帧的内置命名空间

`__code__.co_name` 为 `'module'` 代表其是一个模块, `torch.fx.wrap` 中有涉及到这个检查

```python
import inspect

def f():
    print(dir(inspect.currentframe()))
    print(inspect.currentframe().f_code.co_name)         # 'f'
    # '<module>' 表示是顶级帧
    print(inspect.currentframe().f_back.f_code.co_name)  # '<module>'
    print(inspect.currentframe().f_back.f_back)  # None

f()
```

利用 frame 笔者在阅读源码时用到了一些调试技巧以获取当前函数范围外的变量:

```python
# 调试断点加在 torch.fx._symbolic_trace.py:Tracer.create_args_for_root 函数刚结束时, 笔者想研究 fn 究竟变成了什么
import inspect
caller_frame = inspect.currentframe().f_back
frame = caller_frame
# inspect.stack()[-9].filename
for i in range(13):
    frame=frame.f_back
mymodule = frame.f_globals["mymodule"]
```

另一个小技巧:

```python
x = [1, 2]
def foo():
    pass

foo.__globals__['x'] = 3
print(x)   # 3
```

## 源码阅读 (torch=1.8)

对于上一节的例子而言, 首先需要探索的便是: 

```python
from torch.fx import symbolic_trace
graph_module: torch.fx.GraphModule = symbolic_trace(module)
```

而其实质上等同于

```python
import torch
tracer = torch.fx.Tracer()
graph: torch.fx.Graph = tracer.trace(module)      # 以下主要分析这一行
graph_module = torch.fx.GraphModule(module, graph)
```

对 `Tracer.trace` 的深入研究实际上可以串其大半个 `torch.fx` 模块, 这里首先给出相应的代码目录:

```
torch/fx/
  - _experimental/
  - passes/
  - __init__.py
  - OVERVIEW.md
  - graph_module.py       # 主要是 GraphModule 相关, Tracer.trace 不涉及, 放在后面研究
  - graph.py
  - immutable_collections.py
  - interpreter.py        # Tracer.trace 不涉及, 放在后面研究
  - node.py
  - proxy.py
  - subgraph_rewriter.py  # Tracer.trace 不涉及, 放在后面研究
  - symbolic_trace.py     # Tracer 类, 分析入口
```

### `torch.fx.Tracer` 类概览

首先 `torch.fx.symbolic_trace.Tracer` 继承自 `torch.fx.proxy.TracerBase`, 且在 `torch==1.8.0` 版本中, `TracerBase` 的子类仅有 `Tracer`, 并且对外的接口也都是 `Tracer` 提供的, 在分析 `Tracer.trace` 方法之前, 先看看 `Tracer.__init__` 及 `Tracer` 类的所有方法概览(初看只需粗略看即可, 以后可以回看这里的注释):

```python
# TracerBase 没有定义 __init__ 方法
class Tracer(TracerBase):
    def __init__(self, autowrap_modules : Tuple[ModuleType] = (math, )):
        super().__init__()
        self._autowrap_function_ids: Set[int] = {
            id(value) for name, value in chain(*[m.__dict__.items() for m in autowrap_modules])
            if not name.startswith("_") and callable(value)}
        self._autowrap_search: List[ModuleType] = list(autowrap_modules)
    
    # 重点方法: 注意 TracerBase 也定义了 create_arg 方法, 后面将进一步分析
    def create_arg(self, a: Any) -> "Argument": ...
    
    # 辅助方法: 用于 Tracer.call_module 方法中, 判断一个 m(args) 是否需要直接做成一个 Node, 而不用再深入内部细节, 这里的 m 是 torch.nn.Module 的子类
    # 这里的实现是只要是用户自定义的继承自 torch.nn.Module 的子类, 就继续深入, 否则像 torch.nn.Conv2D, torch.nn.Linear 这类就不深入
    # torch.fx 高阶: 用户可以通过定义 Tracer 的子类重写 is_leaf_module 来改变 trace 的行为, 例如把自定义的 MLP 类也作为叶子节点
    def is_leaf_module(self, mod: torch.nn.Module, module_qualified_name: str):
        return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)

    # 辅助方法, 用于 call_module 中, 帮助判断是否为叶子节点
    def path_of_module(self, mod) -> str: ...
    
    # 重点方法: 用于替换Module.forward方法
    def call_module(self, m, forward, args, kwargs): ...
    
    # 重点方法: 用于构造 forward 函数的输入 Proxy
    def create_args_for_root(self, root_fn, is_module, concrete_args=None): ...

    def trace(self, root, concrete_args) -> Graph: ...

    # ==================== 以下为继承自 TracerBase 的方法 =====================
    # 重点方法: 既用于 create_proxy, 也用于 trace 方法的其他地方
    def create_node(
        self, kind: str, target, args: Tuple[Argument, ...], kwrags: Dict[str, Argument],
        name: str = None, type_expr=None) -> Node: ...

    # Proxy.__init__ 很简单, 仅仅是变量赋值
    def proxy(self, node: Node):
        return Proxy(node, self)
    
    # 重点方法: 内部会调用 create_node
    def create_proxy(
        self, kind: str, target, args: Tuple[Argument, ...], kwrags: Dict[str, Argument],
        name: str = None, type_expr=None) -> Proxy: ...
    
    # 重点方法: 被 Tracer 重载
    def create_arg(self, a: Any) -> Argument: ...

    # 这两个直接报错, 因为无法判断一个 Proxy 对象的值, 也没法迭代(因为它已经是叶子节点了, 内部细节无法得知)
    # 这也就是说 `for i in x` 以及 `if x` 这种代码无法被 torch.fx 所处理
    def to_bool(self, obj: 'Proxy') -> bool: ...
    def iter(self, obj: 'Proxy') -> Iterator: ...

    # 暂不深究
    def keys(self, obj: 'Proxy') -> Any: ...
```

我们主要先关注 `__init__` 方法, 实际上以目前的知识还无法知道在做什么(简版实现里也没有这些操作), 后面回过头来再讲, 因此我们现在应该直接看 `trace` 函数, 但这里先稍稍打乱一下阅读顺序, 先单独看看 `Proxy` 类, 对后面会事半功倍(当然, 如果像笔者一样直接读源码, 可能最开始不太能意识到这个阅读顺序)

### `torch.fx.Proxy`

Proxy 代理的算子

- 魔术方法: 例如 `a + b`, 这里 `a` 和 `b` 原本是 `Tensor` 类型, 经过替换后都是 `Proxy` 类型, 因此在 trace 时由 `Proxy.__add__` 等魔术方法实现, 建立 `call_function` 节点
- `torch.add`: `Proxy.__torch_function__` 实现, 建立 `call_function` 节点 
- `Tensor.add`: `Proxy.__torch_function__` 实现, 建立 `call_method` 节点

再加上

- `Module(...)`: 在 trace 之前, 首先会把 `Module` 的 `__call__` 做全局替换, 因此这类 `out=self.layer_1(out)` 这种代码会进入被替换的函数, 建立 `call_module` 节点
- `Module.attrname`: 在 trace 之前, 首先会把 `Module` 的 `__getattr__` 做全局替换, 因此这类 `self.layer_1` 这种代码会进入被替换的函数, 建立 `get_attr` 节点
- 输入参数的替换: 建立 `placeholder` 节点
- 输出结果在 trace 函数中手动建立 `output` 节点

自定义的追踪函数

```python
# 全局变量
_wrapped_fns_to_patch : List[Tuple[dict, str]]
_wrapped_methods_to_patch : List[Tuple[type, str]]
# 实例变量: autowrap_functions, param_shapes_constant 是 torch==1.8.0 没有的
Tracer.__init__(self, autowrap_modules, autowrap_functions, param_shapes_constant=False)
```

以上便是所有的 node 类型和可追踪的数据操作的全集, 对于不属于上述类型的操作, 则直接进行运算得到结果

tracer 的局限性:

**分支条件直接报错**

```python
if x.sum() > 0:  # 为 x.sum() 以及 {} > 0 建立节点, 然后执行 proxy.__bool__ 时会引发错误(Proxy.__bool__函数报错)
    x = -x
```

**外部函数调用不能正确追踪**

```python
a = torch.tensor(np.random.random(2, 2))  # 这里先对np.random.random直接执行得到结果, 随后直接执行 torch.tensor 得到结果, 这两步都不建立 node
x = x * a  # 检查到 a 不在被 trace 的 model 的属性里, 先执行 setattr(model, "_tensor_constant0", a), 然后为他建立一个 Proxy, 最后执行 x*a 时建立一个node
# z = x * x  # 此时x变量被绑定到了上述返回的node里, 所以这里继续执行 Proxy * Proxy 的操作
```

**动态 shape: 不确定, TODO**

TODO: 根本原因应该也是 Proxy 不能做实际运算: Proxy.shape 仍然是 Proxy 的操作

怎么适当解开上述局限性(Tracer高级特性): TODO


单元测试代码

```python
from torch.fx import Proxy, Tracer, Graph
tracer = Tracer()
tracer.graph = Graph()
proxy = tracer.create_proxy('placeholder', "x", (), {},)

proxy + 1                 # call_function
proxy.add(1)              # call_method
torch.add(proxy, 2)       # call_function
print([node.op for node in list(tracer.graph.nodes)])
```

`Proxy` 类的源码阅读需要注意如下动态的部分 (注意以下是 `torch==1.8.0` 的代码): 简单来说, 就是为 `Proxy` 追加了 `__add__`, `__radd__` 这类魔术方法.

```python
class Proxy:
    ...

reflectable_magic_methods = {
    'add': '{} + {}',
    'sub': '{} - {}',
    'mul': '{} * {}',
    'floordiv': '{} // {}',
    'truediv': '{} / {}',
    'div': '{} / {}',
    'mod': '{} % {}',
    'pow': '{} ** {}',
    'lshift': '{} << {}',
    'rshift': '{} >> {}',
    'and': '{} & {}',
    'or': '{} | {}',
    'xor': '{} ^ {}',
    'getitem': '{}[{}]'
}

magic_methods = dict({
    'eq': '{} == {}',
    'ne': '{} != {}',
    'lt': '{} < {}',
    'gt': '{} > {}',
    'le': '{} <= {}',
    'ge': '{} >= {}',
    'pos': '+{}',
    'neg': '-{}',
    'invert': '~{}'}, **reflectable_magic_methods)

for method in magic_methods:
    def scope(method):
        def impl(*args, **kwargs):
            tracer = args[0].tracer
            target = getattr(operator, method)
            return tracer.create_proxy('call_function', target, args, kwargs)
        impl.__name__ = method
        as_magic = f'__{method}__'
        setattr(Proxy, as_magic, impl)
    scope(method)

def _define_reflectable(orig_method_name):
    method_name = f'__r{orig_method_name}__'

    def impl(self, rhs):
        target = getattr(operator, orig_method_name)
        return self.tracer.create_proxy('call_function', target, (rhs, self), {})
    impl.__name__ = method_name
    impl.__qualname__ = method_name
    setattr(Proxy, method_name, impl)

for orig_method_name in reflectable_magic_methods:
    _define_reflectable(orig_method_name)
```

### `torch.fx.Node`

一个 Node 代表一个 op

- `opcode`: string, 六者之一: `["placeholder", "get_attr", "call_function", "call_module", "call_method", "output"]`. 严格地说其实还有一个特殊类型, 仅在内部使用: `Graph.root` 节点的 `opcode=''`.
- `name`: string, 节点名称, 即 op 的输出, 在同一个 `Graph` 列, `Graph.nodes` 列表里节点名称各不相同.
- `target`: string/callable, 即 op 本身, 对于 `"call_function"` 类型, `target` 是函数本身, 其余类型情况下均为字符串, 这一点的源码验证: 在源码中可以在 `torch.fx.Graph.python_code` 方法中的内部函数 `emit_node` 里各种情况的 `assert` 语句可以得到验证; 或者在 `Tracer.trace` 函数中各处的对 `create_proxy` 调用的实参也可以得到验证
- `args`: tuple[Union[Node, Any]], 位置参数, 元素有可能会是 `Node` 类型或者是“字面量类型” (例如整数等, 代码里对应的是`torch.fx.node.base_types`, 用于 `TracerBase.create_arg` 方法里)
- `kwargs`: Dict[str, Union[Node, Any]], 关键字参数, 元素有可能会是 `Node` 类型或者是“字面量类型”

`Node` 与 `Graph` 的关系如下:

```python
graph: Graph
# graph 实际上包含一个 Node 链表(环状双向链表), 因此 graph 中只持有一个 root 节点 (Node 类型)
# self._root = Node(self, '', 'root', '', (), {})
graph.nodes: _node_list  # @property, _node_list 实质上也是一个类(命名方式没有按驼峰命名)

node_list = list(graph.nodes)

node = node_list[0]

# node_list 实质上一个环状双向链表:
# [x: Node, y: Node, output: Node, root: Node]
# x._next = y,         x._prev = root
# y._next = output,    y._prev = x
# output._next = root, output._prev = y
# root._next = x,      root._prev = output
```

`Node` 还包含几个属性:

```python
node._input_nodes   # Dict[Node, None]: _args, _kwargs 代表了计算得到此节点需要的输入, 而 _input_nodes 是 args 和 kwargs 中类型是 Node 的元素
node._args          # tuple, 这里面也会包含 _input_nodes 中的节点
node._kwargs        # dict

node.users          # Dict[Node, None]: 代表此节点被用在后续哪些节点里
node._prev
node._next
node._erased: bool  # iter(graph.nodes) 的返回结果不会保留 _erased 为 True 的节点
```

`Node` 有一个特殊的内部方法 `__update_args_kwargs`, 在初始化时会被调用以维护好 `_input_nodes`, `users` 属性

```python
class Node:
    # 此函数仅在 self.args, self.kwargs 被赋值, 或者是 Node.__init__ 方法时才会被调用
    # 实质上是在维护整个 graph 中每个节点的 _input_nodes, users 属性
    def __update_args_kwargs(self, new_args : Tuple['Argument', ...], new_kwargs : Dict[str, 'Argument']):
        self._args = new_args
        self._kwargs = new_kwargs

        for old_use in self._input_nodes.keys():
            old_use.users.pop(self)

        self._input_nodes = {}
        # map_arg 的作用是: 递归检查 self._args 的每个元素, 如果是 Node 类型, 则执行函数
        map_arg(self._args, lambda n: self._input_nodes.setdefault(n))
        map_arg(self._kwargs, lambda n: self._input_nodes.setdefault(n))

        for new_use in self._input_nodes.keys():
            new_use.users.setdefault(self)
```

### `Tracer.trace`


调用栈(TODO, 替换为源码)

```
trace:

collect_tensor_attrs(...)  # 不知道用在哪, 注释说用在 create_arg
fn_globals = fn.__globals__
self.create_args_for_root
    TracerBase.create_proxy('placeholder', name, default, {})
        Tracer.create_arg -> super().create_arg
        TracerBase.create_node: 完全是 self.graph.create_node
        TracerBase.proxy: Proxy(node, self)
    也许使用底层API创建函数修改被追踪的函数签名(仅适用于有可变参数的情形)
with _Patcher() as patcher:
    patcher.patch_method(Module.__getattr__)
    patcher.patch_method(Module.__call__)
    # 对全局变量 _wrapped_fns_to_patch, _wrapped_methods_to_patch分别进行patch和patch_method, 这里buildin的用法待研究
    _patch_wrapped_functions(patcher)

    
    
    # 
    _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)

    # self._autowrap_search 实际上就是 Tracer 的默认实例化参数 autowrap_modules , 一般就只是包含 math
    for module in self._autowrap_search:
        _autowrap_check(patcher, module.__dict__, self._autowrap_function_ids)

    self.create_node('output', 'output', (self.create_arg(fn(*args)),), {},
                        type_expr=fn.__annotations__.get('return', None))
```


```python
# Node 类型则应用函数, 否则不应用
def map_arg(a: Argument, fn: Callable[[Node], Argument]) -> Argument:
    return map_aggregate(a, lambda x: fn(x) if isinstance(x, Node) else x)


# map_arg 函数使用了 map_aggregate 函数, 其进一步使用到了一些高级技巧: 使用 type 构建不可变列表及不可变字典, 实现如下:
def _no_mutation(self, *args, **kwargs):
    raise NotImplementedError(f"'{type(self).__name__}' object does not support mutation. {_help_mutation}")

def _create_immutable_container(base, mutable_functions):
    container = type('immutable_' + base.__name__, (base,), {})
    for attr in mutable_functions:
        setattr(container, attr, _no_mutation)
    return container

immutable_list = _create_immutable_container(list,
    ['__delitem__', '__iadd__', '__imul__', '__setitem__', 'append', 'clear', 'extend', 'insert', 'pop', 'remove'])
# __reduce__ 与 pickle 模块相关, 相关的魔术方法还有 __reduce_ex__, __getstate__, __setstate__
immutable_list.__reduce__ = lambda self: (immutable_list, (tuple(iter(self)),))
immutable_dict = _create_immutable_container(dict, ['__delitem__', '__setitem__', 'clear', 'pop', 'popitem', 'update'])
immutable_dict.__reduce__ = lambda self: (immutable_dict, (iter(self.items()),))

```

`collect_tensor_attrs`: 这个有些奇特, 似乎只捕获到"悬挂"的tensor或者子模型(位于 `_modules` 中)的"悬挂"tensor, 不确定在后续的作用

```python
class N(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.relu = torch.nn.ReLU()
        self.a = torch.rand(2)
        # N().__dict__["b"]: KeyError       # N().b: OK
        self.b = torch.nn.Parameter(torch.rand(1))  # 不会被捕获, 因为它会被自动放入 _parameters 中, 而不会是 __dict__["b"]
    def forward(self, x):
        pass

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # torch.nn.ModuleList 最终会落到 Module.add_module, 其本质是对 self._modules 进行添加
        self.blocks = torch.nn.ModuleList([N() for i in range(3)])  # 这里面的 N().a 会被捕获
        self.blocks_list = [N() for i in range(2)]                  # 这里面的 N().a 不会被捕获
        self.layer = N()
    def forward(self, x):
        pass

from typing import List
from torch._C import ScriptObject

tensor_attrs = {}
def collect_tensor_attrs(m : torch.nn.Module, prefix_atoms : List[str]):
    for k, v in m.__dict__.items():
        if isinstance(v, (torch.Tensor, ScriptObject)):
            tensor_attrs[v] = '.'.join(prefix_atoms + [k])
    for k, v in m.named_children():
        collect_tensor_attrs(v, prefix_atoms + [k])

collect_tensor_attrs(M(), [])
print(tensor_attrs)
"""
{tensor([0.9359, 0.2638]): 'blocks.0.a',
 tensor([0.1203, 0.3830]): 'blocks.1.a',
 tensor([0.1230, 0.9907]): 'blocks.2.a',
 tensor([0.7182, 0.1366]): 'layer.a'}
"""
print(N().__dict__)
"""
{'training': True,
 '_parameters': OrderedDict([('b', Parameter containing:
               tensor([0.0683], requires_grad=True))]),
 '_buffers': OrderedDict(),
 '_non_persistent_buffers_set': set(),
 '_backward_hooks': OrderedDict(),
 '_is_full_backward_hook': None,
 '_forward_hooks': OrderedDict(),
 '_forward_pre_hooks': OrderedDict(),
 '_state_dict_hooks': OrderedDict(),
 '_load_state_dict_pre_hooks': OrderedDict(),
 '_load_state_dict_post_hooks': OrderedDict(),
 '_modules': OrderedDict([('linear',
               Linear(in_features=3, out_features=3, bias=True)),
              ('relu', ReLU())]),
 'a': tensor([0.6327, 0.7129])}
"""
```


--------------------------------------
## 源码阅读 (torch=2.0)

`Node`:

```python
node.op: str  # "placeholder", "output", "call_function", "call_module", "call_method" 
node.name: str
node.target: Union[str, torch.Function, Any]
node.args: List[Node]
node.kwargs: Dict[str, Node]
```

图转换手段

```python
# Note that this decomposition rule can be read as regular Python
import torch.nn.functional as F
def relu_decomposition(x):
    return (x > 0) * x

decomposition_rules = {}
decomposition_rules[F.relu] = relu_decomposition

def decompose(model: torch.nn.Module,
              tracer_class : type = fx.Tracer) -> torch.nn.Module:
    """
    Decompose `model` into smaller constituent operations.
    Currently,this only supports decomposing ReLU into its
    mathematical definition: (x > 0) * x
    """
    graph : fx.Graph = tracer_class().trace(model)
    new_graph = fx.Graph()
    env = {}
    tracer = torch.fx.proxy.GraphAppendingTracer(new_graph)
    for node in graph.nodes:
        if node.op == 'call_function' and node.target in decomposition_rules:
            # By wrapping the arguments with proxies,
            # we can dispatch to the appropriate
            # decomposition rule and implicitly add it
            # to the Graph by symbolically tracing it.
            proxy_args = [
                fx.Proxy(env[x.name], tracer) if isinstance(x, fx.Node) else x for x in node.args]
            output_proxy = decomposition_rules[node.target](*proxy_args)

            # Operations on `Proxy` always yield new `Proxy`s, and the
            # return value of our decomposition rule is no exception.
            # We need to extract the underlying `Node` from the `Proxy`
            # to use it in subsequent iterations of this transform.
            new_node = output_proxy.node
            env[node.name] = new_node
        else:
            # Default case: we don't have a decomposition rule for this
            # node, so just copy the node over into the new graph.
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
    return fx.GraphModule(model, new_graph)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 10)
    def forward(self, x):
        return torch.nn.functional.relu(self.linear(x))

print(decompose(M()).code)
```

源码分析需要关注这些:

- Proxy
- Node
- Graph
- GraphModule
- Tracer
- Interpreter, Transformer


`torch.fx._symbolic_proxy._patch_function`: 以下是简化后的核心实现 (`Python >= 3.8`)

```python
import inspect
def _f(): pass
FunctionType = type(_f)   # 即: types.FunctionType
CodeType = type(_f.__code__)  # 即: types.CodeType
HAS_VARSTUFF = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS

def _patch_function(fn: FunctionType, nargs: int) -> FunctionType:
    co = fn.__code__
    co_flags = co.co_flags & ~HAS_VARSTUFF
    co_args: tuple
    co_args = (
        nargs,
        0,
        0,
        co.co_nlocals,
        co.co_stacksize,
        co_flags,
        co.co_code,
        co.co_consts,
        co.co_names,
        co.co_varnames,
        co.co_filename,
        co.co_name,
        co.co_firstlineno,
        co.co_lnotab,
        co.co_freevars,
        co.co_cellvars,
    )
    new_code = CodeType(*co_args)  # type: ignore[arg-type]
    return FunctionType(
        new_code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
    )
    # we need to insert placeholder nodes for *args and **kwargs
    # we can't call this function normally, otherwise it would try to unpack them
    # instead, let's make python think that args and kwargs are normal variables
```

怎么理解最后的三行注释: 注意这里的实参即为被 trace 的函数, 即 `nn.Module.forward`, 有时候这个函数会包含一些可变参数: `args` 与 `kwargs`. 这种情况下, 原本的 `forward` 函数签名假设是 `forward(self, x, *args, **kwargs)`, 那么原本的 `forward` 函数的 `__code__.co_argcount=1`, 这种情况下, 在 `trace` 函数里, 会进行这种调用:

```python
root_fn = forward
# root_fn(x, y, z, d=1, e=2)  # OK
root_fn = _patch_function(root_fn, 3)
# 必须得用下面这种方式调用, 不能用原本的方式调用
# root_fn(x, (y, z), {"d": 1, "e": 2})
```


**`Tracer().trace()` 的总体逻辑**


调试代码

```python
import torch
from typing import Dict, List
from torch.fx import symbolic_trace

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)  # weight: (out_channels, in_channels)
    
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        *args: List[Dict[str, torch.Tensor]],
        **kwargs: Dict[str, torch.Tensor]
    ):
        out = x["a"].clone()
        out -= x["b"]
        out += y
        out += args[0]["u"]
        out += args[0]["v"]
        out += kwargs["c"]

        out = ((out+self.param)@self.linear.weight.T)
        out = out.clamp(0.0, max=1.0)
        return out

mymodule = MyModule()
my_x = {
    "a": torch.rand(3, 4),
    "b": torch.rand(3, 4)
}
my_y = torch.rand(3, 4)
my_args = [{"u": torch.rand(3, 4)}, {"v": torch.rand(3, 4)}]
my_kwargs = {"c": torch.rand(3, 4)}
print(mymodule(my_x, my_y, *my_args, **my_kwargs))

symbolic_traced: torch.fx.GraphModule = symbolic_trace(mymodule, concrete_args={"x": my_x})
```



首先分析一些辅助函数:

```python
# ==================== Tracer.create_proxy ===================================
class Graph:
    def create_node(self, op: str, target: 'Target',
                    args: Optional[Tuple['Argument', ...]] = None,
                    kwargs: Optional[Dict[str, 'Argument']] = None,
                    name: Optional[str] = None,
                    type_expr: Optional[Any] = None) -> Node:
        assert op in ('call_function', 'call_method', 'get_attr', 'call_module', 'placeholder', 'output')
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        candidate = name if name is not None else self._target_to_str(target)
        name = self._graph_namespace.create_name(candidate, None)
        n = Node(self, name, op, target, args, kwargs, type_expr)
        self._graph_namespace.associate_name_with_obj(name, n)
        self._insert(n)
        self._len += 1
        return n

class TracerBase:
    def create_node(self, kind : str, target : Target,
                    args : Tuple[Argument, ...], kwargs : Dict[str, Argument], name : Optional[str] = None,
                    type_expr : Optional[Any] = None) -> Node:
        if kind == 'call_function' and self.check_mutable_operations:  # 可忽略?
            check_for_mutable_operation(target, args, kwargs)
        node = self.graph.create_node(kind, target, args, kwargs, name, type_expr)
        if self.module_stack:
            node.meta['nn_module_stack'] = copy.copy(self.module_stack)
        return node

    # 简化版源码
    def create_proxy(self, kind: str, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any],
                     name: Optional[str] = None, type_expr : Optional[Any] = None):
        args_ = self.create_arg(args)
        kwargs_ = self.create_arg(kwargs)
        assert isinstance(args_, tuple)
        assert isinstance(kwargs_, dict)
        node = self.create_node(kind, target, args_, kwargs_, name, type_expr)
        proxy = self.proxy(node)  # Proxy(node, self)
        return proxy
        
# =================== _create_wrapped_func ============================
# 如果原始函数orig_fn的实参不包含Proxy类型,则原封不动,否则创建node代替函数调用并将node添加至graph中,并返回Proxy
def _create_wrapped_func(orig_fn):
    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        # 如果args和kwargs中有任何一个叶子元素是Proxy类型,那么proxy是最后一个是Proxy类型的叶子节点,否则proxy是None
        proxy = _find_proxy(args, kwargs)
        if proxy is not None:
            return_proxy = proxy.tracer.create_proxy("call_function", orig_fn, args, kwargs)
            return_proxy.node.meta["is_wrapped"] = True
            return return_proxy
        return orig_fn(*args, **kwargs)
    return wrapped

# =============================== _Patcher ================================
# patch 和 patch_method 都是用作"属性"替换, 并同时在 self.patches_made 中做记录,
# 在 with _Patcher() as patcher 上下文退出时按 self.patches_made 倒序替换回原始属性
# patch 和 patch_method 的区别仅在于 patch 是用于字典的替换, patch_method 用于类的属性替换
# 完整源码如下:
class _Patcher:
    def __init__(self):
        super().__init__()
        self.patches_made: List[_PatchedFn] = []
        self.visited: Set[int] = set()

    def patch(self, frame_dict: Dict[str, Any], name: str, new_fn: Callable, deduplicate: bool = True):
        new_fn.__fx_already_patched = deduplicate  # type: ignore[attr-defined]
        if name not in frame_dict and hasattr(builtins, name):
            self.patches_made.append(_PatchedFnDel(frame_dict, name, None))
        elif getattr(frame_dict[name], "__fx_already_patched", False):
            return
        else:
            self.patches_made.append(_PatchedFnSetItem(frame_dict, name, frame_dict[name]))
        frame_dict[name] = new_fn

    def patch_method(self, cls: type, name: str, new_fn: Callable, deduplicate: bool = True):
        new_fn.__fx_already_patched = deduplicate  # type: ignore[attr-defined]
        orig_fn = getattr(cls, name)
        if getattr(orig_fn, "__fx_already_patched", False):
            return
        self.patches_made.append(_PatchedFnSetAttr(cls, name, orig_fn))
        setattr(cls, name, new_fn)

    # visit_once 不关键, 仅仅是一个"备忘录"
    def visit_once(self, thing: Any):
        idx = id(thing)
        if idx in self.visited:
            return False
        self.visited.add(idx)
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        while self.patches_made:
            self.patches_made.pop().revert()
        self.visited.clear()

# 将 frame_dict 中的所有 callable 都替换掉
def _autowrap_check(patcher: _Patcher, frame_dict: Dict[str, Any], function_ids: Set[int]):
    if patcher.visit_once(frame_dict):
        for name, value in frame_dict.items():
            if (not name.startswith("_") and callable(value) and id(value) in function_ids):
                patcher.patch(frame_dict, name, _create_wrapped_func(value))

# 将 fn 所在模块(即.py文件)的所有顶级符号(函数/类名/变量名) 添加到全局 _wrapped_fns_to_patch 列表里(仅作记录)
# fn 必须是模块的顶级函数
def wrap(fn):
    ...

# 这个函数会被导入模块时被调用, 因此会带来将所有 _symbolic_trace.py 下定义的顶级符号添加到全局 _wrapped_fns_to_patch 列表里(仅做记录)
@wrap
def _assert_is_none(value, msg):
    assert value is None, msg
```

然后先整体看一下 `Tracer` 类

```python
class Tracer:
    def __init__(self,
        autowrap_modules: Tuple[ModuleType] = (math,),
        autowrap_functions: Tuple[Callable, ...] = (),
        param_shapes_constant: bool = False,
    ) -> None
        ...
        # 这两个属性在 trace 方法中会被使用到, 这里可以暂时不管
        # autowrap_modules (default): (math,)
        # autowrap_functions (default): ()
        self._autowrap_function_ids: Set[int] = {
            id(value)
            for name, value in chain(*[m.__dict__.items() for m in autowrap_modules])
            if not name.startswith("_") and callable(value)
        }
        self._autowrap_function_ids.update({id(f) for f in autowrap_functions})
        ...

    # 重点方法
    def create_arg(self, a: Any) -> "Argument":
        ...
    
    # 辅助方法
    def is_leaf_module(self, mod: torch.nn.Module, module_qualified_name: str):
        ...

    # 辅助方法
    def path_of_module(self, mod) -> str:
        ...
    
    # 重点方法: 用于替换Module.forward方法
    def call_module(self, m, forward, args, kwargs):
        ...
    
    # 重点方法: 用于替换Module.__getattr__方法
    def getattr(self, attr, attr_val, parameter_proxy_cache):
        ...
    
    # 重点方法: 用于构造 forward 函数的输入 Proxy
    def create_args_for_root(self, root_fn, is_module, concrete_args=None):
        ...
    
    # 伪代码: TODO
    def trace(self, root, concrete_args) -> Graph:
        ...

    def __deepcopy__(self, memo):
        ...
```

然后分析 trace 函数

**`concrete_args` 有什么用?**

```python
# 假设原本的 fn 的函数签名是 forward(self, x: Dict[str, Tensor], y: Tensor, *args: List[Dict], **kwargs),
# 调用方式如下:
forward(
    {"a": torch.rand(2, 3), "b": torch.rand(3, 4)}         # x
    torch.rand(4, 5),                                      # y
    *(torch.rand(2, 3),),                                  # args
    **{"c": torch.rand(4, 5)},                             # kwargs
    )
concrete_args: {"x": {"a": torch.rand(2, 3), "b": torch.rand(3, 4)}}
```

**torch==1.8.0**

不指定 `concrete_args` 时

```python
# fn: 函数签名为 fn(self, x, y, args, kwargs), 即: fn.__code__.co_argcount=5
fn(
    module,
    {"a": torch.rand(2, 3), "b": torch.rand(3, 4)}  # x
    y,                                              # y
    args,                                           # args 是一个列表, 不能传 *args
    kwargs,                                         # kwargs 是一个字典, 不能传 **kwargs
)
args = [module, Proxy(x), Proxy(y), Proxy(_args), Proxy(_kwargs)]
fn(*args)
```

指定 `concrete_args` 时

```python
# fn 同上
args = [module, concrete_args['x'], Proxy(y), Proxy(_args), Proxy(_kwargs)]
```

**torch==2.0.0**

不指定 `concrete_args` 与 `torch==1.8.0` 时一致, 指定 `concrete_args` 时

```python
# fn: 函数签名变为 forward(*args), 即: fn.__code__.co_argcount=0
# 如果需要调用 fn, 必须这样传实参(一共6个)
fn(
    module,
    torch.rand(2, 3),   # 对应于 concrete_args["x"]["a"]
    torch.rand(3, 4),   # 对应于 concrete_args["x"]["b"]
    y,
    args,               # args 是一个列表, 不能传 *args
    kwargs,             # kwargs 是一个字典, 不能传 **kwargs
)
args = [module, concrete_args["x"]["a"], concrete_args["x"]["b"], Proxy(y), Proxy(_args), Proxy(_kwargs)]
fn(*args)
```


```python
# Tracer.trace 方法主体部分
def trace(self, root, concrete_args):
    # step 0: root 如果是 module, fn = module.forward

    # step 1: 此步骤说明见上
    # 备注 (torch==2.0.0): create_args_for_root 源码中的 flatten_fn 的分支仅在指定 concrete_args 时才有可能被触发
    fn, args = self.create_args_for_root(fn, isinstance(root, torch.nn.Module), concrete_args)

    with _Patcher() as patcher:
        # patcher 带有一个属性 patches_made, 它会记录一个列表, 列表中的每一项是一个元组: (cls/frame_dict, name, ori_value)
        # 每个代表被替换的 new_value (ori_value 和 new_value 一般都是函数/方法) 的原始信息, 以便在 patcher.__exit__() 时进行恢复
        
        # 原地替换 Module.__getattr__ 为 Tracer.getattr (这个新方法的行为是先调用Module.__getattr__,然后根据返回结果类型创建Proxy返回)
        patcher.patch_method(torch.nn.Module, "__getattr__", module_getattr_wrapper, deduplicate=False)

        @functools.wraps(_orig_module_call)
        def module_call_wrapper(mod, *args, **kwargs):
            def forward(*args, **kwargs):
                return _orig_module_call(mod, *args, **kwargs)
            _autowrap_check(
                patcher,
                getattr(getattr(mod, "forward", mod), "__globals__", {}),
                self._autowrap_function_ids,
            )
            return self.call_module(mod, forward, args, kwargs)

        # module_call_wrapper 的行为是: 
        patcher.patch_method(torch.nn.Module, "__call__", module_call_wrapper, deduplicate=False)
        _patch_wrapped_functions(patcher)
        _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
        for module in self._autowrap_search:
            _autowrap_check(
                patcher, module.__dict__, self._autowrap_function_ids
            )
        self.create_node(
            "output",
            "output",
            (self.create_arg(fn(*args)),),
            {},
            type_expr=fn.__annotations__.get("return", None),
        )

    self.submodule_paths = None
    return self.graph
```

---
layout: post
title: "(WIP) torch.fx"
date: 2023-11-28 11:10:04 +0800
labels: [pytorch]
---


参考资料: [https://pytorch.org/docs/stable/fx.html](https://pytorch.org/docs/stable/fx.html)

知乎源码分析: [博客](https://zhuanlan.zhihu.com/p/625690498)

torch.fx 应用例子:

- [pytorch Conv+BN fuse](https://github.com/pytorch/pytorch/blob/main/torch/fx/experimental/optimization.py#L50)
- [fx graph mode quantization](https://pytorch.org/docs/master/quantization.html#prototype-fx-graph-mode-quantization)
- [huggingface optimum](https://huggingface.co/docs/optimum/main/en/torch_fx/usage_guides/optimization): 但似乎没有具体的用例


`torch.fx` 模块的主要用途是使用一种方式(`torch.fx.Tracer`, 不同于 `torchscript`)分析源代码, 得到中间表示 `torch.fx.Graph`, 然后程序员可以修改这个中间表示, 最后重新转换回转换后的 python 代码. 也就是说可以完成这个 pipeline: `symbolic tracing -> intermediate representation -> transforms -> Python code generation`, 作用如下:

- 假设现在有很多包含 `nn.Module` 的代码, 现在希望对这些代码做统一的转换, 则可以走上面的完整 pipeline
- 可以直接写一个中间表示的配置文件, 然后直接从配置文件转换为 python 代码

总的来说, 就是可以用比较 hack 的方式修改模型, 下面是一个例子:

这里对于一个已经写好的 `M`, 我希望将 `torch.add` 操作全部替换为 `torch.mul`, 但是又不希望修改 `M` 的源代码

```python
import torch
import torch.fx

# Sample module
class M(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)

def transform(m: torch.nn.Module,
              tracer_class : type = fx.Tracer) -> torch.nn.Module:
    graph : fx.Graph = tracer_class().trace(m)
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    for node in graph.nodes:
        # Checks if we're calling a function (i.e:
        # torch.add)
        if node.op == 'call_function':
            # The target attribute is the function
            # that call_function calls.
            if node.target == torch.add:
                node.target = torch.mul

    graph.lint() # Does some checks to make sure the
                 # Graph is well-formed.

    return fx.GraphModule(m, graph)

# GraphModule 也是 nn.Module
m: torch.fx.GraphModule = transform(M())
```

`symbolic_trace(module)` 实质上基本等同于 `torch.fx.GraphModule(module, torch.fx.Tracer().trace(module))`


`Node`:

```python
node.op: str  # "placeholder", "output", "call_function", "call_module", "call_method" 
node.name: str
node.target: Union[str, torch.Function, Any]
node.args: List[Node]
node.kwargs: Dict[str, Node]
```

`Proxy`: `F.relu` 这种函数可以作用在 `Proxy` 上


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

`torch.fx` 源码里用到的主要工具:

- 利用 pytorch 的 `__torch_function__` 的功能, 运行 `forward` 方法本应该送入 `torch.Tensor` 进行运行, 但在 `torch.fx` 的实现里, 可以用 `Proxy` 来作为替代
- `inspect`, `fn.__code__` 的综合应用: 还包括手动使用 `type(fn.__code__)(...)` (`types.CodeType`) 的方式手动构建代码对象, 以及利用 `type(fn)(fn.__code__, ...)` (`types.FunctionType`) 的方式构建函数 (我们通常是使用 `def` 语句来完成这两个过程)

```python
import types
print(types.CodeType.__doc__)
print(types.FunctionType.__doc__)
```

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


Node: 一个 Node 代表一个 op

- `opcode`: string, 六者之一: `["placeholder", "get_attr", "call_function", "call_module", "call_method", "output"]`
- `name`: string, 节点名称, 即 op 的输出, 在同一个 `Graph` 列, `Graph.nodes` 列表里节点名称各不相同.
- `target`: string/callable, 即 op 本身, 对于 `"call_function"` 类型, `target` 是函数本身, 其余情况下均可用字符串代替
- `args`: tuple[Uinon[Node, Any]], 位置参数, 元素有可能会是 `Node` 类型
- `kwargs`: Dict[str, Union[Node, Any]], 关键字参数, 元素有可能会是 `Node` 类型


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

参考如下资料理解:

- [https://stackoverflow.com/questions/16064409/how-to-create-a-code-object-in-python](https://stackoverflow.com/questions/16064409/how-to-create-a-code-object-in-python)
- [https://github.com/python/cpython/blob/3.8/Include/code.h](https://github.com/python/cpython/blob/3.8/Include/code.h)


系统化地研究 Python AST: 主要涉及的内容包括 `ast`, `dis` 模块

- python 官方文档: Language Reference?
- [ast 模块官方文档](https://docs.python.org/3/library/dis.html)
- [ast 详解文章](https://greentreesnakes.readthedocs.io/en/latest/index.html)

其他:
- [博客](https://towardsdatascience.com/understanding-python-bytecode-e7edaae8734d)
- [博客](https://medium.com/@noransaber685/demystifying-python-bytecode-a-guide-to-understanding-and-analyzing-code-execution-6a163cb83bd1)

**浅尝内置函数 `compile`, `dis.dis`, `fn.__code__` 的基本用法**

```python
import dis
code_string = """
def countdown(n):
    while n > 0:
        print('T-minus', n)
        n -= 1
    print('Blastoff!')
"""
code = compile(code_string, "test", "exec")
dis.dis(code)
```

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
```

```
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

**co_code 与 dis.Bytecode 的关系**

从下面的例子可以看出对应关系如下: **`co_code` 每两个字节代表一条指令, 其中第一个字节代表 opname, 第二个字节代表操作数 arg**, 全部的指令集为: `dis.opname`

```python
import dis
print(list(enumerate(dis.opname)))
[(0, '<0>'), (1, 'POP_TOP'), (2, 'ROT_TWO'), (3, 'ROT_THREE'), (4, 'DUP_TOP'), (5, 'DUP_TOP_TWO'), (6, 'ROT_FOUR'), (7, '<7>'), ...]
```

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
```

```
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



**frame 对象**

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

`f_back`: 指向调用栈中的上一帧。
`f_code`: 当前执行帧所对应的代码对象。
`f_locals`: 当前帧的局部变量字典。
`f_globals`: 当前帧的全局变量字典。
`f_lineno`: 当前执行的行号。
`f_lasti`: 最后执行的指令在字节码中的索引。
`f_builtins`: 当前帧的内置命名空间

```python
import inspect

def f():
    print(dir(inspect.currentframe()))
    print(inspect.currentframe().f_code.co_name)         # 'f'
    # '<module>' 就是顶级帧了
    print(inspect.currentframe().f_back.f_code.co_name)  # '<module>'
    print(inspect.currentframe().f_back.f_back)  # None

f()
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
        out = x["a"]
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

一些调试技巧(获取当前函数范围外的变量):

```python
# 加在 torch.fx._symbolic_trace.py:Tracer.create_args_for_root 函数内
import inspect
caller_frame = inspect.currentframe().f_back
frame = caller_frame
# inspect.stack()[-9].filename
for i in range(13):
    frame=frame.f_back
mymodule = frame.f_globals["mymodule"]
```

源码分析

分析目标是: 

```python
symbolic_trace(module: torch.nn.Module)
```

也就是:

```python
tracer = torch.fx.Tracer()
graph: Graph = tracer.trace(module: torch.nn.Module)
gm = torch.fx.GraphModule(module, graph)
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


```python
# Tracer.trace 方法
def trace(self, root, concrete_args):
    # step 0: root 如果是 module, fn = module.forward

    # step 1:
    # 输入:
    #   fn: 假设原本的 fn 的函数签名是 forward(self, x: Dict[str, Tensor], y: Tensor, *args: List[Dict], **kwargs),
    #   即实参是 forward(
    #           {"a": torch.rand(2, 3), "b": torch.rand(3, 4)}  # x
    #           torch.rand(4, 5),                               # y
    #           torch.rand(2, 3),                               # args
    #           c = torch.rand(4, 5)                            # kwargs
    #           )
    #  concrete_args: {"x": {"a": torch.rand(2, 3), "b": torch.rand(3, 4)}}
    # 输出:
    #   fn: 函数签名变为 forward(*args), 但如果需要调用 fn, 必须这样传实参(一共5个)
    #       fn(
    #           module,
    #           torch.rand(2, 3),   # 对应于 concrete_args["x"]["a"]
    #           torch.rand(3, 4),   # 对应于 concrete_args["x"]["b"]
    #           y,
    #           args,               # args 是一个列表, 不能传 *args
    #           kwargs,             # kwargs 是一个字典, 不能传 **kwargs
    #       )
    #   args: [module, concrete_args["x"]["a"], concrete_args["x"]["b"], Proxy(y), Proxy(_args), Proxy(_kwargs)]
    #   备注: 在不指定 concrete_args 的情况下, fn 的函数签名为 fn(self, x, y, args, kwargs), args 为 [module, Proxy(x), Proxy(y), Proxy(_args), Proxy(_kwargs)]
    #   备注: create_args_for_root 源码中的 flatten_fn 的分支仅在指定 concrete_args 时才有可能被触发
    fn, args = self.create_args_for_root(fn, isinstance(root, torch.nn.Module), concrete_args)   # 【注意点1】

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

    # TODO: 后面的内容需要继续研究
```

未解之谜: 按上面的理解, 在 `【注意点1】` 处调用 `fn` (打断点, 通过 `inspect` 模块调用上层 frame) 似乎得不到正确的结果, 且重复调用结果也不一致.


小技巧:

```python
x = [1, 2]
def foo():
    pass

foo.__globals__['x'] = 3
print(x)   # 3
```
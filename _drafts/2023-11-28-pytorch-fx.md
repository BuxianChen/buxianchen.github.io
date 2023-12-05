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


Node: 一个 Node 代表一个 op

- `opcode`: string, 六者之一: `["placeholder", "get_attr", "call_function", "call_module", "call_method", "output"]`
- `name`: string, 节点名称, 即 op 的输出, 在同一个 `Graph` 列, `Graph.nodes` 列表里节点名称各不相同.
- `target`: string/callable, 即 op 本身, 对于 `"call_function"` 类型, `target` 是函数本身, 其余情况下均可用字符串代替
- `args`: tuple[Uinon[Node, Any]], 位置参数, 元素有可能会是 `Node` 类型
- `kwargs`: Dict[str, Union[Node, Any]], 关键字参数, 元素有可能会是 `Node` 类型


`torch.fx._symbolic_proxy._patch_function`: 以下是简化后的核心实现

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

- [https://stackoverflow.com/questions/16064409/how-to-create-a-code-object-in-python](https://stackoverflow.com/questions/16064409/how-to-create-a-code-object-in-python)
- [https://github.com/python/cpython/blob/3.8/Include/code.h](https://github.com/python/cpython/blob/3.8/Include/code.h)


系统化地研究 python AST

- python 官方文档: Language Reference?
- [ast 模块官方文档](https://docs.python.org/3/library/dis.html)
- [ast 详解文章](https://greentreesnakes.readthedocs.io/en/latest/index.html)

其他:
- [博客](https://towardsdatascience.com/understanding-python-bytecode-e7edaae8734d)
- [博客](https://medium.com/@noransaber685/demystifying-python-bytecode-a-guide-to-understanding-and-analyzing-code-execution-6a163cb83bd1)

例子

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
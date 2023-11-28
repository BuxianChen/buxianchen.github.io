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
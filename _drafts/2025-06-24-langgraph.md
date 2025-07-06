---
layout: post
title: "(P1) langgraph"
date: 2025-06-24 09:05:04 +0800
labels: [langchain,langgraph]
---

## 动机、参考资料、涉及内容

- langgraph 源码阅读 (针对 [0.5.0](https://github.com/langchain-ai/langgraph/tree/0.5.0) 版本代码)
- langgraph 使用, 并以此了解一些 fancy 的 Agent 实现

## 用例

### 例1 (tool use agent): openrouter

```python
import json
import os
from typing import Annotated

# from langchain.chat_models import init_chat_model
from langchain_openai.chat_models import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def get_weather(location: str, date: str) -> str:
    """获取某地,某日期的天气"""
    return f"{location} {date} 的天气是晴"


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


def route_tools(
    state: State,
):
    messages = state.get("messages", [])
    if messages:
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

llm = ChatOpenAI(
    model_name="openai/gpt-4.1-nano",
    # model_name="anthropic/claude-sonnet-4",
    api_key=os.environ["OR-API-KEY"],
    base_url="https://openrouter.ai/api/v1"
)
tools = [get_weather]
llm = llm.bind_tools(tools)
tool_node = BasicToolNode(tools)

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()

state = graph.invoke(
    {
        "messages": [
            HumanMessage(content="北京和天津2025年7月6号的天气相同吗")
        ]
    }
)

print(state)

# res = llm.invoke(
#     [{"role": "user", "content": "北京和天津2025年7月6号的天气相同吗"}]
# )
# print(res)
```

说明: openai 和 claude 模型均支持 multi-tool, 也就是返回的 `tool_calls` 是一个列表, 里面可以包含多个工具调用

## 疑惑

Pregel 的 `__init__` 函数的参数里有如下类型注解:

```python
def __init__(
    self,
    *,
    nodes: dict[str, PregelNode | NodeBuilder],
    channels: dict[str, BaseChannel | ManagedValueSpec] | None,
    ...
):
    ...
```

这里的 ManagedValueSpec 有什么作用, 为什么和 BaseChannel 并列. 同样的疑惑在看 StateGraph.add_node 调用到的 `langgraph.graph.state._get_channel` 方法也有类似的并列关系

ManagedValueSpec 表示的是继承自 ManagedValue 的类, 也就是 `langgraph.managed.IsLastStepManager` 和 `langgraph.managed.RemainingStepsManager` 这两个.

## 待整理源码阅读

### langgraph.graph.state.StateGraph

StateGraph 的主要方法有:

- `__init__`
- `_add_shema`
- `add_node`
- `add_edge`
- `add_conditional_edges`
- `set_entry_point`: 其实就是 `self.add_edge(START, key)`
- `set_conditional_entry_point`: 其实就是 `self.add_conditional_edges(START, path, path_map)`
- `set_finish_point`: 其实就是 `self.add_edge(key, END)`
- `validate`: 在 `self.compile` 中被调用
- `compile`: 转换为 `CompiledStateGraph` 类


```python
# StateLike 基本上就是 Union[typing.TypedDict, dataclasses.dataclass, pydantic.BaseModel]
from langgraph._typing import StateLike
StateT = TypeVar("StateT", bound=StateLike)
InputT = TypeVar("InputT", bound=StateLike, default=StateT)
OutputT = TypeVar("OutputT", bound=Union[StateLike, None], default=StateT)


class StateGraph(Generic[StateT, InputT, OutputT]):
    # 如下是 type hint, 
    # 以下在 __init__ 方法中初始化为相应的空集合/字典/默认值字典
    nodes: dict[str, StateNodeSpec]
    edges: set[tuple[str, str]]
    waiting_edges: set[tuple[tuple[str, ...], str]]
    # self.add_edge(["foo", "bar"], "next") -> self.waiting_edges.add(("foo", "bar"), "next")
    # self.add_edge("foo", "next") -> self.edges.add("foo", "next")
    branches: defaultdict[str, dict[str, Branch]]
    # add_conditional_edges(self, source, path, path_map) 的核心代码仅有:
    # 这里的 name 一般是 path 的函数名. TODO: 同一个 source 节点有两个 conditional_edge 会怎么执行呢?
    # TODO: Branch 是什么?
    # self.branches[source][name] = Branch.from_path(path, path_map, True)
    # if schema := self.branches[source][name].input_schema:
    #     self._add_schema(schema)
    
    
    channels: dict[str, BaseChannel]
    managed: dict[str, ManagedValueSpec]
    state_schema: type[StateT]
    input_schema: type[InputT]
    output_schema: type[OutputT]
    schemas: dict[type[Any], dict[str, BaseChannel | ManagedValueSpec]]
    # 以上属性在 __init__ 方法中受如下代码影响
    # self.input_schema = cast(type[InputT], input_schema or state_schema)
    # self.output_schema = cast(type[OutputT], output_schema or state_schema)
    # self._add_schema(self.state_schema)
    # self._add_schema(self.input_schema, allow_managed=False)
    # self._add_schema(self.output_schema, allow_managed=False)

    compiled: bool  # False
    # 在 __init__ 方法中除了初始化了上面的属性外, 还初始化了 self.config_schema
    # self.config_schema = config_schema


    # _add_schema 完整源码如下:
    # 维护了如下字典:
    # self.channels: dict[str, BaseChannel]
    # self.managed: dict[str, ManagedValueSpec]
    # self.schemas: dict[type[Any], dict[str, BaseChannel | ManagedValueSpec]]
    def _add_schema(self, schema: type[Any], /, allow_managed: bool = True) -> None:
        if schema not in self.schemas:
            _warn_invalid_state_schema(schema)
            # _get_channels 是针对 schema 中的每个字段注解, 确定其合适的 BaseChannel 或 ManagedValueSpec, 优先顺序如下
            # 注意只有字段的注解用 Annotated[...] 这种形式时, 才会做如下(1)(2)(3)优先判断
            # (1) 判断是否为 ManagedValue 的子类
            # (2) 判断是否为明确的 BaseChannel 子类
            # (3) 判断是否为 callable, 且 POSITIONAL_ONLY 或 POSITIONAL_OR_KEYWORD 仅有 2 个, 则判定为 BinaryOperatorAggregate
            # 默认为: name: LastValue(annotation)
            # 备注: LastValue 和 BinaryOperatorAggregate 均为 BaseChannel
            channels, managed, type_hints = _get_channels(schema)
            if managed and not allow_managed:
                names = ", ".join(managed)
                schema_name = getattr(schema, "__name__", "")
                raise ValueError(f"Invalid managed channels detected in {schema_name}: {names}. Managed channels are not permitted in Input/Output schema.")
            self.schemas[schema] = {**channels, **managed}
            for key, channel in channels.items():
                if key in self.channels:
                    if self.channels[key] != channel:
                        if isinstance(channel, LastValue):
                            pass
                        else:
                            raise ValueError(f"Channel '{key}' already exists with a different type")
                else:
                    self.channels[key] = channel
            for key, managed in managed.items():
                if key in self.managed:
                    if self.managed[key] != managed:
                        raise ValueError(f"Managed value '{key}' already exists with a different type")
                else:
                    self.managed[key] = managed
    
    def add_node(
        self,
        node: str | StateNode[StateT],
        action: StateNode[StateT] | None = None,
        *,
        defer: bool = False,
        metadata: dict[str, Any] | None = None,
        input_schema: type[Any] | None = None,
        retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
        cache_policy: CachePolicy | None = None,
        destinations: dict[str, str] | tuple[str, ...] | None = None,
        **kwargs: Unpack[DeprecatedKwargs],
    ) -> Self:
        # 此处省略一堆逻辑, 都是为了确定下面代码所用的变量, 对类本身的修改只有如下
        # ...
        if input_schema is not None:
            self._add_schema(input_schema)
        
        # StateNodeSpec 仅仅是个 namedtuple, 没有额外功能
        self.nodes[node] = StateNodeSpec(
            coerce_to_runnable(action, name=node, trace=False),
            metadata,  # 用途未知, 入参的 metadata 参数
            input=input_schema or self.state_schema,  # 根据 action 的入参类型注解确定 (好像一般node的入参和self.state_schema是一样的)
            retry_policy=retry_policy,  # 用途未知, 入参的 retry_policy 参数
            cache_policy=cache_policy,  # 用途未知, 入参的 cache_policy 参数
            ends=ends,  # 用途未知, ends 由入参的 destination 以及 action 返参类型来确定
            defer=defer,  # 用途未知, 入参的 defer 参数
        )
        return self
```

StateGraph 中用到的一个函数: `_get_channels`

```python
from langgraph.graph.message import add_messages
from langgraph.channels import EphemeralValue, Topic
from typing import TypedDict
from langgraph.graph.state import _get_channels, _get_channel
from langgraph.managed.is_last_step import IsLastStepManager

class State(TypedDict):
    x: Annotated[int, add_messages]
    y: Annotated[str, EphemeralValue]
    z: list[str]
    w: Annotated[bool, IsLastStepManager]

_get_channels(State)[-1]
# 基本上等同于这个
# {
#     name: _get_channel(name, tyh)
#     for name, tyh in
#     get_type_hints(State, include_extras=True).items()
# }

# 结果:
# {'x': <langgraph.channels.binop.BinaryOperatorAggregate at 0x7f8833fffb80>,
#  'y': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7f8833ffc140>,
#  'z': <langgraph.channels.last_value.LastValue at 0x7f8833ffd580>,
#  'w': langgraph.managed.is_last_step.IsLastStepManager}
```

### langgraph.graph.state.CompiledStateGraph

继承关系如下:

```python
class CompiledStateGraph(
    Pregel[StateT, InputT, OutputT],
    Generic[StateT, InputT, OutputT]
):
    # ...
```

### Pregel

一般的描述如下(来源于chatgpt, 但看很多资料有类似的描述), TODO: 确认这个描述, 以及 merge updates 怎么保证顺序性?

```
For each superstep:
    1. Determine all active nodes (those receiving updates)
    2. Run all active nodes concurrently
    3. Each node:
        - Reads state from Channels
        - Returns updates (to state and possibly messages)
    4. Merge updates to channels
    5. Decide next active nodes based on updated channels
Repeat until no more active nodes (or reaches recursion_limit)
```

## 源码目录

## Pregel

### TODO

怎么写出等价的 Pregel, 注意这里的 node 看起来是在往相同的 channel 里写东西, 似乎会造成 channel 里的东西不断被修改永远执行下去

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict

class State(TypedDict):
    x: int

def node1(state: State) -> State:
    state["x"] += 1
    return state

def node2(state: State) -> State:
    state["x"] *= 2
    return state

def node3(state: State) -> State:
    state["x"] **= 2
    return state

builder = StateGraph(State)
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_node("node3", node3)
builder.set_entry_point("node1")
builder.add_edge("node1", "node2")
builder.add_edge("node2", "node3")
builder.set_finish_point("node3")
graph = builder.compile()

initial_state = {"x": 1}

result = graph.invoke(initial_state)
print(result)
```

**以下是错的**

```python
from langgraph.channels import EphemeralValue, Topic
from langgraph.pregel import Pregel, NodeBuilder
from typing import TypedDict, Dict

class State(TypedDict):
    x: int

node1 = (
    NodeBuilder()
    .subscribe_only("x")
    .do(lambda x: x + 1)
    .write_to("x")
)

node2 = (
    NodeBuilder()
    .subscribe_only("x")
    .do(lambda x: x * 2)
    .write_to("x")
)

node3 = (
    NodeBuilder()
    .subscribe_only("x")
    .do(lambda x: x ** 2)
    .write_to("x")
)

channels = {
    "x": EphemeralValue(int),
}

app = Pregel(
    nodes={"node1": node1, "node2": node2, "node3": node3},
    channels=channels,
    input_channels=["x"],
    output_channels=["x"],
)

initial_state = {"x": 1}

result = app.invoke(initial_state)
print(result)
```

### Channel

channel 主要提供的对外接口是 `update` 和 `get`

```python
from langgraph.channels import EphemeralValue, Topic

channel = EphemeralValue(str)
# EphemeralValue 类型必须至少 update 一次后才能调用, get. 可以多次get, 将得到同样的结果
# update 的参数必须是一个长度为 1 的 list
channel.update(["123"])
channel.get()  # 返回 "123"


# Topic 设置 accumulate=True, 则新值被 extend, 假设设置 accumulate=False (默认值), 则新值直接覆盖旧值
channel = Topic(str, accumulate=True)
channel.update(['11', '22'])  # list 长度任意
channel.update(['33', '44'])
value = channel.get()  # 返回 ['11', '22', '33', '44']
```

langgraph 内置的 channel 有如下: [channels/__init__.py](https://github.com/langchain-ai/langgraph/blob/0.5.0/libs/langgraph/langgraph/channels/__init__.py)

```python
from langgraph.channels.any_value import AnyValue
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic
from langgraph.channels.untracked_value import UntrackedValue
```

TODO: 各个的介绍, 疑问如下: LastValue 与 EphemeralValue 的区别. AnyValue 和 UntrackedValue 似乎文档上提及较少. Topic 有什么用?
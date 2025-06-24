---
layout: post
title: "(P1) langgraph"
date: 2025-06-24 09:05:04 +0800
labels: [langchain,langgraph]
---

## 动机、参考资料、涉及内容

- langgraph 源码阅读
- langgraph 使用, 并以此了解一些 fancy 的 Agent 实现

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
# EphemeralValue 类型必须至少 update 一次后才能调用, get. 可以多次get, 将得到同样的结果’
# update 的参数必须是一个长度为 1 的 list
channel.update(["123"])
channel.get()  # 返回 "123"


# Topic 设置 accumulate=True, 则新值被 extend, 假设设置 accumulate=False (默认值), 则新值直接覆盖旧值
channel = Topic(str, accumulate=True)
channel.update(['11', '22'])  # list 长度任意
channel.update(['33', '44'])
value = channel.get()  # 返回 ['11', '22', '33', '44']
```

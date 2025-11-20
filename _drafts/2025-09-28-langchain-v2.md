---
layout: post
title: "(P0) 再探 langchain"
date: 2025-09-28 09:05:04 +0800
labels: [langchain,langgraph,langsmith,langgraph-platform]
---

## 动机、参考资料、涉及内容

主要看 langsmith, langgraph-platform 相关内容, 也看一些关于 langchain/langgraph 的新内容

## langchain v1 生态及代码整体分布



## langsmith: Tracer 底层实现

用法可参考官方文档

```python
# langchain_core.runnables.config.RunnableConfig
class RunnableConfig(TypedDict, total=False):
    # 被 CallbackManager.configure 用到
    tags: list[str]
    metadata: dict[str, Any]
    callbacks: Callbacks

    # 不被 CallbackManager.configure 用到
    run_name: str
    max_concurrency: Optional[int]
    recursion_limit: int
    configurable: dict[str, Any]
    run_id: Optional[uuid.UUID]

# langsmith.schemas.RunBase
# run <=> opentelemetry 里的 span
class RunBase(BaseModel):
    id: UUID
    name: str
    start_time: datetime
    run_type: str
    end_time: Optional[datetime] = None
    extra: Optional[dict] = Field(default_factory=_default_extra)
    error: Optional[str] = None
    serialized: Optional[dict] = None
    events: Optional[list[dict]] = None
    inputs: dict = Field(default_factory=dict)
    outputs: Optional[dict] = None
    reference_example_id: Optional[UUID] = None
    parent_run_id: Optional[UUID] = None
    tags: Optional[list[str]] = None
    attachments: Union[Attachments, dict[str, AttachmentInfo]] = Field(default_factory=dict)
    
    @property
    def metadata(self) -> dict[str, Any]:
        if self.extra is None:
            self.extra = {}
        return self.extra.setdefault("metadata", {})
    @property
    def revision_id(self) -> Optional[UUID]:
        return self.metadata.get("revision_id")

    @property
    def latency(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

# langsmith.run_trees.RunTree
# RunTree 对于 self.client 的使用只用到了它的 create_run, update_run, get_run_url 方法
class RunTree(ls_schemas.RunBase):
    # ls_schemas.RunBase 已有的字段
    name: str
    id: UUID = Field(default_factory=uuid4)
    run_type: str = Field(default="chain")
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict = Field(default_factory=dict)
    # error: Optional[str] = None
    # serialized: Optional[dict] = None
    events: list[dict] = Field(default_factory=list)
    # inputs: dict = Field(default_factory=dict)
    # outputs: Optional[dict] = None
    # reference_example_id: Optional[UUID] = None
    # parent_run_id: Optional[UUID] = None
    tags: Optional[list[str]] = Field(default_factory=list)
    # attachments: Union[Attachments, dict[str, AttachmentInfo]] = Field(default_factory=dict)

    # 新增字段
    # Note: no longer set.
    parent_run: Optional[RunTree] = Field(default=None, exclude=True)
    parent_dotted_order: Optional[str] = Field(default=None, exclude=True)
    child_runs: list[RunTree] = Field(
        default_factory=list,
        exclude={"__all__": {"parent_run_id"}},
    )
    session_name: str = Field(
        default_factory=lambda: utils.get_tracer_project() or "default",
        alias="project_name",
    )
    session_id: Optional[UUID] = Field(default=None, alias="project_id")
    ls_client: Optional[Any] = Field(default=None, exclude=True)
    dotted_order: str = Field(default="", description="The order of the run in the tree.")
    trace_id: UUID = Field(default="", description="The trace id of the run.")  # type: ignore
    dangerously_allow_filesystem: Optional[bool] = Field(default=False, description="Whether to allow filesystem access for attachments.")
    replicas: Optional[Sequence[WriteReplica]] = Field(default=None, description="Projects to replicate this run to with optional updates.")

    @property
    def client(self) -> Client:
        """Return the client."""
        # Lazily load the client
        # If you never use this for API calls, it will never be loaded
        if self.ls_client is None:
            self.ls_client = get_cached_client()
        return self.ls_client

    @property
    def _client(self) -> Optional[Client]:
        # For backwards compat
        return self.ls_client
```

## langgraph: add_messages

### 特殊行为: RemoveMessage

left 是 state 的现有的 message, 里面必然不包含 RemoveMessage, 每次合并后的结果里也不会包含 RemoveMessage, 也就是 RemoveMessage 只有可能出现在 right 中

```python
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage

left = [
    HumanMessage("x", id="1"),
    AIMessage("y", id="2")
]

# RemoveMessage + id=REMOVE_ALL_MESSAGES, 则只保留 right 在最后一个 REMOVE_MESSAGE 之后的内容
# PR: 感觉源码这里应该再做一次检查, 确保之后没有其他 id 为非 REMOVE_ALL_MESSAGES 的 REMOVE_MESSAGE
# 输出: w 4;
right1 = [
    HumanMessage("z", id="3"),
    RemoveMessage(id=REMOVE_ALL_MESSAGES),
    AIMessage("w", id="4")
]

# 删除前边的某条 message, 可以是 left, 也可以是 right 中的 message
# 输出: y 2; z 3; w 4;
right2 = [
    HumanMessage("z", id="3"),
    RemoveMessage(id="1"),
    AIMessage("w", id="4")
]

# id 与之前的重复, 则覆盖之前的 message
# 输出: z 1; y 2; w 4;
right3 = [
    HumanMessage("z", id="1"),
    AIMessage("w", id="4")
]

# 覆盖和删除同时出现, 则按逻辑顺序执行
# 输出: y 2; w 4;
right4 = [
    HumanMessage("z", id="1"),
    RemoveMessage(id="1"),
    AIMessage("w", id="4")
]

for right in [right1, right2, right3, right4]:
    add_messages(left, right)
```

#### 例子

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-5-mini")

agent = create_agent(
    model=model,
    middleware=[
        SummarizationMiddleware(
            model=model,
            messages_to_keep=2,
            max_tokens_before_summary=200
        ),
    ],
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Explain machine learning"*100},
            {"role": "assistant", "content": "something"},
            {"role": "user", "content": "why i need it" * 100},
        ]
    },
)
```

Middleware 一般只适用于 `create_agent` 接口, 其本质是将 `middleware.before_model` 之类的方法作为节点连在 graph 中. 注意到 SummarizationMiddleware 的 return 结果为:

```python
return {
    "messages": [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),
        *new_messages,
        *preserved_messages,
    ]
}
```

而普通的 ChatOpenAI 的 invoke 方法实际上不支持这种特殊的 RemoveMessage. 并且在 `create_agent` 得到的 graph 的 state 中, messages 字段的 reducer 是被标记为了 `add_messages` 的. 因此真正的执行逻辑是:

- 执行 `SummarizationMiddleware.before_model`: 有可能执行后返回上面的 RemoveMessage
- 做状态更新: `add_messages(left, right)`, 此时 `right` 的实参为上面的返回结果, 这里有特殊逻辑: 直接返回 `right` RemoveMessage 之后的 message
- 执行模型

## langchain_core: `BaseMessage`

- BaseMessage
- AIMessage
- HumanMessage
- ChatMessage
- SystemMessage
- FunctionMessage
- ToolMessage
- AIMessageChunk
- ...

```python
class BaseMessage:
    content: str | list[str | dict]
    additional_kwargs: dict = Field(default_factory=dict)
    response_metadata: dict = Field(default_factory=dict)
    type: str
    name: str | None = None
    id: str | None = Field(default=None, coerce_numbers_to_str=True)

```

## langchain_core: `trim_messages`

langchain_core.messages.utils.convert_to_messages
- trim_messages 有用到
- add_messages 有用到

langchain_core.messages.utils.convert_to_openai_messages
- add_messages 有用到
---
layout: post
title: "(P0) 再探 langchain"
date: 2025-09-28 09:05:04 +0800
labels: [langchain,langgraph,langsmith,langgraph-platform]
---

## 动机、参考资料、涉及内容

主要看 langsmith, langgraph-platform 相关内容, 也看一些关于 langchain/langgraph 的新内容

## langsmith: Tracer

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


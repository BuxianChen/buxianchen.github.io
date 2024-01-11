---
layout: post
title: "(P0) llama_index"
date: 2024-01-11 11:10:04 +0800
labels: [llm]
---


**总体**

一个伪代码例子

```python
store1 = Store1(...)
store2 = Store2(...)
# 实际常使用 from_defaults 这个 classmethod 来构造
storage_context = StorageContext(store1, store2)

llm = LLM(...)
embed_model = EmbedModel(...)
prompt_helper = PromptHelper(...)
transformation = Transformation(...)
# 实际常使用 from_defaults 这个 classmethod 来构造
service_context = ServiceContext(llm, embed_model, prompt_helper, transformation)

# 实际常使用 from_documents 这个 classmethod 来构造
index = Index(
    documents,
    storage_context=storage_context,
    service_context=service_context,
    ...
)

query_engine: QueryEngine = index.as_query_engine()
query_engine.query("what is 1")
chat_engine: ChatEngine = index.as_chat_engine()
chat_engine.chat("2+2")
```

**Document**

基本上就是带一些验证逻辑的 `pydantic.v1.BaseModel`, 用了几层的继承: `pydantic.v1.BaseModel` -> `BaseComponent` -> `BaseNode` -> `TextNode` -> `Document`, 这里只简单看下字段

```python
# BaseComponent: 没有字段
# BaseNode:
id_: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID of the node.")
embedding: Optional[List[float]] = Field(default=None, description="Embedding of the node.")
metadata: Dict[str, Any] = Field(default_factory=dict, description="A flat dictionary of metadata fields", alias="extra_info",)
excluded_embed_metadata_keys: List[str] = Field(default_factory=list, description="Metadata keys that are excluded from text for the embed model.",)
excluded_llm_metadata_keys: List[str] = Field(default_factory=list, description="Metadata keys that are excluded from text for the LLM.",)
relationships: Dict[NodeRelationship, RelatedNodeType] = Field(default_factory=dict, description="A mapping of relationships to other node information.",)
hash: str = Field(default="", description="Hash of the node content.")
# TextNode:
text: str = Field(default="", description="Text content of the node.")
start_char_idx: Optional[int] = Field(default=None, description="Start char index of the node.")
end_char_idx: Optional[int] = Field(default=None, description="End char index of the node.")
text_template: str = Field(default=DEFAULT_TEXT_NODE_TMPL, description="Template for how text is formatted, with {content} and {metadata_str} placeholders.",)
metadata_template: str = Field(default=DEFAULT_METADATA_TMPL, description="Template for how metadata is formatted, with {key} and {value} placeholders.",)
metadata_seperator: str = Field(default="\n", description="Separator between metadata fields when converting to string.",)
# Document:
id_: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID of the node.", alias="doc_id",)
_compat_fields = {"doc_id": "id_", "extra_info": "metadata"}
```



**GraphStore**

继承关系:

`typing.Generic` -> `typing.Protocol` -> `llama_index.graph_stores.types.GraphStore` (抽象类) -> `llama_index.graph_stores.Neo4jGraphStore` (对 neo4j-python 的简单包装, 不涉及大模型)


<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td style="width: 100%; word-wrap: break-word; padding=5px; border: 1px solid #ccc; vertical-align: top;"><div markdown="1">
使用 Neo4jGraphStore

```python
from llama_index.graph_stores import Neo4jGraphStore

username = "neo4j"
password = "12345678"
url = "bolt://localhost:7687"
database = "neo4j"

graph_store = Neo4jGraphStore(
    username=username,
    password=password,
    url=url,
    database=database,
    # node_label = "Entity"  # default
)

records = graph_store.get(subj)
print(records)
```
</div></td>
    <td style="width: 100%; word-wrap: break-word; padding=5px; border: 1px solid #ccc; vertical-align: top;"><div markdown="1">
直接使用原生的 neo4j

```python
import neo4j

username = "neo4j"
password = "12345678"
url = "bolt://localhost:7687"
database = "neo4j"
node_label = "Entity"

driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
query = """
    MATCH (n1:%s)-[r]->(n2:%s)
    WHERE n1.id = $subj
    RETURN type(r), n2.id;
"""
prepared_statement = query % (node_label, node_label)

subj = "Saldaña"
with driver.session(database=database) as session:
    data = session.run(prepared_statement, {"subj": subj})
    records = [record.values() for record in data]
print(records)
```
</div></td>
  </tr>
</table>

结果:

```
[
    ['SAID','she would like to see Gamora portrayed as "the most lethal woman in the galaxy"'],
    ['SAID', 'she would like to see Gamora rejoin the Guardians'],
    ['SAID', "Gamora's fate would depend on the plans that Marvel and Gunn have for Vol. 3"],
    ['RETURNED', 'to play a younger version in Endgame'],
    ['ASKED_ABOUT', 'her role in the film'],
    ['SIGNED_TO_PLAY', 'Gamora in one film'],
    ['STATED_THAT', 'Vol. 3 would be the final time she would portray Gamora']
]
```

**StorageContext**

官方文档中大量存在 `StorageContext.from_defaults` 的使用, `StorageContext` 本质上只是一个字典, 没有更多东西

```python
from llama_index.storage.storage_context import StorageContext
storage_context = StorageContext.from_defaults(graph_store=graph_store)
```

`llama_index.storage.Storage` 的源代码 (v0.9.27) 如下:

```python
# llama_index/storage/storage_context.py

DEFAULT_PERSIST_DIR = "./storage"
IMAGE_STORE_FNAME = "image_store.json"
IMAGE_VECTOR_STORE_NAMESPACE = "image"

@dataclass
class StorageContext:
    docstore: BaseDocumentStore
    index_store: BaseIndexStore
    vector_stores: Dict[str, VectorStore]
    graph_store: GraphStore

    @classmethod
    def from_defaults(
        cls,
        docstore: Optional[BaseDocumentStore] = None,
        index_store: Optional[BaseIndexStore] = None,
        vector_store: Optional[Union[VectorStore, BasePydanticVectorStore]] = None,
        image_store: Optional[VectorStore] = None,
        vector_stores: Optional[Dict[str, Union[VectorStore, BasePydanticVectorStore]]] = None,
        graph_store: Optional[GraphStore] = None,
        persist_dir: Optional[str] = None,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "StorageContext":
        if persist_dir is None:
            docstore = docstore or SimpleDocumentStore()
            index_store = index_store or SimpleIndexStore()
            graph_store = graph_store or SimpleGraphStore()
            image_store = image_store or SimpleVectorStore()

            if vector_store:
                vector_stores = {DEFAULT_VECTOR_STORE: vector_store}
            else:
                vector_stores = vector_stores or {DEFAULT_VECTOR_STORE: SimpleVectorStore()}
            if image_store:
                vector_stores[IMAGE_VECTOR_STORE_NAMESPACE] = image_store  # append image store to vector stores
        else:
            docstore = docstore or SimpleDocumentStore.from_persist_dir(persist_dir, fs=fs)
            index_store = index_store or SimpleIndexStore.from_persist_dir(persist_dir, fs=fs)
            graph_store = graph_store or SimpleGraphStore.from_persist_dir(persist_dir, fs=fs)

            if vector_store:
                vector_stores = {DEFAULT_VECTOR_STORE: vector_store}
            elif vector_stores:
                vector_stores = vector_stores
            else:
                vector_stores = SimpleVectorStore.from_namespaced_persist_dir(persist_dir, fs=fs)
            if image_store:
                vector_stores[IMAGE_VECTOR_STORE_NAMESPACE] = image_store  # append image store to vector stores

        return cls(docstore=docstore, index_store=index_store, vector_stores=vector_stores, graph_store=graph_store,)

    def to_dict(self) -> dict: ...  # 略

    @classmethod
    def from_dict(cls, save_dict: dict) -> "StorageContext": ...  # 略

    @property
    def vector_store(self) -> VectorStore: ...  # 略

    def add_vector_store(self, vector_store: VectorStore, namespace: str) -> None:
        self.vector_stores[namespace] = vector_store
```


**LLM**

略

**Embedding Model**

略

**ServiceContext**

`ServiceContext` 也类似于 `StorageContext`, 仅仅是 LLM, PromptHelper, Embedding Model, transformations 的集合

```python
@dataclass
class ServiceContext:
    llm_predictor: BaseLLMPredictor
    prompt_helper: PromptHelper
    embed_model: BaseEmbedding
    transformations: List[TransformComponent]
    llama_logger: LlamaLogger  # 准备弃用
    callback_manager: CallbackManager

    # 入参与field的对应关系见注释
    @classmethod
    def from_defaults(
        cls,
        llm_predictor: Optional[BaseLLMPredictor] = None,          # 准备弃用参数, 使用 llm 参数: llm_predictor
        llm: Optional[LLMType] = "default",                        # llm_predictor
        prompt_helper: Optional[PromptHelper] = None,              # prompt_helper
        embed_model: Optional[Any] = "default",                    # embed_model
        node_parser: Optional[NodeParser] = None,                  # transformation
        text_splitter: Optional[TextSplitter] = None,              # transformation
        transformations: Optional[List[TransformComponent]] = None,# transformation
        llama_logger: Optional[LlamaLogger] = None,                # llama_logger
        callback_manager: Optional[CallbackManager] = None,        # callback_manager, transformation
        system_prompt: Optional[str] = None,                       # llm_predictor
        query_wrapper_prompt: Optional[BasePromptTemplate] = None, # llm_predictor
        # pydantic program mode (used if output_cls is specified)
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,  # llm_predictor
        # node parser kwargs
        chunk_size: Optional[int] = None,                          # transformation
        chunk_overlap: Optional[int] = None,                       # transformation
        # prompt helper kwargs
        context_window: Optional[int] = None,                      # prompt_helper
        num_output: Optional[int] = None,                          # prompt_helper
        # deprecated kwargs
        chunk_size_limit: Optional[int] = None,                    # 准备弃用参数, 使用 chunk_size 参数: transformation
    ) -> "ServiceContext": ... # 略

    # 其余均略去
```

**KnowledgeGraphIndex**

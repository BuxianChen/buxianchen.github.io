---
layout: post
title: "(P1) Langchain"
date: 2023-11-22 11:10:04 +0800
---

## 涉及内容

- [langchain v0.1.0 preview blog (2023/12/12)](https://blog.langchain.dev/the-new-langchain-architecture-langchain-core-v0-1-langchain-community-and-a-path-to-langchain-v0-1/)
- [langchain v 0.1.0 blog (2024/01/08)](https://blog.langchain.dev/langchain-v0-1-0/), 里面也提及了许多使用 Langchain 来构建的项目.

[引用](https://blog.langchain.dev/the-new-langchain-architecture-langchain-core-v0-1-langchain-community-and-a-path-to-langchain-v0-1/)

> `langchain-core` contains simple, core abstractions that have emerged as a standard, as well as LangChain Expression Language as a way to compose these components together. This package is now at version 0.1 and all breaking changes will be accompanied by a minor version bump.

> `langchain-community` contains all third party integrations. We will work with partners on splitting key integrations out into standalone packages over the next month.

> `langchain` contains higher-level and use-case specific chains, agents, and retrieval algorithms that are at the core of your application's cognitive architecture. We are targeting a launch of a stable 0.1 release for langchain in early January.


```
libs/
  - langchain/
  - core/langchain_core/                  # 关键抽象, 尽量少包含三方库
  - community/langchain_community/        # ??
  - partners/
    - openai/langchain_openai/            # 集成组件, 一般是对三方库的简单包装
    - anthropic/langchain_anthropic/
  - cli/langchain_cli/
  - experimental/langchain_experimental/  # ?? 实验组件, 可忽略?
```

python `import` 语句的准则 (笔者观点):

- 自定义组件一般 `import langchain_core`
- 使用官方的组合组件的方式一般用 `import langchain`
- 官方已支持的第三方集成优先用 partner, 然后考虑 `langchain_community`

```python
from langchain_openai.chat_models import ChatOpenAI   # 最优先, 但不是每个第三方集成都会做成一个单独的包
from langchain_community.llms.openai import OpenAIChat  # 次优先
from langchain.llms.openai import OpenAIChat  # 本质上与第二种一致, 但这只是官方的兼容性保证的做法, 不推荐
```

## Tutorial

Langchain 的本质就是以一种作者认为的模块化的方式进行提示工程.

引用: [https://python.langchain.com/docs/get_started/quickstart](https://python.langchain.com/docs/get_started/quickstart)

<blockquote>
<p>LangChain provides many modules that can be used to build language model applications. Modules can be used as standalones in simple applications and they can be composed for more complex use cases. Composition is powered by <b>LangChain Expression Language (LCEL)</b>, which defines a unified <code>Runnable</code> interface that many modules implement, making it possible to seamlessly chain components.</p>

<p>The simplest and most common chain contains three things:</p>

<li><b>LLM/Chat Model</b>: The language model is the core reasoning engine here. In order to work with LangChain, you need to understand the different types of language models and how to work with them.</li>
<li><b>Prompt Template</b>: This provides instructions to the language model. This controls what the language model outputs, so understanding how to construct prompts and different prompting strategies is crucial.</li>
<li><b>Output Parser</b>: These translate the raw response from the language model to a more workable format, making it easy to use the output downstream.</li>
</blockquote>

理解这段话需要借助一下继承关系图

![](../assets/figures/langchain/langchain.png)

继承关系图说明: 以红色作为框线的方框代表的是实际可运行的类(其余均为抽象类), 由于 `langchain>=0.0.339rc0` (2023/11/22) 开始, langchain 代码库进行了一些重构, 主要是将一部分内容单独抽出来放在了 `langchain_core` 模块中, 同一个框中的两个类是别名关系.

LLM/Chat Model 的一个实际例子是 `ChatOpenAI` 类, Prompt Template 的一个实际例子是 `PromptTemplate` 类, 而 Output Parser 需要用户自己继承自 `BaseOutputParser`. 而这三类东西都继承自 `Runable` 抽象类, 这种继承自 `Runable` 的类都称为 ICEL. 所以如果希望研究源码, 可以先研究 `Runable` 抽象类. 在此之前先看一些例子:

以下例子参考: [https://python.langchain.com/docs/get_started/quickstart](https://python.langchain.com/docs/get_started/quickstart)

### 例子1 (llm, prompt, output parser, LCEL basics)

参考自: [https://python.langchain.com/docs/get_started/quickstart#llm-chain](https://python.langchain.com/docs/get_started/quickstart#llm-chain)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])
llm = ChatOpenAI(openai_api_key="sk-xx")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser  # ICEL
s: str = chain.invoke({"input": "how can langsmith help with testing?"})
```

### 例子2 (模型自定义, langserver)

```bash
pip install langchain langserve sse_starlette
```

使用 `python serve.py` 运行, 界面在 `http://localhost:8000/chain/playground/`

```python
#!/usr/bin/env python
# serve.py
from typing import Any, List, Mapping, Optional
from fastapi import FastAPI
from langchain.schema import BaseOutputParser
from langserve import add_routes
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

class CustomLLM(LLM):
    n: int

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[:self.n]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}

class CustomOutputParser(BaseOutputParser[str]):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str) -> str:
        """Parse the output of an LLM call."""
        return text

prompt_template = PromptTemplate.from_template("instruction: {instruction}\nquestion{q}")
llm = CustomLLM(n=30)
parser = CustomOutputParser()
chain = prompt_template | llm | parser
# chain.invoke({"instruction": "1234567890123", "q": "qa"})


# 2. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

# 3. Adding chain route
add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

### 例子 3 (index, RAG)

参考自: [https://python.langchain.com/docs/get_started/quickstart#retrieval-chain](https://python.langchain.com/docs/get_started/quickstart#retrieval-chain)

**PART 1**

```python
# pip install beautifulsoup4 faiss-cpu
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

from langchain_core.documents.base import Document
from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
# Document has two attrs: page_content: str, metadata: Dict
docs: List[Document] = loader.load()  # metadata keys: ['source', 'title', 'description', 'language']
text_splitter = RecursiveCharacterTextSplitter()
documents: List[Document] = text_splitter.split_documents(docs)  # metadata keys: ['source', 'title', 'description', 'language']
vector = FAISS.from_documents(documents, embeddings)
# vector.docstore._dict: Dict[str, Document]  # {"ass": Document(...), "bss": Document(...)}
# vector.index_to_docstore_id: Dict[int, str]  # {0: "ass", 1: "bss"}
# vector.index: faiss.swigfaiss_avx2.IndexFlatL2
```

继承关系图

```
# vector: FAISS
langchain_core.vectorstores.VectorStore(ABC):
langchain_community.vectorstores.FAISS(VectorStore)

# vector.docstore: InMemoryDocstore
langchain_community.docstore.base.AddableMixin(ABC): search, delete
langchain_community.docstore.base.Docstore(ABC): add
langchain_community.docstores.in_memory.InMemoryDocstore(Docstore, AddableMixin)
```

**PART 2**

```python
# pip install grandalf
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import langchain_core

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain: langchain_core.runnables.base.RunnableBinding = create_stuff_documents_chain(llm, prompt)

# 一些探索
document_chain.get_graph().nodes
document_chain.get_graph().edges
print(document_chain.get_graph().draw_ascii())
```

输出

```
# nodes:

{'10df13934301478abd3ba5dd8de10598': Node(id='10df13934301478abd3ba5dd8de10598', data=<class 'pydantic.v1.main.RunnableParallel<context>Input'>),
 'a087b655b2cd4fb49951b3170874329d': Node(id='a087b655b2cd4fb49951b3170874329d', data=<class 'pydantic.v1.main.RunnableParallel<context>Output'>),
 '24444bcbd4e04ac3b6b7a9b75ea70b2d': Node(id='24444bcbd4e04ac3b6b7a9b75ea70b2d', data=PromptTemplate(input_variables=['page_content'], template='{page_content}')),
 '87f1b6b35bed4f128ecafb41c328ae45': Node(id='87f1b6b35bed4f128ecafb41c328ae45', data=RunnablePassthrough()),
 '706cdd684df24146b0b8985fedf4312e': Node(id='706cdd684df24146b0b8985fedf4312e', data=ChatPromptTemplate(input_variables=['context', 'input'], messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'input'], template='Answer the following question based only on the provided context:\n\n<context>\n{context}\n</context>\n\nQuestion: {input}'))])),
 '19fc2995c60745509e4bdebc951e74f9': Node(id='19fc2995c60745509e4bdebc951e74f9', data=ChatOpenAI(client=<openai.resources.chat.completions.Completions object at 0x7f25b1e9f820>, async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x7f25b1ebbcd0>, openai_api_key='sk-xxx', openai_proxy='')),
 '82c0eea4218b4ba8ab35dc8a08364f6b': Node(id='82c0eea4218b4ba8ab35dc8a08364f6b', data=StrOutputParser()),
 'e82fa2229583412eaf4cea0d32818487': Node(id='e82fa2229583412eaf4cea0d32818487', data=<class 'pydantic.v1.main.StrOutputParserOutput'>)}

# edges:

[Edge(source='418e866578f840c8b9e8954000a16883', target='e77a1ecf38c14457bd6606305bbf040b'),
 Edge(source='e77a1ecf38c14457bd6606305bbf040b', target='09a9f05f03e540c3afb008d7eb4bcdfa'),
 Edge(source='418e866578f840c8b9e8954000a16883', target='a424ccc59e3b4b4197333c6e2ee8dcf7'),
 Edge(source='a424ccc59e3b4b4197333c6e2ee8dcf7', target='09a9f05f03e540c3afb008d7eb4bcdfa'),
 Edge(source='09a9f05f03e540c3afb008d7eb4bcdfa', target='497f4a32536440319ad796fe51d26b47'),
 Edge(source='497f4a32536440319ad796fe51d26b47', target='9ded27aa029d41a6a51e9a5af9600955'),
 Edge(source='1fd84bd773be42a1911a64a1424af96c', target='ddf23954923f433ba714f6a29e2670d8'),
 Edge(source='9ded27aa029d41a6a51e9a5af9600955', target='1fd84bd773be42a1911a64a1424af96c')]

# draw_ascii:
           +------------------------+            
           | Parallel<context>Input |            
           +------------------------+            
                ***            ***               
              **                  **             
            **                      **           
+----------------+              +-------------+  
| PromptTemplate |              | Passthrough |  
+----------------+              +-------------+  
                ***            ***               
                   **        **                  
                     **    **                    
           +-------------------------+           
           | Parallel<context>Output |           
           +-------------------------+           
                        *                        
                        *                        
                        *                        
             +--------------------+              
             | ChatPromptTemplate |              
             +--------------------+              
                        *                        
                        *                        
                        *                        
                 +------------+                  
                 | ChatOpenAI |                  
                 +------------+                  
                        *                        
                        *                        
                        *                        
               +-----------------+               
               | StrOutputParser |               
               +-----------------+               
                        *                        
                        *                        
                        *                        
            +-----------------------+            
            | StrOutputParserOutput |            
            +-----------------------+            
```


**PART 3**

```python
from langchain_core.documents import Document

answer: str = document_chain.invoke({
    "input": "how can langsmith help with testing?",
    "context": [Document(page_content="langsmith can let you visualize test results")]
})  # answer: "Langsmith can help with testing by allowing users to visualize test results."


from langchain.chains import create_retrieval_chain
# VectorStoreRetriever 也继承自 langchain_core.runnables.base.Runnable
retriever: langchain_core.vectorstores.VectorStoreRetriever = vector.as_retriever()
retrieval_chain: langchain_core.runnables.base.RunnableBinding = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])
# "LangSmith can help with testing by providing various features ..."
print(retrieval_chain.get_graph().draw_ascii())
```

图示

```
                               +------------------------+                            
                               | Parallel<context>Input |                            
                               +------------------------+                            
                                ***                   ***                            
                            ****                         ***                         
                          **                                ****                     
       +------------------------------+                         **                   
       | Lambda(lambda x: x['input']) |                          *                   
       +------------------------------+                          *                   
                       *                                         *                   
                       *                                         *                   
                       *                                         *                   
           +----------------------+                       +-------------+            
           | VectorStoreRetriever |                       | Passthrough |            
           +----------------------+                      *+-------------+            
                                ***                   ***                            
                                   ****           ****                               
                                       **       **                                   
                              +-------------------------+                            
                              | Parallel<context>Output |                            
                              +-------------------------+                            
                                            *                                        
                                            *                                        
                                            *                                        
                                +-----------------------+                            
                                | Parallel<answer>Input |                            
                                +-----------------------+*                           
                                 ****                     *****                      
                              ***                              *****                 
                            **                                      ******           
           +------------------------+                                     ***        
           | Parallel<context>Input |                                       *        
           +------------------------+                                       *        
                ***            ***                                          *        
              **                  **                                        *        
            **                      **                                      *        
+----------------+              +-------------+                             *        
| PromptTemplate |              | Passthrough |                             *        
+----------------+              +-------------+                             *        
                ***            ***                                          *        
                   **        **                                             *        
                     **    **                                               *        
           +-------------------------+                                      *        
           | Parallel<context>Output |                                      *        
           +-------------------------+                                      *        
                        *                                                   *        
                        *                                                   *        
                        *                                                   *        
             +--------------------+                                         *        
             | ChatPromptTemplate |                                         *        
             +--------------------+                                         *        
                        *                                                   *        
                        *                                                   *        
                        *                                                   *        
                 +------------+                                             *        
                 | ChatOpenAI |                                             *        
                 +------------+                                             *        
                        *                                                   *        
                        *                                                   *        
                        *                                                   *        
               +-----------------+                                  +-------------+  
               | StrOutputParser |                                  | Passthrough |  
               +-----------------+                             *****+-------------+  
                                 ****                     *****                      
                                     ***            ******                           
                                        **       ***                                 
                               +------------------------+                            
                               | Parallel<answer>Output |                            
                               +------------------------+                            
```

### 例子 4 (langchainhub)

```python
from langchain import hub
prompt = hub.pull("hwchase17/openai-functions-agent")

import pickle
with open("openai-functions-agent.pkl", "wb") as fw:
    pickle.dump(prompt, fw)

import pickle
with open("openai-functions-agent.pkl", "rb") as fr:
    reload_prompt = pickle.load(fr)
```

### 例子 5 (async & stream)

**Python 脚本中**

```python
from langchain.chat_models import ChatOpenAI
import asyncio

async def main():
    model = ChatOpenAI()

    chunks = []
    async for chunk in model.astream("hello. tell me something about yourself"):
        chunks.append(chunk)
        print(chunk.content, end="|", flush=True)

asyncio.run(main())
# 基本等价于:
# asyncio.get_event_loop().run_until_complete(main())
```

**Jupyter 环境**

jupyter 环境里可以直接将 `async` 关键字写在外层

```python
from langchain.chat_models import ChatOpenAI
model = ChatOpenAI()

chunks = []
async for chunk in model.astream("hello. tell me something about yourself"):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)
```

### 例子 6 (callbacks, TODO: 待解释)

参考: [https://python.langchain.com/docs/expression_language/how_to/functions#accepting-a-runnable-config](https://python.langchain.com/docs/expression_language/how_to/functions#accepting-a-runnable-config)

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig, RunnableLambda

import json


def parse_or_fix(text: str, config: RunnableConfig):
    fixing_chain = (
        ChatPromptTemplate.from_template(
            "Fix the following text:\n\n```text\n{input}\n```\nError: {error}"
            " Don't narrate, just respond with the fixed data."
        )
        | ChatOpenAI()
        | StrOutputParser()
    )
    for _ in range(3):
        try:
            return json.loads(text)
        except Exception as e:
            text = fixing_chain.invoke({"input": text, "error": e}, config)
    return "Failed to parse"

from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    output = RunnableLambda(parse_or_fix).invoke(
        "{foo: bar}", {"tags": ["my-tag"], "callbacks": [cb]}
    )
    print(output)
    print(cb)
```

输出:

```
{'foo': 'bar'}
Tokens Used: 65
	Prompt Tokens: 56
	Completion Tokens: 9
Successful Requests: 1
Total Cost (USD): $0.00010200000000000001
```

### 例子 7 (ICEL Memory, redis, TODO: 与 modules 文档中的 memory 似乎不相干)

参考: [https://python.langchain.com/docs/expression_language/how_to/message_history](https://python.langchain.com/docs/expression_language/how_to/message_history)


```python
from typing import Optional

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.chat_models import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

REDIS_URL = "redis://localhost:6379/0"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an assistant who's good at {ability}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | ChatOpenAI()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: RedisChatMessageHistory(session_id, url=REDIS_URL, ttl=600),  # 设置失效时间为 600 秒
    input_messages_key="question",
    history_messages_key="history",
)

chain_with_history.invoke(
    {"ability": "math", "question": "What does cosine mean?"},
    config={"configurable": {"session_id": "foobar"}},
)

# 可以尝试直接调 chain, 不会写 history
# messages = chain_with_history.get_session_history("foobar").messages
# chain.invoke(
#     {
#         "ability": "math",
#         "question": "What's its inverse",
#         "history": chain_with_history.get_session_history("foobar").messages  # List[BaseMessage]
#     }
# )

chain_with_history.invoke(
    {"ability": "math", "question": "What's its inverse"},
    config={"configurable": {"session_id": "foobar"}},
)
```


备注: 这个例子里 redis 侧实际发生的事情如下: langchain 的实现是用 `lpush/lrange` 来存储/获取键值对, 而不是使用 `set/get` 的方式

```python
# docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
import redis
import json
from langchain_core.messages import messages_from_dict, message_to_dict
from langchain_core.messages import HumanMessage, AIMessage

REDIS_URL = "redis://localhost:6379/0"
redis_client = redis.from_url(url=REDIS_URL)

prefix = "message_store:"
session_id = "foobar"

# 追加历史:
message = HumanMessage(content='What does cosine mean?')
redis_client.lpush(prefix+session_id, json.dumps(message_to_dict(message)))

message = AIMessage(content='In mathematics, cosine (abbreviated as cos) is a trigonometric function that relates the angle of a right triangle to the ratio of the length of the adjacent side to the hypotenuse. It is defined as the ratio of the length of the side adjacent to an acute angle in a right triangle to the length of the hypotenuse. The cosine function is commonly used in geometry, physics, and engineering to solve problems related to angles and triangles.')
redis_client.lpush(prefix+session_id, json.dumps(message_to_dict(message)))

# 获取历史:
_items = redis_client.lrange(prefix+session_id, 0, -1)
items = [json.loads(m.decode("utf-8")) for m in _items[::-1]]
messages = messages_from_dict(items)
```

备注: 可以通过如下方式去掉对 redis 的依赖:

```python
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
message_history = ChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: message_history,
    input_messages_key="question",
    history_messages_key="history",
)
# invoke 之后可以通过这两种方案来看现有的历史
chain_with_history.get_session_history("foobar").messages  # 通过 chain_with_history 拿到历史的做法
message_history.messages
```

## Cookbook

### ConversationalRetrievalChain (TODO)

## Concept

### `Runnable` vs `Chain`

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# llm, prompt, output_parser 都是 Runnable, 是最小单元
llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])
output_parser = StrOutputParser()

# chain 是 Chain, 其本身也是符合(继承) Runnable 协议的
chain = prompt | llm | output_parser
```

`Chain` 分为旧式的和新式的, TODO: 把这个例子的省略号完善

```python
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# 旧式 Chain: 采用继承的方式串接各个 runnable 组件, 继承关系如下
# StuffDocumentsChain -> BaseCombineDocumentsChain -> Chain -> RunnableSerializable -> (Serializable, Runnable)
chain = StuffDocumentsChain(...)

# 新式 Chain: 直接用 | 以及 RunnableParallel, RunnablePassThrough 等串接各个 runnable 组件, 内部实现大致如下
# (RunnablePassthrough.assign(**{DOCUMENTS_KEY: format_docs}).with_config(...) | prompt | llm | _output_parser).with_config(...)
chain = create_stuff_documents_chain(...)
```

### Memory

官方文档中提到大部分出于 Beta 状态, 不是很理解: [https://python.langchain.com/docs/modules/memory/](https://python.langchain.com/docs/modules/memory/)

> Most of memory-related functionality in LangChain is marked as beta. This is for two reasons:
> - Most functionality (with some exceptions, see below) are not production ready
> - Most functionality (with some exceptions, see below) work with Legacy chains, not the newer LCEL syntax.

- 新类似乎是这里: `BaseChatMessageHistory`, 见[这里](https://python.langchain.com/docs/expression_language/how_to/message_history) 和 [这里](https://python.langchain.com/docs/integrations/memory)
- 旧类似乎是这里: `ConversationBufferMemory`, `ConversationBufferWindowMemory`, 见[这里](https://python.langchain.com/docs/modules/memory/types/)


## Code

### Message

```python
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
# 以下都继承自 BaseMessage
from langchain_core.messages.ai import AIMessage, AIMessageChunk
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk  # 可以指定角色为任意字符串, 用的不多
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk  # 代表用户自行调用函数后得到的结果
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.system import SystemMessage, SystemMessageChunk
from langchain_core.messages.tool import ToolMessage, ToolMessageChunk  # 代表用户自行调用工具后得到的结果
```

### Runnable (ICEL) (TODO)

本节内容作为 [https://python.langchain.com/docs/expression_language/](https://python.langchain.com/docs/expression_language/) 的补充与解释

```python
# langchain_openai.chat_models.base.ChatOpenAI: public class
class ChatOpenAI(BaseChatModel):
    def _stream(...): ...
    def _generate(...): ...
    async def _astream(...): ...
    async def _agenerate(...): ...

class BaseChatModel(BaseLanguageModel[BaseMessage], ABC):
    # invoke, ainvoke, stream, astream, generate, agenerate, generate_prompt, agenerate_prompt

class BaseLanguageModel(RunnableSerializable[LanguageModelInput, LanguageModelOutputVar], ABC):
    # @abstractmethod: generate_prompt, agenerate_prompt

class RunnableSerializable(Serializable, Runnable[Input, Output]):
    # configurable_fields, configurable_alternatives

class Runnable:
    # 包含对外接口: invoke, ainvoke, stream, astream, batch, abatch, astream_log, astream_events
    # 参考文档: https://python.langchain.com/docs/expression_language/interface
```

`langchain_core` 目录下所有以 `"Runnable"` 开头的类

```
Runnable
RunnableSerializable
RunnableSequence
RunnableParallel
RunnableLambda
RunnableEachBase
RunnableEach
RunnableBindingBase
RunnableBinding

RunnableBranch

RunnableConfig  # 仅仅类似一个字典, 不是 runnable
RunnableConfigurableFields
RunnableConfigurableAlternatives

RunnableWithFallbacks

RunnableWithMessageHistory

RunnablePassthrough
RunnableAssign
RunnablePick

RunnableRetry

RunnableAgent
RunnableMultiActionAgent
```

<table>
<tr>
    <th>Runnable方法名</th>
    <th>返回类型(粗略)</th>
    <th>说明</th>
    <th>文档链接</th>
</tr>
<tr>
    <td>assign</td>
    <td>RunnableSerializable: (self | RunnableAssign)</td>
    <td>添加字段</td>
    <td>https://python.langchain.com/docs/expression_language/how_to/passthrough</td>
</tr>
<tr>
    <td>pipe</td>
    <td>RunnableSequence</td>
    <td></td>
    <td></td>
</tr>
<tr>
    <td>pick</td>
    <td>RunnableSerializable: (self | RunnablePick)</td>
    <td></td>
    <td></td>
</tr>
<tr>
    <td>bind</td>
    <td>RunnableBinding</td>
    <td></td>
    <td>https://python.langchain.com/docs/expression_language/how_to/binding</td>
</tr>
<tr>
    <td>with_config, configurable_fields, configurable_alternatives</td>
    <td>RunnableBinding</td>
    <td></td>
    <td>https://python.langchain.com/docs/expression_language/how_to/configure</td>
</tr>
<tr>
    <td>with_listeners</td>
    <td>RunnableBinding</td>
    <td>用于 RunnableWithMessageHistory</td>
    <td>https://python.langchain.com/docs/expression_language/how_to/message_history</td>
</tr>
<tr>
    <td>with_types</td>
    <td>RunnableBinding</td>
    <td></td>
    <td></td>
</tr>
<tr>
    <td>with_retry</td>
    <td>RunnableRetry</td>
    <td></td>
    <td></td>
</tr>
<tr>
    <td>map</td>
    <td>RunnableEach</td>
    <td></td>
    <td></td>
</tr>
<tr>
    <td>with_fallbacks</td>
    <td>RunnableWithFallbacks</td>
    <td></td>
    <td></td>
</tr>
<table>

`RunnableBranch` 看起来是需要直接使用其构造函数的, 参考: [https://python.langchain.com/docs/expression_language/how_to/routing](https://python.langchain.com/docs/expression_language/how_to/routing)

#### Runnable.invoke

```python
class Runnable:
    @abstractmethod
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output: ...
```

#### Runnable.with_config, configurable_fields, configurable_alternatives

参考: [https://python.langchain.com/docs/expression_language/how_to/configure](https://python.langchain.com/docs/expression_language/how_to/configure)

- `configurable_fields` 用于将 Runnable 的一些参数暴露给 `with_config` 或 `invoke` 调节
- `configurable_alternatives` 用于将 Runnable 串接为 Chain 后, 将某个 Runnable 整个进行替换, 暴露给 `with_config` 或 `invoke` 调节

**模板**

```python
# 必须先 configurable_fields 或 configurable_alternatives, 再使用 with_config
runnable = runnable.configurable_fields(...)  # 不同的 runnable 有不同的可设置 key, key 不能乱配
runnable = runnbale.configurable_alternatives(...)  # 配置可以替换整个runnable为其他runnable
runnable.with_config(configurable={"foo": 0.9})
```

**例子**

```python
from langchain.prompts import PromptTemplate
runnable = PromptTemplate.from_template("This is history: {history}, This is content: {content}")
runnable = runnable.configurable_fields(template=ConfigurableField(id="custom_template"))  # "template" in runnable.__fields__.keys()
runnable.with_config(configurable={"custom_template": "{history} {content}"}).invoke({"history": "1", "content": "2"})
```

三种在 `invoke` 时设置 config 的做法:

```python
prompt = PromptTemplate.from_template("Tell me a joke about {topic}").configurable_alternatives(
    ConfigurableField(id="prompt_choice"),
    default_key="joke",
    poem=PromptTemplate.from_template("Write a short poem about {topic}"),
)

prompt.with_config(config={"configurable": {"prompt_choice": "poem"}}).invoke({"topic": "book"})
prompt.with_config(configurable={"prompt_choice": "poem"}).invoke({"topic": "book"})
prompt.invoke({"topic": "book"}, config={"configurable": {"prompt_choice": "poem"}})
# prompt.invoke({"topic": "book"}, configurable= {"prompt_choice": "poem"})  # ERROR: 不能用 configurable={"prompt_choice": "poem"}
```

**一个 Chain 的例子**

也可以混合使用 `configurable_fields` 和 `configurable_alternatives`, 参考上面的文档即可

```python
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
runnable = PromptTemplate.from_template("This is history: {history}, This is content: {content}")
runnable2 = PromptTemplate.from_template("context: {context}")
params = {"history": "1", "content": "2"}

# 必须对每个 runnable 组件设置 configurable_fields, 并且 id 必须不同, 否则会冲突
runnable = runnable.configurable_fields(template=ConfigurableField(id="custom_template1"))
runnable2 = runnable2.configurable_fields(template=ConfigurableField(id="custom_template2"))

chain = (runnable | (lambda x: {"context": x.text}) | runnable2)

# 注意对 chain 本身设置 configurable_fields 只能设置 ['name', 'first', 'middle', 'last']
# chain.configurable_fields(template=ConfigurableField(id="custom_template"))  # ERROR! only support: ['name', 'first', 'middle', 'last']
chain.with_config(
    configurable={
        "custom_template1": "custom_template1 {history} {content}",
        "custom_template2": "custom_template2 {context}"
    }
).invoke(params)
```

#### RunnableConfig

继承自字典类型, TODO: 具体包含的 key

#### RunnableLambda

参考文档 [https://python.langchain.com/docs/expression_language/how_to/functions](https://python.langchain.com/docs/expression_language/how_to/functions)

```python
# RunnableLambda 实现的大略逻辑如下
class RunnableLambda(Runnable[Input, Output]):
    def __init__(
        self,
        func: Union[
                Callable[[Input], Output],
                Callable[[Input], Iterator[Output]],
                Callable[[Input, RunnableConfig], Output],
                Callable[[Input, CallbackManagerForChainRun], Output],
                Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output],
        ],
        afunc = None,
        name: Optional[str] = None,
    ) -> None:
        ...

    def _invoke(
        self,
        input: Input,
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Output:
        output = call_func_with_variable_args(
            self.func, input, config, run_manager, **kwargs
        )
        return cast(Output, output)
```

**例子**

```python
from langchain_core.runnables import RunnableLambda, RunnableConfig

def func1(x):
    return x

def func2(x, config: RunnableConfig):
    return x["num"] + config["configurable"]["total"]

RunnableLambda(func1).invoke({"num": 1})  # {'num': 1}
RunnableLambda(func2).invoke({"num": 1}, config={"configurable": {"total": 100}})  # 101
RunnableLambda(func2).invoke({"num": 1}, config={"configurable": {"total": 100}, "foo": 2})  # 也 OK: 101
```

备注: `RunnableLambda` 没有继承自 `RunnableSerializable` 因此没有 `configurable_fields`, `configurable_alternatives` 方法, 并且 `with_config` 方法也不能设置 `configurable`

### Memory

以这个例子为例: [https://python.langchain.com/docs/modules/memory/types/buffer_window](https://python.langchain.com/docs/modules/memory/types/buffer_window)

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
conversation_with_summary = ConversationChain(
    llm=OpenAI(temperature=0),
    # We set a low k=2, to only keep the last 2 interactions in memory
    memory=ConversationBufferWindowMemory(k=2),
    verbose=True
)

conversation_with_summary.predict(input="Hi, what's up?")
conversation_with_summary.predict(input="What's their issues?")
conversation_with_summary.predict(input="Is it going well?")
conversation_with_summary.predict(input="What's the solution?")
```

这里稍微展开一下 `ConversationChain.predict` 的逻辑: 首先 `ConversationChain` 应该是属于所谓的 lagacy chains, `predict` 方法应该也会在 langchain 的后续版本完全被 `invoke` 方法取代, 而 `invoke` 方法的主要执行流程为:

```python
# ConversationChain -> LLMChain -> Chain
# 以下均为大体流程
input = {"input": "Hi, what's up?"}
def invoke(self, input):
    inputs = self.prep_inputs(input)  # Chain 中定义
    outputs = self._call(inputs)      # LLMChain 中定义
    final_outputs: Dict[str, Any] = self.prep_outputs(inputs, outputs)  # Chain 中定义
    return final_outputs

def prep_inputs(self, inputs)
    if self.memory is not None:
        external_context = self.memory.load_memory_variables(inputs)
        inputs = dict(inputs, **external_context)
    return inputs

def prep_outputs(self, inputs, outputs):
    if self.memory is not None:
        self.memory.save_context(inputs, outputs)
    return outputs

def _call(inputs):
    prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
    self.llm.xxx(...)
    return self.create_outputs(...)
```

从上面可以看出, 只需要关注 `load_memory_variables` 和 `save_context` 方法即可 (因为调用来自于 `Chain` 这个父类)


**疑问**: 这里这个例子 [https://python.langchain.com/docs/expression_language/how_to/message_history](https://python.langchain.com/docs/expression_language/how_to/message_history) 是怎么回事 `RunnableWithMessageHistory`? 似乎它与本节的完全独立, 不在一个体系里: 本节的 memory 需要实现 `load_memory_variables` 和 `save_context` 方法, 而 `RunnableWithMessageHistory` 里的 memory 要求实现 `messages` 属性以及 `add_message` 方法


### callback (tracing, visibility)

开箱即用的: [https://python.langchain.com/docs/integrations/callbacks](https://python.langchain.com/docs/integrations/callbacks)

- Argilla: feedback 标注, 应该可完全私有化部署, [langchain文档](https://python.langchain.com/docs/integrations/callbacks/argilla), [官网](https://argilla.io/)
- Comet: tracing, 似乎不可私有部署, [langchain文档](https://python.langchain.com/docs/integrations/callbacks/comet_tracing), [官网](https://www.comet.com/site/)
- Confident
- Context
- Infino
- Label Studio
- LLMonitor
- PromptLayer
- SageMaker Tracking
- Streamlit
- Trubrics

自定义: [https://python.langchain.com/docs/modules/callbacks/](https://python.langchain.com/docs/modules/callbacks/)

## LangSmith

可以脱离 Langchain 使用, 但似乎必须借助 LangSmith 服务, 不能本地部署.

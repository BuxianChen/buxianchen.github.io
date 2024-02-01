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

## Cookbook

### ConversationalRetrievalChain (TODO)

## Code

**Runnable**

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

## LangSmith

可以脱离 Langchain 使用, 但似乎必须借助 LangSmith 服务, 不能本地部署.

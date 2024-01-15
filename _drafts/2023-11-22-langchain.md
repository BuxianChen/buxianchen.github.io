---
layout: post
title: "(P1) Langchain"
date: 2023-11-22 11:10:04 +0800
---

## 涉及内容

`langchain >= 0.1.0`: [langchain blog (2024/01/08)](https://blog.langchain.dev/langchain-v0-1-0/)


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


例子1

例子2 (模型自定义)

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

## Cookbook
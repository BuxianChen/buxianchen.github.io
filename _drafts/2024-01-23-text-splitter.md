---
layout: post
title: "(P0) Text Splitter for LLMs"
date: 2024-01-23 17:10:04 +0800
labels: [llm]
---

## 动机、参考资料、涉及内容

LLM 在应用时通常会使用到 RAG 技术, 其核心组件是文档检索模块, 以向量检索为例, 需要使用用户问检索文档库, 检索质量极为重要 (尤其是召回率, 如果答案不在召回结果里, LLM 的回复结果是可预想地差). 通常的设定是我们手头上有一个开源的 Embedding 模型, 并且也不打算对它做微调, 因此提升检索质量只能从文档切分下手. `langchain`, `llama_index` 等 LLM 应用框架里内置了一些文档切分方法, 虽然在特定场景和数据场景下, 可能需要自己编写切分文档脚本, 但这些内置的切分方法也是一个不错的默认选项, 参考它们也有利于帮助写出特定场景下的切分文档脚本.

## `llama_index.node_parser.text.sentence.SentenceSplitter`

**分析入口**

```python
# 准备工作
# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'
from llama_index import SimpleDirectoryReader
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
idx = documents[0].text[:2000].rfind(".")
text = documents[0].text[:idx+1]
documents[0].set_content(text)
print(text)

from llama_index.node_parser.text.sentence import SentenceSplitter
from llama_index.schema import TextNode
from typing import List

splitter = SentenceSplitter.from_defaults(chunk_size=150, chunk_overlap=50)
text_nodes: List[TextNode] = splitter(documents)
```

**说明**

TODO: 通常会是在调用 `from_documents` 函数时被自动触发, `llama_index` 相关的内容, 而实现分析追求框架无关

**实现分析**

```python
import nltk
from typing import List

tokenizer = nltk.tokenize.PunktSentenceTokenizer()

def split(text: str) -> List[str]:
    spans = list(tokenizer.span_tokenize(text))
    sentences = []
    for i, span in enumerate(spans):
        start = span[0]
        if i < len(spans) - 1:
            end = spans[i + 1][0]
        else:
            end = len(text)
        sentences.append(text[start:end])

    return sentences
```
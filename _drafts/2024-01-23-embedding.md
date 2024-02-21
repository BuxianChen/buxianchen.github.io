---
layout: post
title: "(P1) Embedding Models"
date: 2024-01-23 11:10:04 +0800
labels: [llm]
---

## 动机、参考资料、涉及内容

一些可用的 Embedding 模型与 ReRanking 模型使用及注意事项

leaderboard: [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

## 可用模型

### OpenAI

TL;DR: 目前只需要关注一个模型 `text-embedding-ada-002` 即可, 用例参考以下

- 官方使用手册[https://platform.openai.com/docs/guides/embeddings/use-cases](https://platform.openai.com/docs/guides/embeddings/use-cases)

-----

第一代的 16 个模型已在 2024/01/04 下线

参考:
- [https://platform.openai.com/docs/deprecations/first-generation-text-embedding-models](https://platform.openai.com/docs/deprecations/first-generation-text-embedding-models)
- [https://openai.com/blog/introducing-text-and-code-embeddings](https://openai.com/blog/introducing-text-and-code-embeddings)
- [https://platform.openai.com/docs/guides/embeddings/embedding-models](https://platform.openai.com/docs/guides/embeddings/embedding-models)
- tokenizer: [https://github.com/openai/tiktoken/blob/main/tiktoken/model.py](https://github.com/openai/tiktoken/blob/main/tiktoken/model.py)

```
# V1 的 16 个模型

text-similarity-ada-001   # $0.004 / 1K tokens
text-search-ada-doc-001
text-search-ada-query-001
code-search-ada-code-001
code-search-ada-text-001

text-similarity-babbage-001   # $0.005 / 1K tokens
text-search-babbage-doc-001
text-search-babbage-query-001
code-search-babbage-code-001
code-search-babbage-text-001

text-similarity-curie-001    # 0.02 / 1K tokens
text-search-curie-doc-001
text-search-curie-query-001

text-search-davinci-doc-001   # 0.2 / 1K tokens
text-search-davinci-query-001

# V2
text-embedding-ada-002        # 0.0004 / 1K tokens
```

- 文档检索一般是要使用 `text-search-*-doc-001` 做文档嵌入, 使用 `text-search-*-query-001` 做查询文档的嵌入, 以此实现检索功能
- 聚类任务一般是使用 `text-similarity-*-001` 做文档嵌入, 计算文档间的相似度

注: tokenizer 指的是 [tiktoken](https://github.com/openai/tiktoken/blob/main/tiktoken/model.py)

- **gpt2**: gpt-2
- r50k_base: 使用此 tokenizer 的模型似乎全部弃用了
- p50k_base: 使用此 tokenizer 的模型似乎全部弃用了
- p50k_edit: 使用此 tokenizer 的模型似乎全部弃用了
- **cl100k_base**: gpt3.5, gpt4, text-embedding-ada-002

<table>
<tr>
    <th>model</th>
    <th>description</th>
    <th>use cases</th>
    <th>tokenizer</th>
    <th>max tokens</th>
    <th>备注</th>
</tr>
<tr>
    <td>text-similarity-{ada, babbage, curie, davinci}-001</td>
    <td>Text similarity: Captures semantic similarity between pieces of text.</td>
    <td>Clustering, regression, anomaly detection, visualization</td>
    <td>GPT-2/GPT-3</td>
    <td>2046</td>
    <td>归一化余弦相似度</td>
</tr>
<tr>
    <td>text-search-{ada, babbage, curie, davinci}-{query, doc}-001</td>
    <td>Text search: Semantic information retrieval over documents.</td>
    <td>Search, context relevance, information retrieval</td>
    <td>GPT-2/GPT-3</td>
    <td>2046</td>
</tr>
<tr>
    <td>code-search-{ada, babbage}-{code, text}-001</td>
    <td>Code search: Find relevant code with a query in natural language.</td>
    <td>Code search and relevance</td>
    <td>GPT-2/GPT-3</td>
    <td>2046</td>
</tr>
<tr>
    <td>text-embedding-ada-002</td>
    <td>ALL</td>
    <td>ALL</td>
    <td>cl100k_base</td>
    <td>8191</td>
    <td>余弦相似度, embedding 接口返回结果本身已做归一化</td>
</tr>
</table>


### FlagEmbedding

适用于中文的最新模型如下, 完整列表参考[README.md](https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list)

<table>
<tr>
    <th>模型</th>
    <th>语言</th>
    <th>描述</th>
    <th>tokenizer</th>
    <th>max tokens</th>
    <th>备注</th>
</tr>
<tr>
    <td>BAAI/bge-large-zh-v1.5, BAAI/bge-base-zh-v1.5, BAAI/bge-small-zh-v1.5</td>
    <td>中文</td>
    <td>用于文档检索(query与text-embedding), 文档相似度(text-embedding)</td>
    <td>三个模型采用一个相同的 wordpiece based tokenizer</td>
    <td>512 - 2 = 510</td>
    <td>embedding 需要取最后一个隐层的 CLS token, 不能用所有 token 的平均</td>
</tr>
<tr>
    <td>BAAI/bge-reranker-large, BAAI/bge-reranker-base</td>
    <td>中文/英文</td>
    <td>对检索召回结果进行 rerank 的模型</td>
    <td>两个模型采用一个相同的 sentencepiece based tokenizer</td>
    <td>两个句子拼接不超过 512 - 4 = 508</td>
    <td></td>
</tr>
</table>

**用法**

embedding 模型:

使用代码参考 [https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding)

注意点:

- 检索时, 相似度的排序才是重要的, 卡阈值不靠谱, 如果需要卡阈值, 官方建议卡 0.8-0.9
- 关于是否加前缀, `bge-*-zh-1.5` 模型一般可以都不加前缀, 但用短文本查询长文本的情况下还是推荐在短文本前加前缀. 无论在哪种情况下, 文档库都不需要加前缀
- Q-A 检索时: Q 需要加上前缀 `为这个句子生成表示以用于检索相关文章：`, 文档库不需要加前缀
- Q-Q 检索时: Q 与 Q 都不需要加前缀

rerank 模型:

使用代码参考 [https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)

注意点:

- 输出结果没有做归一化, 即相似度值会是一个不限定取值区间的浮点数
- 注意文本对不要太长, 会发生截断 (两句话都有可能发生截断, 可参考 huggingface 关于截断的说明)


**tokenizer**

```python
from transformers import AutoTokenizer
rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
embedding_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh-v1.5")
text = "a"
text_pair = ["a", "b"]

def debug_tokenizer(tokenizer, text, text_pair):
    input_ids = tokenizer(text)['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(f"text: {text}, token_ids: {input_ids}, tokens: {tokens}")
    
    input_ids = tokenizer([text_pair])['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(f"text pair: {text_pair}, token_ids: {input_ids}, tokens: {tokens}")

print("embedding tokenizer:")
debug_tokenizer(embedding_tokenizer, text, text_pair)
print("\nreranking tokenizer:")
debug_tokenizer(rerank_tokenizer, text, text_pair)
```

输出 (注意 embedding 模型我们一般只关注单句话, reranking 模型我们只关注句子对)

```
embedding tokenizer:
text: a, token_ids: [101, 143, 102], tokens: ['[CLS]', 'a', '[SEP]']
text pair: ['a', 'b'], token_ids: [101, 143, 102, 144, 102], tokens: ['[CLS]', 'a', '[SEP]', 'b', '[SEP]']

reranking tokenizer:
text: a, token_ids: [0, 10, 2], tokens: ['<s>', '▁a', '</s>']
text pair: ['a', 'b'], token_ids: [0, 10, 2, 2, 876, 2], tokens: ['<s>', '▁a', '</s>', '</s>', '▁b', '</s>']
```
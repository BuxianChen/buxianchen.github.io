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

### OpenAI (TODO: 003 模型)

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
    <td>BAAI/bge-m3</td>
    <td>多语言(支持中英文)</td>
    <td>query 与 document 都无需加前缀</td>
    <td>TODO</td>
    <td>8192 - ?</td>
    <td>可以得到 dense, sparse, colbert 向量</td>
</tr>
<tr>
    <td>BAAI/bge-large-zh-v1.5, BAAI/bge-base-zh-v1.5, BAAI/bge-small-zh-v1.5</td>
    <td>中文</td>
    <td>用于文档检索(query-text), 文档相似度(text-text)</td>
    <td>三个模型采用一个相同的 wordpiece based tokenizer</td>
    <td>512 - 2 = 510</td>
    <td>embedding 需要取最后一个隐层的 CLS token, 不能用所有 token 的平均</td>
</tr>
<tr>
    <td>BAAI/bge-reranker-v2-m3</td>
    <td>中文/英文</td>
    <td>对检索召回结果进行 rerank 的模型</td>
    <td>TODO</td>
    <td>两个句子拼接不超过 8192 - ?</td>
    <td></td>
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

#### bge-{small,base,large}-zh-1.5

使用代码参考 [https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding)

注意点:

- 检索时, 相似度的排序才是重要的, 卡阈值不靠谱, 如果需要卡阈值, 官方建议卡 0.8-0.9
- 关于是否加前缀, `bge-*-zh-1.5` 模型一般可以都不加前缀, 但用短文本查询长文本的情况下还是推荐在短文本前加前缀. 无论在哪种情况下, 文档库都不需要加前缀
- Q-A 检索时: Q 需要加上前缀 `为这个句子生成表示以用于检索相关文章：`, 文档库不需要加前缀
- Q-Q 检索时: Q 与 Q 都不需要加前缀

tokenizer:

```python
from transformers import AutoTokenizer
text = "a"
text_pair = ["a", "b"]

def debug_tokenizer(tokenizer, text, text_pair):
    input_ids = tokenizer(text)['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(f"text: {text}, token_ids: {input_ids}, tokens: {tokens}")
    
    input_ids = tokenizer([text_pair])['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(f"text pair: {text_pair}, token_ids: {input_ids}, tokens: {tokens}")

embedding_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh-v1.5")
print("embedding tokenizer:")
debug_tokenizer(embedding_tokenizer, text, text_pair)
```

输出: 注意 embedding 模型我们一般只关注单句话

```
embedding tokenizer:
text: a, token_ids: [101, 143, 102], tokens: ['[CLS]', 'a', '[SEP]']
text pair: ['a', 'b'], token_ids: [101, 143, 102, 144, 102], tokens: ['[CLS]', 'a', '[SEP]', 'b', '[SEP]']
```

#### bge-rerank-{large,base}

使用代码参考 [https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)

注意点:

- 输出结果没有做归一化, 即相似度值会是一个不限定取值区间的浮点数
- 注意文本对不要太长, 会发生截断 (两句话都有可能发生截断, 可参考 huggingface 关于截断的说明)

tokenizer:

```python
from transformers import AutoTokenizer
rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")

print("\nreranking tokenizer:")
debug_tokenizer(rerank_tokenizer, text, text_pair)
```

输出: 注意 reranking 模型我们只关注句子对

```
reranking tokenizer:
text: a, token_ids: [0, 10, 2], tokens: ['<s>', '▁a', '</s>']
text pair: ['a', 'b'], token_ids: [0, 10, 2, 2, 876, 2], tokens: ['<s>', '▁a', '</s>', '</s>', '▁b', '</s>']
```

#### bge-m3

官方的推荐的检索方案: 使用 bge-m3 的 dense + sparse 进行混合检索, 然后使用 bge-reranker-v2. 向量数据库层面支持混合检索的有 milvus>=2.4, Vespa.

bge-m3 得名于多模式 (Multi-Functionality), 多语种 (Multi-Linguality), 多颗粒度 (Multi-Granularity)

- Multi-Functionality: 即生成下面的 dense, sparse, colbert 向量. dense 向量的维度是 1024; sparse 向量可以看作是维数等于词表长度的稀疏向量, 或者看作是词袋模型, 只有句子中包含的 token 对应的维度不为 0; colbert 则保留了句子中每个 token 的向量表示, 因此可被用于 multi-vector retrieval, 也就是一个句子有多个向量表示 (句子的 token 数量个向量)
- Multi-Linguality: 多语种, 尤其是中英文
- Multi-Granularity: 实际上就是指句子的最大 token 数由 bge-1.5 的 512 提升至了 8192

计算 embedding 的逻辑如下:

```python
# B: batch size, L: 序列长度, C: 隐层输出, V: vocab_size
B, L, C, V = 2, 4, 3, 5
# hidden_state: (B, L, C)

################ dense_embedding: (B, C) ######################
# 使用 CLS token
dense_embedding = hidden_state[:, 0]  # (B, C)
dense_embedding = torch.nn.functional.normalize(dense_embedding, dim=-1)  # 默认进行归一化

################ sparse_embedding: (B, V) ######################
input_ids = torch.tensor([[1, 1, 2, 2], [1, 3, 3, 0]])
sparse_embedding = torch.zeros(B, L, V)
sparse_linear = torch.nn.Linear(C, 1)
token_weight = sparse_linear(hidden_state)
# token_weight: (B, L, 1)
# [
#     [[0.6152], [0.6736], [0.0937], [0.3646]],
#     [[0.5414], [0.3734], [0.0577], [0.0790]]
# ]

sparse_embedding = torch.scatter(sparse_embedding, dim=-1, index=input_ids.unsqueeze(-1), src=token_weights)
# sparse_embedding: (B, L, V)
# [
#     [[0, 0.6152, 0, 0, 0],
#     [0, 0.6736, 0, 0, 0],
#     [0, 0, 0.0937, 0, 0],
#     [0, 0, 0.3646, 0, 0]],

#     [[0, 0.5414, 0, 0, 0],
#     [0, 0, 0, 0.3734, 0],
#     [0, 0, 0, 0.0577, 0],
#     [0.0790, 0, 0, 0, 0]],
# ]

sparse_embedding = torch.max(sparse_embedding, dim=1).values  # (B, V)
# sparse_embedding: [[0, 0.6736, 0.3646, 0, 0], [0.0790, 0.5414, 0, 0.3734, 0]]
unused_tokens = [0, 4]  # cls_token_id, eos_token_id, pad_token_id, unk_token_id
sparse_embedding[:, unused_tokens] *= 0  # 注意: 没有归一化过程, 特殊 token 置为 0

##### colbert_embedding: [(L1-1, C), (L2-1, C), ..., (LB-1, C)] ####
# L1, ..., LB 分别为不加 padding 的长度, 减 1 是为了算相似度时不考虑 CLS token
colbert_linear = torch.nn.linear(C, C)
colbert_embedding = colbert_linear(hidden_state)
# mask: (B, L), padding 的位置为 0, 其余位置为 1
colbert_embedding = colbert_embedding[:, 1:] * mask[:, 1:][:, :, None]  # colbert_embedding: (B, L-1, C), padding 位置全部置为了 0
colbert_embedding = torch.nn.functional.normalize(colbert_embedding, dim=-1)  # 默认进行归一化
colbert_embedding = [e[:m.sum()-1] for e, m in zip(colbert_embedding.cpu().numpy(), mask.cpu.numpy())]
```


计算相似度的逻辑如下:

```python
######### dense: 余弦相似度 ###############
# query_dense: (C,), doc_dense: (C,)
dense_score = query_dense @ doc_dense


############ sparse: 内积 ################
# query_sparse: (V,), doc_sparse: (V,)
sparse_score = query_sparse @ doc_sparse


############## colbert ###################
# query_colbert: (Lq-1, C), doc_colbert: (Ld-1, C)
scores = query_colbert @ doc_colbert.T   # scores: (Lq-1, Ld-1)
scores, _ = scores.max(-1)               # scores: (Lq-1)
colbert_score = torch.mean(scores)       # 注意这个相似度不是对称的 colbert_score(q, d) != colbert_score(d, q)


############ 混合分数 ####################
# 官方代码中只包含两种混合方式: sparse+dense 和 colbert+sparse+dense
# demo 中所配权重为: {"dense": 0.4, "sparse": 0.2, "colbert": 0.4}, 但这个权重不确定是否为最优
sparse_dense = (0.4 * dense + 0.2 * sparse) / 0.6
colbert_sparse_dense = (0.4 * dense + 0.2 * sparse + 0.4 * colbert)
```

#### bge-visualized-m3

支持多语言; 图片, 文字, 一张图片+一段文字. 其主要原理是将图片打成 patch (ViT 的思路), 然后将文字和图片分别 embedding 后, 以 `text-cls-token, image-patches, text-tokens` 拼接, 然后继续堆几层 self-attention, 最后取 `text-cls-token` 的 embedding 作为最终的 embedding.

注意: `bge-visualized-m3` 模型实际上只是文字模型, 需要搭配视觉模型 `EVA02-CLIP-L-14` 来使用

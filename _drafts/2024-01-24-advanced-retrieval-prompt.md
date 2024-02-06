---
layout: post
title: "(LTS) Advanced LLM Prompts & Retrieval & Agents"
date: 2024-01-24 11:10:04 +0800
labels: [llm]
---

## 动机、参考资料、涉及内容

一些关于大模型 Prompt, RAG, Agent (工具使用) 的用例记录

## 资源汇总

- langchainhub: [https://smith.langchain.com/hub](https://smith.langchain.com/hub)
- langchain/template: 一些使用 langchain 的 example, [https://github.com/langchain-ai/langchain/tree/master/templates](https://github.com/langchain-ai/langchain/tree/master/templates)

## reciprocal rerank fusion: 多个检索结果整合

- llama_index: [https://docs.llamaindex.ai/en/stable/examples/retrievers/reciprocal_rerank_fusion.html](https://docs.llamaindex.ai/en/stable/examples/retrievers/reciprocal_rerank_fusion.html)
- 原始实现: [https://github.com/Raudaschl/rag-fusion](https://github.com/Raudaschl/rag-fusion)

伪代码如下:

```python
query = "how can I became a doctor"
prompt = "Generate 4 search queries related to query: {query}"
generated_queries = llm(prompt).split("\n")
all_documents = [f"doc {i}" for i in range(10)]

def search(generated_query, all_documents):
    docs = random.choices(all_documents, 3)
    scores = [random.random() for i in range(3)]
    return {doc: score for doc, score in zip(doc, score)}

def reciprocal_rank_fusion(search_results_dict, k=60):
    fused_scores = {}
    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            previous_score = fused_scores[doc]
            fused_scores[doc] += 1 / (rank + k)
    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:4]}  # 假设只留 4 个结果
    return reranked_results

search_results_dict = {}
for generated_query in generated_queries:
    search_results_dict[generated_query] = search(generated_query, all_documents)
reranked_results = reciprocal_rank_fusion(search_results_dict)

prompt = "Based on the context:\n{context}\nAnswer the question {query}".format(context="\n".join(reranked_results.keys()), query=query)
answer = llm(prompt)
```

## Parent Document Retriever

参考: [https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever)

本质上是先按大粒度切分文档 (parent), 然后再对这些大块的文档继续切分并做 embedding (child), 并保留 child to parent 的映射, 检索时向量相似度在 child 上做, 但返回的文档是 parent 的. 以下为简易实现供参考

```python
from functools import partial
from typing import Dict, List

def demo_splitter(doc, chunk_size):
    n = (len(doc) - 1) // chunk_size + 1
    return [doc[i*chunk_size: (i+1)*chunk_size] for i in range(n)]

docs = ["1"*10000, "2"*10000]
parent_splitter = partial(chunk_size=1000)
child_splitter = partial(chunk_size=200)

parent_docstore: Dict[str, str] = {}  # parent-id -> parent chunk text
child_idx_to_parent: Dict[str] = {}  # child-idx -> parent-id
child_vectors: List[List[float]] = []

n = 0
for i, doc in enumerate(docs):
    parent_docs = parent_splitter(doc)
    for j, parent_doc in enumerate(parent_docs):
        parent_key = f"parent:{i}:{j}"
        parent_docstore[] = parrent_doc
        child_docs = child_splitter(parrent_doc)
        for k, child_doc in enumerate(child_docs):
            child_idx_to_parent[n] = parent_key
            # embedding_model: Callable[str, List[float]]
            child_vectors.append(embedding_model(child_doc))
            n += 1

def search(query: str, n=4) -> List[str]:
    emb = embedding_model(query)
    idxes: List[int] = get_similar(emb, child_vectors, n)
    
    parent_keys = set()
    for idx in idxes:
        parent_keys.add(child_idx_to_parent[idx])

    parent_docs = []
    for parent_key in parent_keys:
        parent_docs.append(parent_docstore[parent_key])
    
    return parent_docs

docs = search("333")
```

## HyDE

对于问题 `query`, 先让大模型生成答案, 然后根据答案做搜索 (大致是 Answer-Answer 匹配?), 然后再进行 RAG 让大模型给出答案. 核心在于期望能提升检索模型的性能

参考 [https://github.com/langchain-ai/langchain/tree/master/templates/hyde](https://github.com/langchain-ai/langchain/tree/master/templates/hyde) 或原始论文 [https://arxiv.org/abs/2212.10496](https://arxiv.org/abs/2212.10496), 用 Langchain 来实现十分简洁:

```python
# hyde_prompt
hyde_prompt = """Please write a passage to answer the question 
Question: {question}
Passage:"""

# RAG prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI()

# Query transformation chain
# This transforms the query into the hypothetical document
hyde_chain = hyde_prompt | model | StrOutputParser()

# RAG chain
chain = (
    RunnableParallel(
        {
            # Generate a hypothetical document and then pass it to the retriever
            "context": hyde_chain | retriever,
            "question": lambda x: x["question"],
        }
    )
    | prompt
    | model
    | StrOutputParser()
)
```

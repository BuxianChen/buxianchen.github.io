---
layout: post
title: "(P1) Advanced LLM Prompts & Retrieval & Agents"
date: 2024-01-24 11:10:04 +0800
labels: [llm]
---

## 动机、参考资料、涉及内容

一些关于大模型 Prompt, RAG, Agent (工具使用) 的用例记录

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

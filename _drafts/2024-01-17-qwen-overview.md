---
layout: post
title: "(P0) 通义千问-Overviw"
date: 2024-01-17 11:10:04 +0800
labels: [llm]
---

## 动机、参考资料、涉及内容

涉及内容

- 从宏观层面探索通义千问系列模型
- 前通义千问官方博客的阅读记录: [https://qwenlm.github.io/blog](https://qwenlm.github.io/blog)

不涉及内容

- 一些细节类的东西即使本篇博客提及, 也可能会在其他博客里重复

目标(TODO: 基本上要全部移除):

- 对话拼接格式: chatml
- react: 通义千问多轮的 ReAct 是怎么实现的
- function call / tool usage: 通义千问号称支持 function call, 因此探索下怎么进行实现的
- tokenizer

**QWen-1**

主要分析官方的这份代码: [openai_api.py](https://github.com/QwenLM/Qwen/blob/204c2c59f49cfa7461e8e02d5ad2f6b3d082f08c/openai_api.py)

相关资料: [QWen 官方 example](https://github.com/QwenLM/Qwen/blob/main/examples/)

**QWen-1.5**

- [https://github.com/modelscope/modelscope-agent](https://github.com/modelscope/modelscope-agent)




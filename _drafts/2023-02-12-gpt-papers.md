---
layout: post
title: "(P1) GPT-1，2，3，instructGPT，chatGPT"
date: 2023-02-12 13:31:04 +0800
labels: [limu, paper]
---

## 动机、参考资料、涉及内容

见下文

## GPT-1, GPT-2, GPT-3

- [视频讲解](https://www.bilibili.com/video/BV1AF411b7xQ)
- [GPT-1 论文: Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [GPT-2 论文: Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-3 论文: Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)

相关论文的时间线及参数量：

- transformer: Google (2017.6)
- GPT-1: OpenAI (2018.6), 110M, 预训练加微调
- bert: Google (2018.10), base: 110M, large: 340M, 预训练加微调
- GPT-2: OpenAI (2019.2), 117M/345M/762M/1542M, 用 prompt 做 zero-shot（但当时不叫 prompt，但这个做法之前有论文提出过）
- GPT-3: OpenAI (2020.5), large: 175 billion, 主推 few-shot 的 prompt 方式，论文中也被称作 in-context learning


**一些有意思的观点和记录**

- GPT 系列一直采用单向的标准的语言模型，预训练难度高于 BERT 的掩码预训练任务，所以模型和数据量比较小时（GPT-1 与 BERT-base 规模相当），效果不如 BERT，但数据量及模型大小加大之后，GPT 这种训练方式得到的预训练模型可能更强。
- Common Crawl：一个公开的爬虫项目（GPT-2 论文中有提及，但没有使用。GPT-3 论文使用了该数据集，但做了许多的数据清洗）

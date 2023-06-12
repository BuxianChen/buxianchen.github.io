---
layout: post
title: "(WIP) LLMs Survey"
date: 2023-05-04 14:31:04 +0800
labels: [paper]
---

## 动机、参考资料、涉及内容

动机

- 积累一些关于大模型的基本知识(常用术语、指标)，以及一些具体的数值

参考资料

- 2021.03综述：[Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/abs/2003.08241)
- 2023.04综述：[A Survey of Large Language Models](https://arxiv.org/abs/2303.18223)

涉及内容

- 大模型（例子）的训练数据集构成、模型大致架构及目标函数
- 一些可以参照的数值，例如某模型对多大数据量需要使用多少显存训练多久

不涉及内容

- 大模型结构的一些具体细节（暂时打算另起一篇博客结合 🤗 Transformers 中的具体实现进行介绍）

## 术语

**petaflop/s-day(pfs-day)**

- K: 10^3 或 2^10
- M: 10^6 或 2^20
- G: 10^9 或 2^30
- T: 10^12 或 2^40
- P: 10^15 或 2^50

petaflop/s-day(pfs-day) 是计算量的单位。1pfs-day为：假设计算机每秒计算 1 千万次（10^15或2^50），计算机计算 1 天（86400秒约10^5）的总计算量（约为10^20)。

参考资料：[OpenAI blog](https://openai.com/research/ai-and-compute)

## 硬件

[V100 计算能力](https://images.nvidia.cn/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf): 

## 大模型的 Scaling Law

- [OpenAI paper(2020.1) Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf)
- Chinchilla Scaling Law: [DeepMind paper(2022) Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf)
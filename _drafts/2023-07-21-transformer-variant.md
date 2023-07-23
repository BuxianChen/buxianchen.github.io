---
layout: post
title: "(WIP) Transformers 变体"
date: 2023-07-21 20:31:04 +0800
labels: [paper]
---

## 动机、参考资料、涉及内容【加上各处链接】

动机

最近发现了一些工作之间的引用关系, 例如:

- chatglm2 中使用了 Multi-Query Attention 技术, 实际测试下来对推理速度的提升还是十分显著的
- chatglm2, gptj, llama 等模型均使用了 RoPE 位置编码技术
- T5, llama 等模型均使用了 RMSNorm 而非 LayerNorm, 前者据说相对后者也能减少一些计算量
- RWKV 模型受到 AFT 的启发

虽然近期出现的许多大模型似乎还是基本沿用 Attention is all you need 这篇论文里的原始 Transformer 架构, 但近期也从各种消息渠道里发现了许多试图对 Transformer 进行改进甚至挑战的工作, 例如: RWKV, RetNet 等, 而 Transformer 从 2017 年被提出至今已有 6 年, 笔者相信 Transformer 架构并非完美, 而新的架构几乎必然是在以前的工作中受到启发, 因此本文将对 Transformer 的一些改进工作进行介绍. 本文只假定读者熟悉 Attention is all you need 论文里的原始架构.

本文的第二个动机是找个机会细细品读一下苏剑林大佬的一些博客, 算是做个笔记

参考资料

- 计划阅读苏剑林大佬的博客
  - 2020/07/04 [线性Attention的探索：Attention必须有个Softmax吗？](https://spaces.ac.cn/archives/7546)
  - 2020/08/07 [修改Transformer结构，设计一个更快更好的MLM模型 ](https://spaces.ac.cn/archives/7661)
  - 2020/12/01 [Performer：用随机投影将Attention的复杂度线性化](https://spaces.ac.cn/archives/7921)
  - 2020/12/04 [层次分解位置编码，让BERT可以处理超长文本](https://spaces.ac.cn/archives/7947)
  - 2021/02/03 [让研究人员绞尽脑汁的Transformer位置编码](https://spaces.ac.cn/archives/8130)
  - 2021/02/16 [Nyströmformer：基于矩阵分解的线性化Attention方案](https://spaces.ac.cn/archives/8180)
  - 2021/03/08 [Transformer升级之路：1、Sinusoidal位置编码追根溯源](https://spaces.ac.cn/archives/8231) 等系列博客(**重点参考**)
  - ...
- 一些关于 Context 长度扩充的论文及资料:
  - 2023/07, [LongNet: Scaling Transformers to 1,000,000,000 Tokens](https://arxiv.org/abs/2307.02486)


关于[科学空间](https://spaces.ac.cn/), 笔者简单探索了一下大佬的博客归档, 有如下发现:

- 这个[页面](https://spaces.ac.cn/content.html)是所有文章的归档, 详细“食用指南”参考这篇[博客](https://spaces.ac.cn/archives/6508)
- 笔者看到的大佬的第一篇与机器学习/深度学习相关的博客是 2015/06/06 发表的这篇博客: [闲聊：神经网络与深度学习](https://spaces.ac.cn/archives/3331)
- 大佬早期也有许多与技术无关的随笔, 今年来基本保持每周一篇

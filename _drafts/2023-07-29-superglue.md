---
layout: post
title: "(WIP) SuperGLUE 数据集介绍"
date: 2023-07-29 10:00:04 +0800
labels: [dataset]
---

## 动机、参考资料、涉及内容

动机

在学习 [P-Tuning V2](https://github.com/THUDM/P-tuning-v2) 论文时看到了这个 [SuperGLUE](https://huggingface.co/datasets/super_glue) 这个 NLU 数据集, 之前在别处也看到过这个数据集. 在翻看了一下 [SOTA 榜单](https://paperswithcode.com/dataset/superglue) 后, 发现 PaLM 540B 这种模型似乎也会在这个数据集上做一下结果. 不确定到目前为止, 这个数据集是不是被做烂了, 但还是先姑且记录一下, 供学习参考

参考资料

- SuperGLUE [原始论文 (2019.05)](https://arxiv.org/abs/1905.00537), [官网](https://super.gluebenchmark.com/)
- huggingface.co 上维护的 [SuperGLUE 数据集](https://huggingface.co/datasets/super_glue)
- 一篇 [知乎文章](https://zhuanlan.zhihu.com/p/383945098)

## 数据集介绍

RTE 是一个二分类问题, 根据一句话 (premise) 的内容. 判断一个论断 (hypothesis) 是否正确  (entailment). 更学术地说: 某个前提 (论据, premise) 成立时是否蕴含 (支持, entailment) 了假设 (假说, hypothesis).


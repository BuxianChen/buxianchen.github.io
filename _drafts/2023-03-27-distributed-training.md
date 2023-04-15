---
layout: post
title: "(WIP) Pytorch Distributed Training"
date: 2023-03-27 14:31:04 +0800
labels: [pytorch, distributed]
---

## 动机、参考资料、涉及内容

动机：

- 随着模型越来越大，需要理解分布式训练/推理的一些知识

参考资料：

- pytorch 官方文档
- 一些博客:
  - https://www.cnblogs.com/rossiXYZ/p/15815013.html
  - https://lilianweng.github.io/posts/2021-09-25-train-large/
- 李沐的相关视频
- 其他

涉及内容：

- 分布式训练的基本概念（以Pytorch的相关Tutorial为主）
- 主流库的使用（重点）及原理简介，包括：Pytorch DDP、Pytorch FSDP、deepspeed、Megatron-LM
- 采用实际例子进行对比

不涉及内容：

- 各种分布式训练框架过于深入的实现细节

## DDP

介绍完原理直接甩示例代码

## FSDP

关于 Pytorch FSDP:

- (2021/07/15)FairSeq早期实现的版本的介绍博客：https://engineering.fb.com/2021/07/15/open-source/fsdp/
- (2022/03/14)Pytorch在1.11版本引入FSDP的博客：https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/
- Pytorch的tutorial
  - 入门：https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
  - 进阶：https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html

## deepspeed


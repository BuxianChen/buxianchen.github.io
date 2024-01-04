---
layout: post
title: "(DEAD) Memory efficient training"
date: 2023-05-25 23:20:04 +0800
labels: [pytorch]
---

## 动机、参考资料、涉及内容

动机

- 训练时占用的显存分析
- torch.profile 使用
- torch.amp
- gradient checkpointing
- bitsandbytes 使用及原理(不确定是否另起博客)
- 🤗 peft(不确定是否另起博客)

参考资料

- pytorch
- 🤗 peft

涉及内容

- 原理及使用, 附带一些参考的github项目

不涉及内容

- 多卡训练

## 模型训练显存大小估算

【待确定是否与自动微分相关】

本节以 Bert 为例理论分析模型训练时需要的显存, 理论分析的作用是: 如果实际测量发现与理论分析有偏差, 例如实际需要显存明显高于理论分析, 则可以检查实现上是否有问题; 另外也可以在做之前大致估算成本 (如果没有第三方资料可以查找的话)

## `torch.profile` 的使用

使用 `torch.profile` 可以通过实测, 判断模型运行时的瓶颈

## FP16 训练

FP16 训练至少见过几种实现

- Apex
- torch.amp
- 老版本 mmcv 自己实现的一套

【待确定】推理时一般要不就纯 FP32 推理, 要不就纯 FP16 推理, 一般不会出现混用

## gradient checkpointing

【待确定是否与自动微分相关】

## 几种 PEFT 方法概述

🤗 peft

## LoRA

### LoRA 原理

### 🤗 peft 对 LoRA 的实现
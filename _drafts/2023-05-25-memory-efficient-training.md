---
layout: post
title: "(WIP) Memory efficient training"
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
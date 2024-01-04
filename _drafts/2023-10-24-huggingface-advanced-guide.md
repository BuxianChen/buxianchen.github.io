---
layout: post
title: "(LTS) huggingface summary"
date: 2023-10-24 11:12:04 +0800
labels: [huggingface]
---

## 动机、参考资料、涉及内容

动机

之前已有一系列跟 huggingface 相关的博客, 话题主要涉及到:

- transformers.PretrainedModel
  - huggingface_hub
  - autogptq
  - bitandbytes
  - empty_weight: accelerate
- transformers.TokenizerBase, tokenizers
- transformers.GenerationMixin
- transformers.pipeline
- transformers.AutoModel
- accelerate: 不完善
- peft
- safetensor
- optimum: 不完善
- datasets: 不完善
- text-generation-inference: 暂未涉及

本文主要目的如下:

- 简明地说明每一块的底层技术
- 怎样做二次开发
---
layout: post
title: "(WIP) Basics of CPU and GPU"
date: 2023-11-18 10:10:04 +0800
labels: [gpu,cuda]
---

## 动机、参考资料、涉及内容

一些关于 GPU 和 CPU 的知识

参考资料:

- [https://www.telesens.co/2019/02/16/efficient-data-transfer-from-paged-memory-to-gpu-using-multi-threading/](https://www.telesens.co/2019/02/16/efficient-data-transfer-from-paged-memory-to-gpu-using-multi-threading/): PCIe, pytorch dataloader 的 `pin_memory`的含义, page-locked memory


## 内容

组成????

- 计算机硬盘, 主存, CPU(三级缓存+寄存器+计算单元+调度器)
- GPU DRAM(显存), CUDA Core...


计算机主存(目前家用电脑内存大小一般为16GB左右)分为两类:

- pageable ("un-pinned") memory
- page-locked ("pinned") memory

CPU 到 GPU 数据的拷贝过程:

```
CPU (pageable memory) -> CPU (page-locaked memory) --[PCIe]--> GPU
```
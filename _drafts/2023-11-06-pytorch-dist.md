---
layout: post
title: "(P1) torch.distributed 总览及底层工具"
date: 2023-11-04 14:10:04 +0800
labels: [deepspeed]
---

## 动机、参考资料、涉及内容

为了阅读 fairscale 源码, 涉及到 `torch.distributed` 模块, DP/DDP/FSDP/Pipe/Tensor Parallel 这些上层技术都用到了这些基础设施 (例如通信原语), 本文只涉及这些基础设施. 这些接口的含义似乎大体都类似于 MPI/OpenMP


这里额外收录一些关于分布式训练的资料:

- https://www.cnblogs.com/rossiXYZ/p/15815013.html
- https://lilianweng.github.io/posts/2021-09-25-train-large/

关于 Pytorch FSDP:

- (2021/07/15)FairSeq早期实现的版本的介绍博客：https://engineering.fb.com/2021/07/15/open-source/fsdp/
- (2022/03/14)Pytorch在1.11版本引入FSDP的博客：https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/
- Pytorch的tutorial
  - 入门：https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
  - 进阶：https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html


## APIs

- `process_group`

```python
# 初始化默认的 process_group, 必须写在程序开头
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group("nccl", rank=rank, world_size=world_size)
```


```python
# 获取到已创建默认的 process_group
import torch.distributed as dist
dist.group.WORLD  # torch._C._distributed_c10d.ProcessGroupNCCL
isinstance(dist.group.WORLD, torch._C._distributed_c10d.ProcessGroup)  # True
```
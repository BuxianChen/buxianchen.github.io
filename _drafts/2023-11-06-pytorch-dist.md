---
layout: post
title: "(WIP) torch.distributed"
date: 2023-11-04 14:10:04 +0800
labels: [deepspeed]
---

## 动机、参考资料、涉及内容

为了阅读 fairscale 源码, 涉及到 `torch.distributed` 模块

本质上, 这些接口的含义都类似于 MPI/OpenMP

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
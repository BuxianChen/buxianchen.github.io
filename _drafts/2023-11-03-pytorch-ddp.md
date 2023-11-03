---
layout: post
title: "(WIP) Pytorch Distributed Data Parallel"
date: 2023-11-03 10:10:04 +0800
labels: [pytorch]
---

## 动机、参考资料、涉及内容


## 实现

相关的API内容参考: [https://pytorch.org/docs/1.7.0/distributed.html](https://pytorch.org/docs/1.7.0/distributed.html)

### 样例 1

参考: [https://pytorch.org/docs/1.7.1/notes/ddp.html](https://pytorch.org/docs/1.7.1/notes/ddp.html)

```python
# test.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def example(rank, world_size):  # 注意: rank 参数会被自动传递, 由 nprocs 决定
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank)
    print("init weight:", "<rank {rank}>", model.weight[0, 0].item())
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=1.)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    print("weight grad 1:", "<rank {rank}>", model.weight.grad[0, 0].item())
    # update parameters
    optimizer.step()
    print("weight update 1:", "<rank {rank}>", model.weight[0, 0].item())

    optimizer.zero_grad()

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    print("weight grad 2:", "<rank {rank}>", model.weight.grad[0, 0].item())
    # update parameters
    optimizer.step()
    print("weight update 2:", "<rank {rank}>", model.weight[0, 0].item())

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
```

运行

```bash
python test.py
```

输出结果(不包括 `#` 后的内容)

```
init weight: <rank 1> -0.0235  # 注意初始化参数实际上不一致(与下一行对比) 
init weight: <rank 0> -0.0271
weight grad 1: <rank 1> 0.0116 # 这一行在 backward 之后被打印, 因此梯度被平均了
weight grad 1: <rank 0> 0.0116
weight updated 1: <rank 1> -0.0387  # 这里很有趣, 更新后rank 1的权重跟rank 0一致了, 并且权重更新以 rank 0 为准
weight updated 1: <rank 0> -0.0387  # -0.0387 = -0.0271 - 1*0.0116
weight grad 2: <rank 1> 0.0289
weight grad 2: <rank 0> 0.0289
weight updated 2: <rank 1> -0.0676
weight updated 2: <rank 0> -0.0676
```

分析 1:

这里需要额外指出的一个要点是: **梯度是被平均的**, 可以通过改写上面的程序做验证

- 怎么验证梯度是**被平均**的: 可以通过将 `Linear` 的输入输出维度定为 1, 然后将模型输入定为 2, `labels` 也设置为 1
  ```python
  model = nn.Linear(1, 1, bias=False)
  if rank == 0:
    model.weight.data.fill_(2.)
  else:
    model.weight.data.fill_(200.)
  labels = torch.tensor([[1.]]).to(rank)
  ```
- 过程猜想: 前向传播各自算, `backward` 时自动触发进程间通讯: 梯度用 reduce 操作做平均, `optimizer.step` 操作似乎是先更新 `<rank 0>` 的参数, 然后把 `<rank 0>` 的参数广播给其他进程.
- 但是上面的解释似乎不准确 (可以尝试将 `<rank 0>` 的初始权重设为 2, `<rank 1>` 的初始权重设为 `200`), 输出结果如下: 注意只考虑一个进程时, 梯度的计算公式如下, `loss = (w*1-1)**2, w.grad = 2*w-2`, 由结果可以看出**梯度是被平均的**
  ```
  init weight: <rank 1> 200.
  init weight: <rank 0> 2.
  weight grad 1: <rank 1> 2.  # 2=2*w-2=2*2-2
  weight grad 1: <rank 0> 2.
  weight updated 1: <rank 1> 0.
  weight updated 1: <rank 0> 0.
  weight grad 2: <rank 1> -2.  # -2=2*w-2=2*0-2
  weight grad 2: <rank 0> -2.
  weight updated 2: <rank 1> 2.
  weight updated 2: <rank 0> 2.
  ```
  **猜想**可能在第一次前向传播时触发了一次通讯, 检查两个进程的权重是否一致, 如果不一致则在 backward 时以 `<rank 0>` 为准, 然后第一次更新权重时需要将权重广播 (并设置一个标记表示权重已同步), 后面就可以免去参数广播这一步

分析 2:

注意看这个例子中使用的是 `torch.multiprocessing.spawn`, 并且搭配的是普通的脚本方式 `python test.py`, 这不是平时的典型用法. 而且注意到 `torch.multiprocessing.spawn` 的调用方式如下:

```python
mp.spawn(example,
    args=(world_size,),
    nprocs=world_size,
    join=True)
# example 实际上有 2 个参数, rank 与 world_size, 这里第一个参数 rank 会被分别自动传为 0, 1
```


### DistributedSampler, dist.all_gather

```python
# sampler.py
import torch
import os
import torch.distributed as dist

class DemoDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self._data = [i for i in range(n)]
    def __getitem__(self, i):
        return self._data[i] * 10
    def __len__(self):
        return len(self._data)

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    world_size = os.environ["WORLD_SIZE"]
    local_rank = os.environ["LOCAL_RANK"]
    dataset = DemoDataset(16)
    batch_size_per_gpu = 4
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    # 训练时总共的 batch_size 是 batch_size_per_gpu * world_size
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size_per_gpu)
    for epoch in range(2):
        sampler.set_epoch(epoch)  # 注意设置这个!!
        for item in dataloader:
            item = item.to(local_rank)  # 要手动移至 GPU
            print(f"epoch {epoch}:", local_rank, "=>", item)
            # 提前分配空间, to(local_rank) 是必要的
            all_item = [torch.zeros(batch_size, dtype=torch.long).to(local_rank) for i in range(world_size)]
            dist.all_gather(all_item, item)  # 必须确保所有进程都执行 all_gather
            print("epoch {epoch}:", local_rank, "all =>", all_item)
            if local_rank == 0:  # 一般来说这个区域打印日志或写文件
                pass
# python -m torch.distributed.launch --nnodes 1 --nproc_per_node 2 sampler.py
```

输出

```
epoch 0: 0 => tensor([120, 90, 110, 130], device='cuda:0')
epoch 0: 1 => tensor([20, 150, 40, 70], device='cuda:1')
epoch 0: 0 all  => [tensor([120, 90, 110, 130], device='cuda:0'), tensor([20, 150, 40, 70], device='cuda:0')]
epoch 0: 1 all  => [tensor([120, 90, 110, 130], device='cuda:0'), tensor([20, 150, 40, 70], device='cuda:0')]
# 略 ...
```

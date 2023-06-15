---
layout: post
title: "(WIP) Pytorch 性能分析工具"
date: 2023-06-15 14:10:04 +0800
labels: [pytorch]
---

## 动机、参考资料、涉及内容

动机

- python 程序性能分析
- pytorch 程序性能分析

参考资料

- torch.profiler: 主要用于分析模型调用各个部分的耗时
  - torch.profiler 官方教程: [https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
  - torch.profiler API 文档: [https://pytorch.org/docs/stable/profiler.html](https://pytorch.org/docs/stable/profiler.html)
  - tensorboard 集成 torch.profiler 的官方教程: [https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
  - 本文只关注 pytorch>=1.8.0 API, 因此老版本被弃用的 API 的教程 [https://pytorch.org/tutorials/beginner/profiler.html](https://pytorch.org/tutorials/beginner/profiler.html) 不予讨论
- torch.utils.benchmark: 主要用于分析总体的时间对比, 对标标准库 timeit
  - torch.utils.benchmark 官方教程: [https://pytorch.org/tutorials/recipes/recipes/benchmark.html](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)

涉及内容

- 以上参考资料内容简要复述
- torch.profiler 更深入的使用, 教程中只介绍了 `with profiler(...) as prof` 和 `prof.key_averages(...).table(...)` 的用法, 但这种用法最终返回的"报表"是字符串形式的, 适合于肉眼观察, 不利于使用代码进行进一步处理和分析, 因此需要结合 API 文档和源码进一步研究
- 一般的 python 程序性能分析工具乃至于一般的程序性能分析工具(linux 工具)
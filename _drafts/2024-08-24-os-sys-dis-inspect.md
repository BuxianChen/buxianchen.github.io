---
layout: post
title: "(LTS) python 底层内置模块 (os/sys/inspect/dis/ast 等)"
date: 2024-08-24 13:00:04 +0800
labels: [python]
---

## 动机

在研究一些底层源码时, 会涉及到一些底层接口, 尤其是 os/sys

## 内容

### sys._getframe, inspect.stack, types.FrameType

这两个函数都是用于查看当前栈帧信息, 前者更底层些

```python
import inspect
import os
import sys


def foo():
    bar()

def bar():
    print(sys._getframe(1))   # sys._getframe(0) 是当前帧, sys._getframe(1) 是上级调用帧
    print(sys._getframe(1).__class__)  # frame 类型, 实际上是 types.FrameType
    print("="*20)
    stack = inspect.stack()   # stack[0] 是当前帧, 也就是 bar, stack[1] 是上级调用帧, 也就是 foo
    for frame_info in stack:  # frame_info 是 inspect.FrameInfo 对象
        print(frame_info)
        print(frame_info.__class__)
        print(frame_info.frame)  # frame 类型
        print(frame_info.frame.__class__)

foo()
```


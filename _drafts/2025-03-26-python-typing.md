---
layout: post
title: "(LST) Python typing 模块"
date: 2025-03-26 10:05:04 +0800
labels: [python]
---

## 动机、参考资料、涉及内容

替代这部分笔记: [https://buxianchen.gitbook.io/notes/language/python/python_standard_library](https://buxianchen.gitbook.io/notes/language/python/python_standard_library)

## mypy

```bash
# 严格检查单个文件
mypy --strict your_file.py
```

## typing

### Generic

TODO

```python
from typing import Generic, TypeVar

SendRequestT = TypeVar('SendRequestT')
SendResultT = TypeVar('SendResultT')

class BaseSession(Generic[SendRequestT, SendResultT]):
    def send_request(self, request: SendRequestT) -> None:
        return None
```

### 适用于 kwargs 的 Unpack 注解

```python
from typing import TypedDict
from typing_extensions import Unpack

class DeprecatedKwargs(TypedDict, total=False):
    old_param: int
    legacy_mode: bool

def foo(x: int, **kwargs: Unpack[DeprecatedKwargs]) -> None:
    print(kwargs["old_param"])  # mypy 不会报错, 且 IDE 会自动提醒可选的 key 有 "old_param"
    print(kwargs["y"])  # mypy 报错

if __name__ == "__main__":
    # 运行正常
    foo(1, y=2, old_param=3)  # mypy 报错, 且 IDE 会自动提醒可选的入参 old_param 和 legacy_param
```
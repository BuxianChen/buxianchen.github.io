---
layout: post
title: "(LST) Python typing 模块"
date: 2025-03-26 10:05:04 +0800
labels: [python]
---

## 动机、参考资料、涉及内容

替代这部分笔记: [https://buxianchen.gitbook.io/notes/language/python/python_standard_library](https://buxianchen.gitbook.io/notes/language/python/python_standard_library)

## mypy

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
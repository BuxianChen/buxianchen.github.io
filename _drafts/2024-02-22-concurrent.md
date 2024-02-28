---
layout: post
title: "(P1) 并发 (for Python)"
date: 2024-02-22 10:05:04 +0800
labels: [python]
---

## 动机、参考资料、涉及内容

threading, multiprocessing, asyncio 内容杂记

## threading

### threading.Event

参考博客: [https://www.pythontutorial.net/python-concurrency/python-threading-event/](https://www.pythontutorial.net/python-concurrency/python-threading-event/)

`threading.Event` 是对 bool 类型的一个包装, 使用 `event.set()` 表示将其设置为 `True`, 使用 `event.clear()` 表示将其设置为 `False`, `event.is_set()` 判断其为 `True` 或 `False`, `event.wait()` 会阻塞当前线程, 直到有别的线程将其设置为 `True`, 本线程才会继续执行. 更详细的说明上面的博客比较清楚.

```python
from threading import Thread, Event
from time import sleep

def task(event: Event, id: int):
    print(f'Thread {id} started. Waiting for the signal....')
    event.wait()
    print(f'Received signal. The thread {id} was completed.')

def main():
    event = Event()  # 默认为 False
    t1 = Thread(target=task, args=(event,1))
    t2 = Thread(target=task, args=(event,2))
    t1.start()
    t2.start()
    print('Blocking the main thread for 3 seconds...')
    sleep(3) 
    event.set()

if __name__ == '__main__':
    main()
```

上面的例子中, 直到主线程将 `event` 设置为 `True` 时, 两个子线程才继续执行下去.
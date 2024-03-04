---
layout: post
title: "(P1) 并发 (for Python)"
date: 2024-02-22 10:05:04 +0800
labels: [python]
---

## 动机、参考资料、涉及内容

threading, multiprocessing, asyncio 内容杂记

## os

### os.waitpid

`os.waitpid(-1, os.WNOHANG)` 似乎应该理解为? 从已经结束的子进程里取出一个, 如果全部被取出则报错

```python
import os
import time

for i in range(2):
    if os.fork() == 0:  # 子进程
        time.sleep(0.2)
        os._exit(0)
    else:  # 父进程
        pass

print(os.waitpid(-1, os.WNOHANG))  # (0, 0)
time.sleep(0.1)
print(os.waitpid(-1, os.WNOHANG))  # (0, 0)
time.sleep(1)
print(os.waitpid(-1, os.WNOHANG))  # (7746, 0)
print(os.waitpid(-1, os.WNOHANG))  # (7747, 0)
try:
    print(os.waitpid(-1, os.WNOHANG))
except OSError as err:
    print(err)  # ChildProcessError: [Errno 10] No child processes
```

### os._exit

[https://www.geeksforgeeks.org/python-os-_exit-method/](https://www.geeksforgeeks.org/python-os-_exit-method/)

`os._exit` 通常用于 `os.fork` 产生的子进程里退出子进程, 它会自动做些清理工作, 例如关闭没有被关闭的文件描述符. 如果是用于提前关闭主进程, 则需要使用 `sys.exit`

### os.pipe

`os.pipe` 用于创建管道, 管道分为读端与写端, 使用 `os.fork` 的子进程与其父进程共享文件描述符, 但是子进程关闭读端不会引起父进程关闭读端, 因此 `os.pipe` 常用于进程间的通信. `multiprocessing.Process` 的底层也使用到了 `os.pipe`

**例 1: 父子进程通信**

```python
import os

# 创建管道
read_end, write_end = os.pipe()  # 此处返回的是两个文件描述符

# 在子进程中，关闭不需要的管道端，并向管道写入数据
if os.fork() == 0:  # 子进程
    print(f"Child End, read_end: {read_end}, write_end: {write_end}")
    os.close(read_end)  # 关闭读端
    data_to_send = b"Hello, pipe!"
    os.write(write_end, data_to_send)
    os.close(write_end)  # 写完数据后关闭写端
    exit()  # 或者使用 os._exit(0)

# 在父进程中，关闭不需要的管道端，并从管道读取数据
else:  # 父进程
    print(f"Parent End, read_end: {read_end}, write_end: {write_end}")
    os.close(write_end)  # 关闭写端
    data_received = os.read(read_end, 1024)
    os.close(read_end)  # 读完数据后关闭读端
    print("Received:", data_received.decode())
```

输出 (推测是 0, 1, 2 文件描述符分别被 STDIN, STDOUT, STDERR 占据, 然后接下来就是 3 和 4):

```
Parent End, read_end: 3, write_end: 4
Child End, read_end: 3, write_end: 4
Received: Hello, pipe!
```

**例 2: 兄弟进程通信**

```python
import os

def child_write(write_end):
    os.close(write_end)  # 关闭不需要的管道端
    data_to_send = b"Hello from child!"
    os.write(read_end, data_to_send)
    os.close(read_end)  # 写完数据后关闭管道

def child_read(read_end):
    os.close(read_end)  # 关闭不需要的管道端
    data_received = os.read(write_end, 1024)
    print("Received:", data_received.decode())

# 创建管道
read_end, write_end = os.pipe()

# 创建第一个子进程，负责写数据
if os.fork() == 0:
    child_write(write_end)
    exit()

# 创建第二个子进程，负责读数据
if os.fork() == 0:
    child_read(read_end)
    exit()

# 父进程关闭管道
os.close(read_end)
os.close(write_end)
```


**例 3: 创建多个子进程**

```python
import os

ends = []

for i in range(3):
    # 创建管道
    read_end, write_end = os.pipe()
    
    ends.append(read_end)
    ends.append(write_end)

    # 在子进程中，关闭不需要的管道端，并向管道写入数据
    if os.fork() == 0:  # 子进程
        print(f"Child End, read_end: {read_end}, write_end: {write_end}")
        os.close(read_end)  # 关闭读端
        data_to_send = f"Hello, pipe write_end: {write_end}!".encode()
        os.write(write_end, data_to_send)
        os.close(write_end)  # 写完数据后关闭写端
        os._exit(0)

    # 在父进程中，关闭不需要的管道端，并从管道读取数据
    else:  # 父进程
        print(f"Parent End, read_end: {read_end}, write_end: {write_end}")
        # os.close(write_end)  # 关闭写端
        data_received = os.read(read_end, 1024)
        # os.close(read_end)  # 读完数据后关闭读端
        print("Received:", data_received.decode())

for end in ends:
    os.close(end)
```

输出

```
Parent End, read_end: 3, write_end: 4
Child End, read_end: 3, write_end: 4
Received: Hello, pipe write_end: 4!
Parent End, read_end: 5, write_end: 6
Child End, read_end: 5, write_end: 6
Received: Hello, pipe write_end: 6!
Parent End, read_end: 7, write_end: 8
Child End, read_end: 7, write_end: 8
Received: Hello, pipe write_end: 8!
```

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

## multiprocessing

### Process 源码

**分析 Python 3.9 源码**

参考: [https://www.cnblogs.com/ellisonzhang/p/10440453.html](https://www.cnblogs.com/ellisonzhang/p/10440453.html)

注意点:

- multiprocessing.Process 中的 daemon 属性与 Linux 中的 daemon 进程是不同的概念. 
  - 在 multiprocessing 模块中, Process 类的 daemon 属性用于指定进程是否为守护进程. 守护进程是一种在主进程结束时自动结束的子进程. 如果将 daemon 属性设置为 True, 则子进程将随着主进程的结束而结束; 如果设置为 False, 则子进程将在主进程结束后继续运行. 默认情况下, daemon 属性的值为 False.
  - 在 Linux 中，daemon 进程是一种在后台运行且不受终端影响的进程。它们通常在系统启动时启动，并在后台持续运行，常用于提供系统服务或执行周期性任务。
- `import multprocessing` 的时候发生了一些行为
- `Process.start` 触发 `ForkProcess` 的初始化, 初始化触发 `ForkProcess._launch`, `_launch` 触发 `Process._bootstrap`, `_bootstrap` 触发 `Process.run`, 而 `Process.run` 很简单, 只是将 `target` 运行, 因此继承 `Process` 可以考虑对 `run` 重载

**测试代码**

```python
import multiprocessing

def print_hello():
    print("hello world")

# 疑问: 此处打印 globals(), 发现并没有 multiprocessing.__init__ 对 globals() 添加的属性
if __name__ == "__main__":
    # 疑问: 此处打印 globals(), 发现并没有 multiprocessing.__init__ 对 globals() 添加的属性
    p = multiprocessing.Process(target=print_hello)
    p.start()
    p.join()
```

**import multiprocessing**

```python
# multiprocessing.__init__.py
import sys
from . import context

__all__ = [x for x in dir(context._default_context) if not x.startswith('_')]
# 注意此处加了些全局变量, 但似乎在入口脚本中没有体现
globals().update((name, getattr(context._default_context, name)) for name in __all__)

SUBDEBUG = 5
SUBWARNING = 25

if '__main__' in sys.modules:  # 疑似与 multiprocessing 代码需要在 `__main__` 下被调用相关, 待研究
    sys.modules['__mp_main__'] = sys.modules['__main__']
```

继续追溯 `multiprocessing/context.py` 在被 import 的时候发生的一些初始化动作

```python
# multiprocessing/context.py
import os
import sys
import threading

from . import process    # 这里面初始化了一些全局变量
from . import reduction  # 这里面似乎没有初始化全局变量

# 省略若干代码

# 这里只体现 sys.platform == "linux" 的情况
class ForkProcess(process.BaseProcess):
    _start_method = 'fork'
    @staticmethod
    def _Popen(process_obj):
        from .popen_fork import Popen   # !!真正的逻辑发生在这里!!, multiprocessing.popen_fork.Popen 是一个 class
        return Popen(process_obj)

class SpawnProcess(process.BaseProcess):
    _start_method = 'spawn'
    @staticmethod
    def _Popen(process_obj):
        from .popen_spawn_posix import Popen
        return Popen(process_obj)

class ForkServerProcess(process.BaseProcess):
    _start_method = 'forkserver'
    @staticmethod
    def _Popen(process_obj):
        from .popen_forkserver import Popen
        return Popen(process_obj)

class ForkContext(BaseContext):
    _name = 'fork'
    Process = ForkProcess

class SpawnContext(BaseContext):
    _name = 'spawn'
    Process = SpawnProcess

class ForkServerContext(BaseContext):
    _name = 'forkserver'
    Process = ForkServerProcess
    def _check_available(self):
        if not reduction.HAVE_SEND_HANDLE:
            raise ValueError('forkserver start method not available')

_concrete_contexts = {
    'fork': ForkContext(),
    'spawn': SpawnContext(),
    'forkserver': ForkServerContext(),
}

_default_context = DefaultContext(_concrete_contexts['fork'])  # linux 下默认用 fork
```

继续追溯 `multiprocessing/process.py` 待后续分析

```python
# multiprocessing/process.py
import os
import sys
import signal
import itertools
import threading   # 这个 import 也初始化了一些东西
from _weakrefset import WeakSet

# ...

# _MainProcess 很简单, 后面分析 BaseProcess 时可以看到主要时为 BaseProcess 设置一些属性用
class _MainProcess(BaseProcess):
    def __init__(self):
        self._identity = ()
        self._name = 'MainProcess'
        self._parent_pid = None
        self._popen = None
        self._closed = False
        self._config = {'authkey': AuthenticationString(os.urandom(32)), 'semprefix': '/mp'}
        # Everything in self._config will be inherited by descendant processes.
    def close(self):
        pass

_parent_process = None
_current_process = _MainProcess()
_process_counter = itertools.count(1)
_children = set()
del _MainProcess   # 这里又被 del 掉了

_exitcode_to_name = {}

for name, signum in list(signal.__dict__.items()):
    if name[:3]=='SIG' and '_' not in name:
        _exitcode_to_name[-signum] = f'-{name}'

# For debug and leak testing
_dangling = WeakSet()
```

这样我们在 `import multiprocessing` 模块时, 至少有如下这些分析目标:

- `multiprocessing/context.py` 中的 `_default_context` 是什么
- `multiprocessing/process.py` 中的 `_current_process` 是什么

我们先回到测试代码的这一行: `p = multiprocessing.Process(target=print_hello)`

```python
# process.py: BaseProcess
# context.py: Process/ForkProcess/SpawnProcess/ForkServerProcess, BaseContext/ForkContext/SpawnContext/ForkServerContext
class Process(process.BaseProcess):
    _start_method = None
    @staticmethod
    def _Popen(process_obj):
        return _default_context.get_context().Process._Popen(process_obj)   # !!!最要紧之处!!!
```

于是我们回到了对 `_default_context` 的研究

```python
# 注意: 在 linux 上, 默认是 fork, 因此 _default_context 本质上是这个
# _default_context = DefaultContext(ForkContext())
# 而 DefaultContext 与

class DefaultContext(BaseContext):
    Process = Process

    def __init__(self, context):
        self._default_context = context
        self._actual_context = None

    def get_context(self, method=None):
        if method is None:  # 一般来说会进到这个分支
            if self._actual_context is None:
                self._actual_context = self._default_context
            return self._actual_context
        else:
            return super().get_context(method)

    def set_start_method(self, method, force=False):
        if self._actual_context is not None and not force:
            raise RuntimeError('context has already been set')
        if method is None and force:
            self._actual_context = None
            return
        self._actual_context = self.get_context(method)

    def get_start_method(self, allow_none=False):
        if self._actual_context is None:
            if allow_none:
                return None
            self._actual_context = self._default_context
        return self._actual_context._name

    def get_all_start_methods(self):
        if sys.platform == 'win32':
            return ['spawn']
        else:
            methods = ['spawn', 'fork'] if sys.platform == 'darwin' else ['fork', 'spawn']
            if reduction.HAVE_SEND_HANDLE:
                methods.append('forkserver')
            return methods

class ForkContext(BaseContext):
    _name = 'fork'
    Process = ForkProcess
```

`Process` 的 `_Popen` 方法: `return _default_context.get_context().Process._Popen(process_obj)`

- `_default_context.get_context()`: `ForkContext`
- `_default_context.get_context().Process`: `ForkProcess`


```python
class BaseProcess:
    def start(self):
        '''
        Start child process
        '''
        self._check_closed()
        assert self._popen is None, 'cannot start a process twice'
        assert self._parent_pid == os.getpid(), \
               'can only start a process object created by current process'
        assert not _current_process._config.get('daemon'), \
               'daemonic processes are not allowed to have children'
        _cleanup()
        self._popen = self._Popen(self)
        self._sentinel = self._popen.sentinel
        # Avoid a refcycle if the target function holds an indirect
        # reference to the process object (see bpo-30775)
        del self._target, self._args, self._kwargs
        _children.add(self)
```


```python
# multiprocessing/popen_fork.py 的完整源码:
import os
import signal

from . import util

__all__ = ['Popen']

class Popen(object):
    method = 'fork'

    def __init__(self, process_obj):
        util._flush_std_streams()
        self.returncode = None
        self.finalizer = None
        self._launch(process_obj)

    def duplicate_for_child(self, fd):
        return fd

    def poll(self, flag=os.WNOHANG):
        if self.returncode is None:
            try:
                pid, sts = os.waitpid(self.pid, flag)
            except OSError:
                return None
            if pid == self.pid:
                self.returncode = os.waitstatus_to_exitcode(sts)
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            if timeout is not None:
                from multiprocessing.connection import wait
                if not wait([self.sentinel], timeout):
                    return None
            # This shouldn't block if wait() returned successfully.
            return self.poll(os.WNOHANG if timeout == 0.0 else 0)
        return self.returncode

    def _send_signal(self, sig):
        if self.returncode is None:
            try:
                os.kill(self.pid, sig)
            except ProcessLookupError:
                pass
            except OSError:
                if self.wait(timeout=0.1) is None:
                    raise

    def terminate(self):
        self._send_signal(signal.SIGTERM)

    def kill(self):
        self._send_signal(signal.SIGKILL)

    def _launch(self, process_obj):
        code = 1
        parent_r, child_w = os.pipe()
        child_r, parent_w = os.pipe()
        self.pid = os.fork()
        if self.pid == 0:
            try:
                os.close(parent_r)
                os.close(parent_w)
                code = process_obj._bootstrap(parent_sentinel=child_r)  # 此处似乎并没有关闭 child_r
            finally:
                os._exit(code)  # 这里会自动关闭 child_r 与 child_w
        else:
            os.close(child_w)
            os.close(child_r)
            self.finalizer = util.Finalize(self, util.close_fds, (parent_r, parent_w,))
            self.sentinel = parent_r

    def close(self):
        if self.finalizer is not None:
            self.finalizer()  # 此处内部会关闭 parent_r 和 parent_w
```

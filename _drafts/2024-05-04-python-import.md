---
layout: post
title: "(P1) Python import 相关"
date: 2024-05-04 22:05:04 +0800
labels: [python,import]
---

## 动机、参考资料、涉及内容

**参考资料**

- [1] Python 官方文档, 对 import 语句的描述: [https://docs.python.org/3/reference/simple_stmts.html#import](https://docs.python.org/3/reference/simple_stmts.html#import)
- [2] Python 官方文档, 对 import system 的描述: [https://docs.python.org/3/reference/import.html#importsystem](https://docs.python.org/3/reference/import.html#importsystem)
- [3] Python 官方文档, 对内置模块 importlib 的描述: [https://docs.python.org/3/library/importlib.html](https://docs.python.org/3/library/importlib.html)
- [4] Python 官方文档, 对内置函数 `__import__` 的描述: [https://docs.python.org/3/library/functions.html#import__]（https://docs.python.org/3/library/functions.html#import__
- [5] Python 官方文档, 对 `typing.ModuleType` 的描述: [https://docs.python.org/3/library/types.html#types.ModuleType](https://docs.python.org/3/library/types.html#types.ModuleType)

## FAQ

所谓的 module 实际上就是 `types.ModuleType` 类型的实例, 而 package 是指带有 `__path__` 属性的 module

## (TODO) import 语句

参考[1](https://docs.python.org/3/reference/simple_stmts.html#import)


### `types.ModuleType`

```python
import re
import types
re.__class__ is types.MuduleType  # True
```

参考[5](https://docs.python.org/3/library/types.html#types.ModuleType)

```python
types.ModuleType(name, doc=None)
# ModuleType 有如下属性
# __doc__: 入参 doc
# __loader__: importlib.machinery.ModuleSpec.loader
# __name__: importlib.machinery.ModuleSpec.name
# __package__: importlib.machinery.ModuleSpec.parent
# __spec__: importlib.machinery.ModuleSpec

# 以下在文档中没有提及, 估计是可选的(需确认)
# __file__
# __path__
```

### loading

参考[2](https://docs.python.org/3/reference/import.html#loading), 使用 module spec 来 load module 的过程大致如下:

```python
module = None
if spec.loader is not None and hasattr(spec.loader, 'create_module'):
    # It is assumed 'exec_module' will also be defined on the loader.
    module = spec.loader.create_module(spec)
if module is None:
    module = ModuleType(spec.name)
# The import-related module attributes get set here:
_init_module_attrs(spec, module)  # 应该是设置 module.__loader__, ...

if spec.loader is None:
    # unsupported
    raise ImportError
if spec.origin is None and spec.submodule_search_locations is not None:
    # namespace package
    sys.modules[spec.name] = module
elif not hasattr(spec.loader, 'exec_module'):
    module = spec.loader.load_module(spec.name)
else:
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        try:
            del sys.modules[spec.name]
        except KeyError:
            pass
        raise
return sys.modules[spec.name]
```

## (TODO) module & package

module 包含如下属性:

- `__spec__`:
  - `name`: `__name__`, fully qualified name, 例如: `m.__spec__.name="pkg.subpkg"`
  - `loader`: `__loader__`
  - `origin`:
  - `loader_state`:
  - `submodule_search_locations`: list(str), 代表子模块搜索路径
  - `_cached`: `__cached__`, str
  - `parent`: `__package__`, 基本上也类似于 `__name__`
- `__doc__`:
- `__path__`: 只有 package 才有 `__path__` 属性, 否则只是普通的 module
- `__file__`:

## (TODO) `__import__`

import 语句是 `__import__` 函数的语法糖 ([3](https://docs.python.org/3/library/importlib.html#importlib.import_module)), import 语句的语法是以下几种 (精准定义[1](https://docs.python.org/3/reference/simple_stmts.html#import)):

```python
import module1 as names1, module2 as names2
from module import a as x, b as y
from module import *
# import ..a  # Error!
```

而 `__import__` 函数的函数签名[4](https://docs.python.org/3/library/functions.html#import__)是 (注意: 不要直接使用 `__import__`, 而应该使用 `importlib.import_module`):

```python
__import__(name, globals=None, locals=None, fromlist=(), level=0)
```

注意 `locals` 参数虽然被包含在函数签名中, 但 `__import__` 函数的实现没有使用这个参数, `__import__` 函数与 `import` 语句的对应关系如下:

```python
import spam
# spam = __import__('spam', globals(), locals(), [], 0)      # __import__ + 名称绑定

import spam.ham
# spam = __import__('spam.ham', globals(), locals(), [], 0)  # __import__ + 名称绑定

from spam.ham import eggs, sausage as saus
# _temp = __import__('spam.ham', globals(), locals(), ['eggs', 'sausage'], 0)  # __import__
# eggs = _temp.eggs     # 名称绑定
# saus = _temp.sausage  # 名称绑定

from ..a import x
# module_a = __import__("a", globals(), locals(), ["x"], 2)  # __import__
# x = module_a.x        # 名称绑定
```

综上, import 语句其实分为两步:

- 构造合适的参数调用 `__import__` 函数, 函数将返回一个“模块”
- 利用返回的模块进行名称绑定

注意: import 语句确实会调用 `__import__` 函数, 但 `importlib.import_module` 不会调用 `__import__` 函数([2](https://docs.python.org/3/reference/import.html#importsystem)).

## importlib

先介绍一些简单的方法, 然后系统介绍. 官方文档[3](https://docs.python.org/3/library/importlib.html)中有一些例子介绍 importlib 的使用, 从对外接口上, 主要是涉及下面的 API

- `importlib.import_module`: 实现搜索目录下模块的导入, 内部会使用 `importlib.util.module_from_spec`
- 结合 `import.util.spec_from_file_location` 和 `import.util.module_from_spec` 来进行任意位置的 python 模块导入
- `importlib.reload`: 重新导入已导入的模块, 主要用于调试(估计 Jupyter 中的魔术指令 autoreload 应该就是基于此)

### (Finish) `importlib.import_module`

```python
importlib.import_module(name, package=None)  # 函数签名

import pkg.mod
# pkg_mod_module = importlib.import_module('..mod', package="pkg.subpkg")
# pkg_mod_module = importlib.import_module('pkg.mod')
# 但注意使用 import 语句后续应该用 pkg.mod.a 而使用 importlib 后续只能用 pkg_mod_module

from pkg.a import x
# import importlib
# module_a = importlib.import_module(".a", "pkg")
# module_a = importlib.import_module("pkg.a")  # 均可,不确定区别
# x = module_a.x

from ..a import x
# import importlib
# # __package__ == 'pkg.subpkg'
# module_a = importlib.import_module('..a', package=__package__)
# x = module_a.x
```

### (Finish) `importlib.import_module` 源码探索

TO;DR: 跳到本节最后的总结

Python 版本: 3.9.16

```python
# importlib/__init__.py
def import_module(name, package=None):
    level = 0
    if name.startswith('.'):
        if not package:
            msg = "the 'package' argument is required to perform a relative import for {!r}"
            raise TypeError(msg.format(name))
        for character in name:
            if character != '.':
                break
            level += 1
    return _bootstrap._gcd_import(name[level:], package, level)
```

如果使用相对导入, 则必须设置 `package`, 例如: `name="..a.b.c", package="d.e"`, 那么导入的包是 `d.a.b.c`, 一个实际的例子:

```python
# langchain_community/agent_toolkits/json/prompt.py
# langchain_community/docstore/base.py
from importlib import import_module
# module: <module 'langchain_community.docstore.base'
module = import_module("..docstore.base", "langchain_community.agent_toolkits")
```

如果使用绝对导入, 则一般无需设置 `package`, 从上面可以看出, `import_module` 最终会由 `_gcd_import` 来完成:

```python
name, package = "..a.b.c", "d.e"
level = 2
_bootstrap._gcd_import("a.b.c", "d.e", 2)

name, package = "a.b.c", None
level = 0
_bootstrap._gcd_import("a.b.c", None, 0)
```

接下来继续看看 `_bootstrap._gcd_import`

```python
# importlib/_bootstrap.py
def _gcd_import(name, package=None, level=0):
    _sanity_check(name, package, level)  # 仅仅是对入参做检查, 比较简单
    if level > 0:
        name = _resolve_name(name, package, level)
    return _find_and_load(name, _gcd_import)
```

其中 `_resolve_name` 会将相对导入处理为绝对导入, 例如:

```python
name, package, level = "..a.b.c", "d.e", 2
name = _resolve_name(name, package, level)  # name = "d.a.b.c"
```

接下来继续看 `_find_and_load(name, _gcd_import)`, 初看上去第二个参数有些奇怪

```python
_NEEDS_LOADING = object()

def _find_and_load(name, import_):
    with _ModuleLockManager(name):  # 加锁去锁操作, 从略
        module = sys.modules.get(name, _NEEDS_LOADING)  # 先找缓存 sys.modules
        if module is _NEEDS_LOADING:
            return _find_and_load_unlocked(name, import_)

    if module is None:
        message = 'import of {} halted; None in sys.modules'.format(name)
        raise ModuleNotFoundError(message, name=name)

    _lock_unlock_module(name)  # 加锁去锁操作, 从略
    return module

# _find_and_load_unlocked:
_ERR_MSG_PREFIX = 'No module named '
_ERR_MSG = _ERR_MSG_PREFIX + '{!r}'

def _find_and_load_unlocked(name, import_):
    path = None
    # name="a.b.c.d", name.rpartition('.') -> ['a.b.c', '.', 'd']
    # name="a", name.rpartition('.') -> ['', '', 'a']
    parent = name.rpartition('.')[0]
    if parent:
        if parent not in sys.modules:
            # _call_with_frames_removed(import_, parent) 等价于 import_(parent),
            # 假设 name = "a.b.c", 那么会产生递归, 最终结果会按: "a", "a.b", "a.b.c" 的顺序 import
            _call_with_frames_removed(import_, parent)  
        # 有可能 parent 在导入时 import 了子包, 使得 sys.modules 中已包含这个缓存
        if name in sys.modules:
            return sys.modules[name]
        parent_module = sys.modules[parent]  # 这里 parent 必然在 sys.modules 的缓存中
        try:
            path = parent_module.__path__  # 由于 parent 有子模块 name, 所以 parent_module 必然是一个包
        except AttributeError:
            msg = (_ERR_MSG + '; {!r} is not a package').format(name, parent)
            raise ModuleNotFoundError(msg, name=name) from None
    spec = _find_spec(name, path)      # 重点!!
    if spec is None:
        raise ModuleNotFoundError(_ERR_MSG.format(name), name=name)
    else:
        module = _load_unlocked(spec)  # 重点!!
    if parent:
        # Set the module as an attribute on its parent.
        parent_module = sys.modules[parent]
        child = name.rpartition('.')[2]
        try:
            setattr(parent_module, child, module)  # 将子模块绑定到父模块的命令空间内(设置为属性)
        except AttributeError:
            msg = f"Cannot set an attribute on {parent!r} for child module {child!r}"
            _warnings.warn(msg, ImportWarning)
    return module
```

接下来, 我们继续简单看看 `_find_spec` 和 `_load_unlocked` 函数, 前者返回 `ModuleSpec`, 后者将利用这个 spec 执行导入模块的过程, 在此之前, 我们先看些基础:

```python
import sys
print(sys.meta_path)
# [
#     <_distutils_hack.DistutilsMetaFinder at 0x7fd8a409f400>,
#     _frozen_importlib.BuiltinImporter,
#     _frozen_importlib.FrozenImporter,
#     _frozen_importlib_external.PathFinder,
#     <six._SixMetaPathImporter at 0x7fd8a2f275b0>,
#     <pkg_resources.extern.VendorImporter at 0x7fd8a1a7e3d0>
# ]
for finder in sys.meta_path:
    print(finder.find_spec)  # 每一项都有 find_spec 函数
```

这里简要说明下这几个抽象类:

- `importlib.abc.Finder`:
  - `find_module(self, fullname, path=None)`: 抽象方法
- `MetaPathFinder(Finder)`:
  - `find_module(self, fullname, path=None)`: deprecated, 具体方法, 内部调用 self.find_spec
  - `invalidate_caches(self)`: 空方法 (即直接 pass)
- `PathEntryFinder(Finder)`:
  - `find_loader(self, fullname)`: deprecated, 具体方法, 内部调用 self.find_spec
  - `find_module=_bootstrap_external._find_module_shim`: 具体方法
  - `invalidate_caches(self)`: 空方法 (即直接 pass)


下面先看 `_find_spec`

```python
def _find_spec_legacy(finder, name, path):
    loader = finder.find_module(name, path)
    if loader is None:
        return None
    return spec_from_loader(name, loader)

def _find_spec(name, path, target=None):
    meta_path = sys.meta_path   # sys.meta_path 里是一些 finder
    if meta_path is None:
        raise ImportError("sys.meta_path is None, Python is likely shutting down")

    if not meta_path:
        _warnings.warn('sys.meta_path is empty', ImportWarning)

    # We check sys.modules here for the reload case.  While a passed-in target will usually indicate a reload there is no guarantee, whereas sys.modules provides one.
    is_reload = name in sys.modules
    for finder in meta_path:
        with _ImportLockContext():  # 仅仅是加锁去锁操作
            try:
                find_spec = finder.find_spec
            except AttributeError:
                spec = _find_spec_legacy(finder, name, path)
                if spec is None:
                    continue
            else:
                spec = find_spec(name, path, target)  # 重点!! 不同的 finder 都会有各自的实现, spec 里会包含 loader 信息, 也就是确定 loader 用哪个
        if spec is not None:
            # The parent import may have already imported this module.
            if not is_reload and name in sys.modules:
                module = sys.modules[name]
                try:
                    __spec__ = module.__spec__
                except AttributeError:
                    # We use the found spec since that is the one that we would have used if the parent module hadn't beaten us to the punch.
                    return spec
                else:
                    if __spec__ is None:
                        return spec
                    else:
                        return __spec__
            else:
                return spec
    else:
        return None
```

然后看 `_load_unlocked(spec)`

```python
def _load_unlocked(spec):
    # A helper for direct use by the import system.
    if spec.loader is not None:
        # Not a namespace package.
        if not hasattr(spec.loader, 'exec_module'):
            return _load_backward_compatible(spec)

    module = module_from_spec(spec)  # !!重点!!

    # This must be done before putting the module in sys.modules (otherwise an optimization shortcut in import.c becomes wrong).
    spec._initializing = True
    try:
        sys.modules[spec.name] = module
        try:
            if spec.loader is None:
                if spec.submodule_search_locations is None:
                    raise ImportError('missing loader', name=spec.name)
                # A namespace package so do nothing.
            else:
                spec.loader.exec_module(module)  # !!重点!!
        except:
            try:
                del sys.modules[spec.name]
            except KeyError:
                pass
            raise
        # Move the module to the end of sys.modules. We don't ensure that the import-related module attributes get set in the sys.modules replacement case.  Such modules are on their own.
        module = sys.modules.pop(spec.name)
        sys.modules[spec.name] = module
        _verbose_message('import {!r} # {!r}', spec.name, spec.loader)
    finally:
        spec._initializing = False

    return module
```

综上, 简单来说, `import_module` 的逻辑是:

- 先查找缓存 `sys.modules`
- 依次导入父模块 `parent_module`, 父模块的导入是递归的
- 再查找依次缓存 `sys.module`
- 按 `sys.meta_path` 里的每个 finder 逐个尝试 `find_spec`, 直至其中一个返回 ModelSpec
- 获取 ModuleSpec: `module = module_from_spec(spec)`
- 将 module 存入 `sys.modules` 缓存: `sys.modules[spec.name] = module`
- 运行 module: `spec.loader.exec_module(module)`, 包含读取 py 文件内容, 编译为 Python 字节码并执行
- 将子模块绑定至父模块下: `setattr(parent_module, child_name, module)`


### (Finish) `importlib.util.module_from_spec`

一般来说, 一个 importer 大致需要经历这几步:

```python
spec = finder.find_spec(...)  # spec 中已包含 loader
module = module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
```

注意 `finder.find_spec` 和 `spec.loader.exec_module` 才是最重要的过程, `module_from_spec` 是用 ModuleSpec 构造 ModuleType 的过程, 此过程比较直白, 没有复杂的逻辑

`importlib.util.module_from_spec` 源码 (Python 3.9.16):

```python
def module_from_spec(spec):
    """Create a module based on the provided spec."""
    # Typically loaders will not implement create_module().
    module = None
    if hasattr(spec.loader, 'create_module'):
        # If create_module() returns `None` then it means default module creation should be used.
        module = spec.loader.create_module(spec)  # 默认的 create_module 实现会返回 None, 因此可能会继续向下执行 _new_module
    elif hasattr(spec.loader, 'exec_module'):
        raise ImportError('loaders that define exec_module() must also define create_module()')
    if module is None:
        module = _new_module(spec.name)  # 也就是: module = typs.ModuleType(spec.name)
    _init_module_attrs(spec, module)  # 主要就是同步 spec.parent 与 __package__ 之类的字段
    return module

def _init_module_attrs(spec, module, *, override=False):
    if (override or getattr(module, '__name__', None) is None):
        try:
            module.__name__ = spec.name
        except AttributeError:
            pass
    # __loader__
    if override or getattr(module, '__loader__', None) is None:
        loader = spec.loader
        if loader is None:
            if spec.submodule_search_locations is not None:
                if _bootstrap_external is None:
                    raise NotImplementedError
                _NamespaceLoader = _bootstrap_external._NamespaceLoader
                loader = _NamespaceLoader.__new__(_NamespaceLoader)
                loader._path = spec.submodule_search_locations
                spec.loader = loader
                module.__file__ = None
        try:
            module.__loader__ = loader
        except AttributeError:
            pass
    # __package__
    if override or getattr(module, '__package__', None) is None:
        try:
            module.__package__ = spec.parent
        except AttributeError:
            pass
    # __spec__
    try:
        module.__spec__ = spec
    except AttributeError:
        pass
    # __path__
    if override or getattr(module, '__path__', None) is None:
        if spec.submodule_search_locations is not None:
            try:
                module.__path__ = spec.submodule_search_locations
            except AttributeError:
                pass
    # __file__/__cached__
    if spec.has_location:
        if override or getattr(module, '__file__', None) is None:
            try:
                module.__file__ = spec.origin
            except AttributeError:
                pass

        if override or getattr(module, '__cached__', None) is None:
            if spec.cached is not None:
                try:
                    module.__cached__ = spec.cached
                except AttributeError:
                    pass
    return module
```

### `importlib.reload`

### importlib 模块的总体组织

importlib 的 API 主要分为 4 块:

- 函数: 例如 `importlib.import_module`
- 抽象类 (`importlib.abc` 模块): 例如: `importlib.abc.MetaPathFinder`
- 具体实现 (`importlib.machinery` 模块): 例如: `importlib.machinery.PathFinder`
- utility (`importlib.util 模块`)

```
object
 +-- MetaPathFinder  ----------------------------------> importlib.machinery.PathFinder
 +-- PathEntryFinder ----------------------------------> importlib.machinery.FileFinder
 +-- Loader
      +-- ResourceLoader --------+
      +-- InspectLoader          |  -------------------> importlib.machinery.NamespaceLoader
           +-- ExecutionLoader --+  -------------------> importlib.machinery.ExtensionFileLoader
                                 +-- FileLoader  ------> importlib.machinery.SourcelessFileLoader
                                 +-- SourceLoader------> importlib.machinery.SourceFileLoader

importlib.__import__(name, globals=None, locals=None, fromlist=(), level=0)  # 避免使用, 且注意不是内置函数 __import__ 的别名, 请使用 import_module
importlib.import_module(name, package=None)
importlib.invalidate_caches()
importlib.reload(module)

importlib.machinery.ModuleSpec   # 要重点关注
importlib.util.cache_from_source(path, debug_override=None, *, optimization=None)
importlib.util.source_from_cache(path)
importlib.util.decode_source(source_bytes)
importlib.util.resolve_name(name, package)
importlib.util.find_spec(name, package=None)
importlib.util.module_from_spec(spec)
importlib.util.spec_from_loader(name, loader, *, origin=None, is_package=None)
importlib.util.spec_from_file_location(name, location, *, loader=None, submodule_search_locations=None)
importlib.util.source_hash(source_bytes)
importlib.util._incompatible_extension_module_restrictions(*, disable_check)
class importlib.util.LazyLoader(loader)
```

### 案例: transformers

`get_class_in_module` [源码](https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/dynamic_module_utils.py)摘录如下:

```python
import typing
from typing import Union
from pathlib import Path

HF_MODULES_CACHE = "/home/buxian/.cache/huggingface/modules"

def get_class_in_module(class_name: str, module_path: Union[str, os.PathLike]) -> typing.Type:
    """
    Import a module on the cache directory for modules and extract a class from it.

    Args:
        class_name (`str`): The name of the class to import.
        module_path (`str` or `os.PathLike`): The path to the module to import.

    Returns:
        `typing.Type`: The class looked for.
    """
    name = os.path.normpath(module_path).rstrip(".py").replace(os.path.sep, ".")
    module_spec = importlib.util.spec_from_file_location(name, location=Path(HF_MODULES_CACHE) / module_path)
    module = sys.modules.get(name)
    if module is None:
        module = importlib.util.module_from_spec(module_spec)
        # insert it into sys.modules before any loading begins
        sys.modules[name] = module
    # reload in both cases
    module_spec.loader.exec_module(module)
    return getattr(module, class_name)

class_xx = get_class_in_module(class_name="ChatGLMTokenizer", module_path = "transformers_modules/chatglm2-6b/tokenization_chatglm.py")
```

除了上述用法以外, transformers 其他地方只用到了:

- `importlib.invalidate_caches()`
- `importlib.import_module(...)`

原理: 一个简化的例子

```python
name = "a"
path = "pkg/a.py"  # 仅包含一行代码: x = 1
from importlib.machinery import SourceFileLoader
from importlib.util import spec_from_file_location, module_from_spec

spec = spec_from_file_location(name, path)  # 根据 path 确定 loader 是 SourceFileLoader
module = module_from_spec(spec)

# spec.loader.exec_module(module)

# method 1:
# code_obj = spec.loader.get_code(module.__name__)

# method 2:
# src_byte = spec.loader.get_data(path)  # 以二进制读取 "pkg/a.py"
# code_obj = spec.loader.source_to_code(src_byte, path)  # 转化为 Python 字节码

# method 3: SourceFileLoader.exec_module 的实际的主要执行过程
import _io
with _io.FileIO(path, 'r') as file:
    src_byte = file.read()
code_obj = compile(src_byte, path, 'exec', dont_inherit=True, optimize=-1)  # 编译为 Python 字节码
exec(code_obj, module.__dict__)  # 调用 exec, 将执行结果绑定到 module.__dict__ 里去
print(module.x)
```


### 案例: langchain template (重新写好写完善来)

langchain template 使用 `langchain serve` 运行时通常需要对 template 执行 `pip install -e .`, 以下是一个尝试去掉 `pip install -e .` 的做法(可能不完善), 当然最好还是使用 `pip install -e .`

```python
# serve_multiple_template.py
import importlib.util
import sys

def load_module_from_path(module_name, module_path):
    # 创建模块加载规范
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    # 创建新模块
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path.rsplit('/', 2)[1]] = module  # sys.modules["my_template"] = module
    
    # 这个能支持 __init__.py 里使用相对导入, 也就是 from .chain import chain
    module.__spec__.submodule_search_locations = [module_path.rsplit('/', 1)[0]]

    # 这个能支持 __init__.py 里使用绝对导入, 也就是 from my_template.chain import chain
    module.__spec__.__package__ = module_name
    
    # 执行模块（加载到内存）
    spec.loader.exec_module(module)
    return module

module = load_module_from_path('my_template', '/home/buxian/wsl2-test/myapp2/my-template/my_template/__init__.py')
chain = module.chain
print(chain)

# from langserve import add_routes
# ...
```

运行方式

```bash
cd /home/buxian/wsl2-test/myapp2
lanchain template new my-template
cd /path/to/serve_multiple_template.py-parent
python serve_multiple_template.py
```

说明: `langchain serve` 命令是通过读取 `pyproject.toml` 文件的如下 table 字段来获取 `uvicorn.run(app, ...)` 里 `app` 参数的, 例如在这个例子里: `app=my_template:chain`

```
[tool.langserve]
export_module = "my_template"
export_attr = "chain"
```

说明: `langchain serve` 针对 template 类型只能运行一个 template, 针对对 app 类型, 可以在 `app/serve.py` 内对 `app=FastAPI()` 使用 `langserve.add_routes(app, ...)` 添加多个 template, 这样可以同时运行多个 template.

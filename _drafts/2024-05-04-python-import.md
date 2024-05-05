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

## FAQ

所谓的 module 实际上就是 `types.ModuleType` 类型的实例, 而 package 是指带有 `__path__` 属性的 module

## import 语句与 `__import__`

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

先介绍一些简单的方法, 然后系统介绍. 官方文档[3](https://docs.python.org/3/library/importlib.html)中有一些例子介绍 importlib 的使用

### `importlib.import_module`

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

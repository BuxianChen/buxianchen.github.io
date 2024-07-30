---
layout: post
title: "(LTS) Python Internal"
date: 2024-02-18 15:10:04 +0800
labels: [python]
---

## 动机、参考资料、涉及内容

**动机**

- 对于 `pydantic` v1 至 v2 的迁移, 官方提供了一个迁移工具 `bump-pydantic`, 可以将符合 v1 版本的 python 代码转换为 v2 版本的 python 代码. 而这个工具在内部大量使用了 `libcst` 这个工具, 涉及到 python 的 AST (Abstract Syntax Tree) 与 CST (Concrete Syntax Tree)
- pytorch 2.0 以及 torch.fx 涉及到一些关于 python 字节码相关的内容
- 对编译器及 Python 虚拟机感兴趣

**参考资料**

- Python 官方文档 (待补充)
- 深入理解 Python 虚拟机: [https://nanguage.gitbook.io/inside-python-vm-cn/](https://nanguage.gitbook.io/inside-python-vm-cn/)
- 关于 Python AST 模块的补充介绍: [https://greentreesnakes.readthedocs.io/en/latest/index.html](https://greentreesnakes.readthedocs.io/en/latest/index.html)
- 这门关于机器学习系统的课程似乎也涉及一些和编译相关的内容: [https://github.com/chenzomi12/DeepLearningSystem](https://github.com/chenzomi12/DeepLearningSystem)
- cpython internal book (python 3.9): [https://realpython.com/products/cpython-internals-book/](https://realpython.com/products/cpython-internals-book/)

## 杂记

**`ast.parse`, `compile`, `exec`**

```
source code string -- `ast.parse` -> AST -- `compile` -> code object -- `exec` -> run result
```

备注: 根据 [greentreesnakes](https://greentreesnakes.readthedocs.io/en/latest/tofrom.html) 里所描述的, python 没有内置工具可以做反向的过程, 但是有一些三方工具 (这里补充一个: [depyf](https://github.com/thuml/depyf))

```
code object -> AST -> source code string
```

示例

```python
# example 1:
tree = ast.parse("print('hello world')")
# tree: <_ast.Module object at 0x9e3df6c>
code_obj = compile(tree, filename="<ast>", mode="exec")
# code_obj: <code object <module> at 0x7fe8da7f3920, file "<ast>", line 1>
exec(code_obj)  # 打印 hello world

# example 2:
def foo():
    print("foo bar")
exec(foo.__code__)  # 打印 foo bar
```

**初识 ast**

参照 [AST 官方文档](https://docs.python.org/3.12/library/ast.html) 理解

```python
ast_module = ast.parse("num+1")  # ast.Module
expr = ast_module.body[0]  # ast.Expr
expr.value                 # ast.BinOp
expr.value.left, expr.value.op, expr.value.right  # ast.Name, ast.Add, ast.Constant

expr.value.left.id, expr.value.left.ctx  # 'num', ast.Load
expr.value.right.value, expr.value.right.kind  # 1, None
```

**捕获exec执行代码的报错**

辅助函数:

```python
# tcdata/__init__.py
# Empty


# tcdata/a.py
from .b import bar

def foo(x):
    x = x ** 2
    return bar(x)

# tcdata/b.py
def bar(x):
    if x == 4:
        raise ValueError("cannot be 4")
    else:
        return 1 / (x - 4)
```

主代码:

```python
# traceback_test.py
from tcdata.a import foo
import traceback
import sys
from io import StringIO


command = """
# This is Comment-1
def test():
    # This is Comment-2
    raise ValueError("cannot test")
x = 2
print(f"x={x}")
# This is Comment-3
print(foo(x))
"""

old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

namespace = {}
namespace.update(globals())
try: 
    exec(command, namespace, None)  # 如果直接传入 globals(), 则会污染, 导致后面可以打印 x
    res = mystdout.getvalue()
    sys.stdout = old_stdout
except Exception as e:
    sys.stdout = old_stdout
    error_strs = []
    error_strs.append("Python Code (Action Input) Execution Error")

    tbs = traceback.extract_tb(e.__traceback__)
    if tbs:
        for tb in tbs:
            filename, line, func, text = tb
            if filename == "<string>":
                try:
                    command_text_line = command.split("\n")[line-1]
                    error_strs.append(f"Error occurred at line {line} of the code:")
                    error_strs.append(command_text_line)
                except:
                    pass
                break
    res = mystdout.getvalue() + "\n" + "\n".join(error_strs) + "\n" + repr(e)

# print(namespace["x"])  # OK
print(res)
# print(x)   # Error
```

输出:

```
x=2

Python Code (Action Input) Execution Error
Error occurred at line 9 of the code:
print(foo(x))
ValueError('cannot be 4')
```

可以看到, 上面的代码正确定位了 `command` 中的报错行以及内容, 并且报错前的输出也进行了保留: `x=2`, 且保留 `exec` 执行时的报错 `ValueError('cannot be 4')`

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
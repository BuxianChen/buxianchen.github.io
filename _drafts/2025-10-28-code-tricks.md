---
layout: post
title: "(LTS) Code Tricks"
date: 2025-10-28 09:05:04 +0800
---

# Python 中 deepcopy 一个函数

在 Python 中, 函数不能被 `copy.deepcopy`, 可以通过下面的 `copy_func` 来做到这一点, 在 huggingface transformers 库中, 它有如下用法:

假设有个类 Base, 现在需要从它派生出 ModelBase 和 DataBase, 但又希望它们继承到的 `push_to_hub` 的 docstring 有所不同(假设只是槽位有所不同). 这个需求可以这样做到: 继承之后通过 `copy_func` 替换原本的方法拿到各自独特的方法, 然后再对 docstring 进行修改. 

```python
import functools
import types

def copy_func(f):
    """Returns a copy of a function f."""
    # Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g

class Base:
    def push_to_hub(self):
        """class {class_name} push to hub docstring"""
        pass

class ModelBase(Base):
    pass

class DataBase(Base):
    pass

# 此时继承来的方法是完全相同的
print(id(ModelBase.push_to_hub), id(DataBase.push_to_hub))
# 输出: 2205480481184 2205480481184

ModelBase.push_to_hub = copy_func(ModelBase.push_to_hub)
if ModelBase.push_to_hub.__doc__ is not None:
    ModelBase.push_to_hub.__doc__ = ModelBase.push_to_hub.__doc__.format(class_name="ModelBase")

DataBase.push_to_hub = copy_func(DataBase.push_to_hub)
if DataBase.push_to_hub.__doc__ is not None:
    DataBase.push_to_hub.__doc__ = DataBase.push_to_hub.__doc__.format(class_name="DataBase")

# 此时原本的方法已被复制, 是不一样的方法
print(id(ModelBase.push_to_hub), id(DataBase.push_to_hub))
# 输出: 2205483486400 2205483491520
print(ModelBase.push_to_hub.__doc__)
# 输出: class ModelBase push to hub docstring
print(DataBase.push_to_hub.__doc__)
# 输出: class DataBase push to hub docstring
```


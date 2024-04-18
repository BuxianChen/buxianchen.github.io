---
layout: post
title: "(P1) 配置/数据文件格式"
date: 2024-04-18 09:45:04 +0800
labels: [config, yaml, json, xml, toml, ini]
---

## 动机、参考资料、涉及内容

- python 打包涉及到: `setup.cfg`, `pyproject.toml` 等文件格式
- yaml 文件有些高级的用法, 例如变量引用

## toml

### 语法

直接参考自 [https://toml.io/en/v1.0.0](https://toml.io/en/v1.0.0)

```toml
# 这是注释格式, 在 toml 的术语里, a 称为 table (其实就是字典类型)
a.b = "a/b"    # 转换为 json: {"a": {"b": "a/b"}}
a.c = 1.23
a.d = true

# 在 Unix 上是: "This\nis\nxxx", 在Windows上是 "This\r\nis\r\nxxx"
a.e = """This
is
xxx
"""

a.f = 'C:\Users\nodejs\templates'  # 单引号括起来的字符串不需要转义
a."g.h" = 3    # 转换为 json: {"a": {"g.h": 3}}

# "no new line"
a.i = """no \
new line
"""

# array
integers = [1, 2, 3]
colors = [
    "red",
    "yellow",
    "green"
]
nested_arrays_of_ints = [ [ 1, 2 ], [3, 4, 5] ]
nested_mixed_array = [ [ 1, 2 ], ["a", "b", "c"] ]
string_array = [ "all", 'strings', """are the same""", '''type''' ]

# Mixed-type arrays are allowed
numbers = [0.1, 0.2, 0.5, 1, 2, 5 ]
contributors = [
  "Foo Bar <foo@example.com>",
  { name = "Baz Qux", email = "bazqux@example.com", url = "https://example.com/bazqux" }
]

# table (字典, 哈希表)
# 转换为 json {"table-1": {"key1": "some string", "key2": 123}}
[table-1]
key1 = "some string"
key2 = 123

# 转换为 json {"dog": {"tater.man": {"type": {"name": "pug"}}}}
[dog."tater.man"]
type.name = "pug"

# inline table
names = { first = "Tom", last = "Preston-Werner" }


# Arrays of table: peotry.lock 里常见
[[products]]
name = "Hammer"
sku = 738594937

[[products]]  # empty table within the array

[[products]]
name = "Nail"
sku = 284758393
color = "gray"

# 以上对应于 json 是:
# {"products": [
#     {"name": "Hammer", "sku": 738594937},
#     {},
#     {"name": "Nail", "sku": 284758393, "color": "gray"}
#   ]
# }
```

一个更高阶的用法:

```toml
[[fruits]]
name = "apple"

[fruits.physical]  # subtable
color = "red"
shape = "round"

[[fruits.varieties]]  # nested array of tables
name = "red delicious"

[[fruits.varieties]]
name = "granny smith"


[[fruits]]
name = "banana"

[[fruits.varieties]]
name = "plantain"
```

对应的 json 版本

```json
{
  "fruits": [
    {
      "name": "apple",
      "physical": {
        "color": "red",
        "shape": "round"
      },
      "varieties": [
        { "name": "red delicious" },
        { "name": "granny smith" }
      ]
    },
    {
      "name": "banana",
      "varieties": [
        { "name": "plantain" }
      ]
    }
  ]
}
```

### pyproject.toml

```toml
```
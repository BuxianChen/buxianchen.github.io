---
layout: post
title:  "grep/find/sed/awk tutorial"
---

## glob 与正则表达式


## grep

正则表达式的介绍略去，需要注意的是 grep 并不支持所有的正则表达式语法，例如不能使用类似 `.*?` 进行非贪婪匹配。

## find

```
find . -type f -name a*
```

## sed

sed 命令最常见的作用是用来进行文本替换，例子如下：

```bash
echo -e "abcabc\nabcdef" | sed "s/ab/de/"  # 只匹配替换一次
echo -e "abcabc\nabcdef" | sed "s/ab/de/g"  # 尽可能多地替换
```

上述用法的一般形式为：`sed "s/pat1/pat2/`，其中 `pat1` 为将要被替换的字符串，`pat2` 为替换后的字符串。如果希望 `pat1` 与 `pat2` 为正则表达式，则需要使用 `-r` 选项，例如：

```bash
echo -e "hello world\nhello bob" | sed -r "s/hello (.*)/\1/"
```

sed 命令还有其他的作用

## awk

awk 命令的一般形式如下：

```bash
awk 'BEGIN {statements_0} pattern {statements_1} END {statements_2}' filename
```

`BEGIN {statements_0}`，`pattern`，`END {statements_2}` 均为可选项，运行原理如下：

- 如果存在 `BEGIN {statements_0}`，首先执行 `statements_0`
- 逐行执行：如果该行能与 `pattern` 匹配，则执行 `statements_1` 的内容
- 如果存在 `END {statements_2}`，则执行 `statements_2`

awk 也可接受 stdin 的输入。`statements` 里可以包含多条命令，不同的命令使用 `;` 作为分隔符。

- `pattern` 是一个筛选条件，例如：
  - `$1 == 1 && $2 ~ /^ab/` 表示满足第一项为 `1`，第二项满足正则表达式 `/^ab/` 的行，即第二项以 `ab` 为开头。
  - `NR < 5` 表示第 1 行至第 4 行；`NR==2, NR==4` 表示第二行至第四行
  - `/linux/` 表示能匹配正则表达式 `linux` 的行（此例中即为包含 `linux` 字符串的行）；`!/linux/` 表示不能匹配正则表达式 `linux` 的行
- `statements` 例子为
  - `print $1` 表示打印该行以空格作为分隔符的第一项
  - `i++` 表示变量 `i` 自加一
- awk 定义了一些内置的特殊变量可以在 `statements` 使用，例如：`NR` 表示第几行，`NF` 表示该行一共有多少项（field），`$0` 表示整行的文本内容，`$1` 表示第一项的文本内容，`$NF` 表示最后一项的文本内容，`$(NF-1)` 表示倒数第二项的内容。如果需要将分隔符进行修改，可以使用 `awk -F ,` 将分隔符定义为逗号。

### 例子

awk 可以使用如下方法引入外部变量
```bash
var1="a1"; var2="a2"
echo | awk '{print v1 ":" v2}' v1=$var1 v2=$var2
```

运行结果如下：

```text
a1:a2
```

```bash
echo -e "1,bc\n1,abcd\n2,abc" | awk -F , '$1 == 1 && $2 ~ /^ab/ {print $0}'
```

运行结果如下：

```
1,abcd
```
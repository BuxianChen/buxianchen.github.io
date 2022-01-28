---
layout: post
title:  "grep/find/sed/awk tutorial"
---

## glob 与正则表达式

glob 与正则表达式都是用来描述通用的字符串模式。然而各种 shell 命令以及编程语言对 glob 与正则表达式的支持各不相同，在写法上也存在较大差异。

### glob

具体语法可以参考[博客](http://www.ruanyifeng.com/blog/2018/09/bash-wildcards.html)。

### Simple Regular Expressions

具体使用方法参考[维基百科](https://en.wikibooks.org/wiki/Regular_Expressions/Simple_Regular_Expressions)。总结如下：

```
., [], [^ ], ^, $, ()   均为常见的含义
\n   n可以取值为1~9, 例如(ab)c\1表示匹配abcab, 此种用法不被POSIX Extended Regular Expression所接受
a*   表示匹配若干个a
[xyz]*   可以匹配xyx
\1*  例如(a.)c\1*可以匹配abcab, 但不能匹配abcac
\(xx\)*   是非法的写法
```

### POSIX Basic Regular Expression

简称为 BRE，具体使用方法参考[维基百科](https://en.wikibooks.org/wiki/Regular_Expressions/POSIX_Basic_Regular_Expressions)。主要用来向后兼容 Simple RegularExpression

```
.  用于匹配任意单个字符, 而[a.b]表示a或.或b
[]  特殊情况处理：[]abc], [abc-], [-abc]：匹配]必须放开头, 匹配-必须放开头或结尾
[^] 特殊情况处理：[^]abc], [^-abc], [^abc-]
^  匹配开头
$  匹配结尾
*
```

```
\{m\}, \{m,\}, \{m,n\}
\(\)  里面被当作是一个单一元素\(ab\)*表示匹配abab
\n    表示匹配第n个括号, 与Simple Regular Expression 兼容, 但不被ERE所使用
```

许多命令的默认情况下使用 BRE。

### POSIX Extended Regular Expression

简称为 ERE，与 BRE 的区别主要在于

```
{m}, {m,}, {m,n}
()  里面被当作是一个单一元素(ab)*表示匹配abab
\n  非法
```

### Perl Regular Expression




## grep

grep 支持的正则表达式语法可参考[维基百科](https://en.wikibooks.org/wiki/Grep)，它所支持的正则表达式包含 POSIX Basic Regular Expression。在默认情况（不加例如 `-E` 等参数时）下，支持如下写法：

```
*, ., ^, $, [], [^ ], \(\), \n, \{i\}, \{i.j\}. \{i,\}
```



## find

使用 `find -regex` 时，默认使用 Emacs Regular Expressions，但可以使用 `-regextype` 来修改这一行为。

```
find . -type f -name 'a*'  # 使用glob
```

find 命令还可以用来对找到的文件执行命令，例如：

```
find . -type d -name 'a*' -exec ls {} \;
```

> 备注：此处 `\;` 是必须的，执行逻辑是假定找到的目录名为 `ab`、`ac`，则执行
>
> ```
> ls ab;
> ls ac;
> ```
>
> 也可以使用 `+` 替换 `\;`，但此时变为
>
> ```
> ls ab ac
> ```
>
> 关于反斜杠为什么是必须的可以参照 [stackoverflow](https://stackoverflow.com/questions/20913198/why-are-the-backslash-and-semicolon-required-with-the-find-commands-exec-optio)。

## sed

sed 命令最常见的作用是文本替换，例子如下：

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

## perl

perl 是一种编程语言，功能强于 sed，引用[维基](https://en.wikibooks.org/wiki/Sed)对 sed 的介绍：

> **sed** ("**s**tream **ed**itor") is [Unix](https://en.wikibooks.org/wiki/Unix) utility for parsing and transforming text files, with ports available on a variety of operating systems. For many purposes, it has been superseded by [perl](https://en.wikibooks.org/wiki/Perl) (or the earlier [AWK](https://en.wikibooks.org/wiki/AWK)), but for simple transforms in shell scripts, sed retains some use.
---
layout: post
title: "(LTS) Jekyll 排版记录"
date: 2021-12-19 22:30:04 +0800
---

本篇博客主要用于记录 Jekyll 排版的一些样例


数学公式使用的基本逻辑是使用 [MathJax](https://github.com/mathjax/MathJax)

行内的公式: $x_1=1$

独占一行的公式:

$$
x_2=1
$$

绝对值符号使用这种写法 $\lvert x \rvert$

argmax 使用这种写法 $\arg\max_{x}{f(x)}$

帽子符号有时候会有些诡异, hat 后面不要使用花括号(不确定)? $\hat{W}ss_l$, $\hat{W}_l=W_l$, $\hat W_l=W_l$

注意如果公式中需要连续使用两个花括号时, 中间要加一个空格, 避免与 liquid 语法冲突: $\frac{ {xy}_1}{2}$

涉及到乘法, 最好使用 cdot 或者 times, 不要图省事使用普通的乘号, 它会容易与 markdown 语法的加粗/斜体混淆, 例如: **加粗文字**, $3\times x\cdot y$

MathJax 与 KaTex 都是用来在网页显示数学公式的库: 对比可参考这篇[博客](https://squidfunk.github.io/mkdocs-material/reference/math/)

分列需要使用 HTML 的语法, 并且需要借助 `kramdown` 提供的对 HTML 标签解析 `markdown="1"` 属性的功能, 才能实现. 具体的原理是使用 HTML 的 `table`, `tr`, `th`, `td` 标签建立表格实现分列, 在一个单元格 `td` 内, 使用 `div` 标签, 并配置 `markdown="1"` 使得单元格内可以使用 markdown 语法. 为了使这一解析过程生效, 需要修改 `_config.yaml` 文件:

```yaml
# _config.yaml
markdown: kramdown

kramdown:
  input: GFM
```

以下是一个例子: 注意`div`和`td`的结束标签必须顶格

<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td style="width: 100%; word-wrap: break-word; padding=5px; border: 1px solid #ccc; vertical-align: top;"><div markdown="1">
**代码段1**
```python
a = 1
b = "aaaaaaaaaaaaaaaaaaaaaaaaaa"
```
</div></td>
    <td style="width: 100%; word-wrap: break-word; padding=5px; border: 1px solid #ccc; vertical-align: top;"><div markdown="1">
**代码段2**
</div></td>
    <td style="width: 100%; word-wrap: break-word; padding=5px; border: 1px solid #ccc; vertical-align: top;"><div markdown="1">
**代码段3**
```python
a = 1
b = "sssssssssssssssssssssssssssssssssssssss"
```
</div></td>
  </tr>
</table>

作图 (svg)

<svg width="400" height="200">
    <circle cx="50" cy="50" r="20" fill="blue" />
    <circle cx="150" cy="50" r="20" fill="red" />
    <line x1="50" y1="50" x2="150" y2="50" stroke="black" stroke-width="2" />
</svg>

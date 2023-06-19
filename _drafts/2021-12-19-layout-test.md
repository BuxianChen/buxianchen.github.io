---
layout: post
title: "(WIP) Jekyll 数学公式排版记录"
date: 2021-12-19 22:30:04 +0800
---

本篇博客主要用于记录数学公式的坑, 基本逻辑是使用 [MathJax](https://github.com/mathjax/MathJax)

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
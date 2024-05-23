---
layout: post
title: "(LTS) Latex Tikz"
date: 2024-05-07 10:05:04 +0800
labels: [latex,tikz]
---

## 动机、参考资料、涉及内容

TODO: 也许后续移入新的仓库用latex重写

## 例子

```latex
\documentclass{article}
\usepackage{graphicx}
\usepackage{tikz}

\title{foo title}
\author{foo author}
\date{May 2024}
\begin{document}
\maketitle
\section{Introduction}

\begin{tikzpicture}
    \fill[yellow] (0,1) rectangle (3,2);
    \draw[step=1cm, gray, thin] (0,0) grid (3,2);
    \draw[step=1cm, gray, thin] (5,0) grid (6,4);
\end{tikzpicture}
\end{document}
```

- `tikz` 的坐标原点是在左下角, `(2, 3)` 代表的是 `x=2, y=3`, 即距离左侧 2cm, 距离下侧 3cm.
- `\fill[yellow] (0,1) rectangle (3,2);` 表示将 `(xmin=0, ymin=1)` 到 `(xmax=3, ymax=2)` 的矩形区域填充为黄色
- `\draw[step=1cm, gray, thin] (5,0) grid (6,4);` 表示从 `(xmin=5, ymin=0)` 到 `(xmax=6,xmin=4)` 画网格, 网格间隔为 1cm

以下是上面的代码运行结果示意: `E` 表示空白, `W` 表示网格, `Y` 代表黄色

```
EEEEEW
EEEEEW
YYYEEW
WWWEEW  
```
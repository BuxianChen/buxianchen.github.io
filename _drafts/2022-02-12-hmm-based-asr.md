---
layout: post
title: "(WIP)HMM-GMM Based ASR (Auto Speech Recogition)"
date: 2022-02-11 22:30:04 +0800
---
## 概述

本博客主要参考台湾大学李琳山老师的两门课程：《数字语音处理》、《数字信号处理》

声音是一个一维的信号（时间序列），根据前人的研究，将其处理为一个 $D$ 维的时间序列能更好地描述声音信号。另一方面，从文本的角度看待人发出的声音，声音是由基本单位构成的，而这些基本单位将组成一个一个的词。HMM-GMM 做语音识别的思路是，为每个声音的基本单位建立一个 HMM-GMM 模型。声音生成的过程为：每个基本单位的声音信号一小段 $D$ 维的时间序列。

## HMM

## 声学模型

人发出声音的过程为：由肺部产生一股一股的“气”，经由唇齿舌的作用下，产生声音。


## 附录 A：傅里叶变换

### A.1 信号（待补充）
离散情形下，如下信号被称为激冲信号：

$$
\delta(n)=\left\{
\begin{aligned}
1&\quad n=1\\
0&\quad others
\end{aligned}
\right.
$$

平行地，连续情形下，如下信号被称为激冲信号

### A.2 线性时不变系统

在信号理论中，一个“系统”指的是输入一个信号，输出也是一个信号的东西。而线性时不变系统指的是满足如下条件的系统：

离散情况（信号的定义域为整数域 $\mathbb{Z}$，值域为复数域 $\mathbb{C}$）：
- 对于任意的 $k\in\mathbb{Z}$ 以及任意的信号 $x(n)$，假定该系统将信号 $x(n)$ 变换至 $y(n)$，那么该系统会将 $\tilde{x}(n):=x(n-k)$ 变换到 $\tilde{y}(n):=y(n-k)$；
- 对于任意的信号 $x_1(n), x_2(n)$，假定该系统会将其分别变换至 $y_1(n), y_2(n)$。那么对于任意的实数(复数) $a_1, a_2$，该系统会将 $\tilde{x}(n):=a_1x_1(n)+a_2x_2(n)$ 变换到 $\tilde{y}(n):=a_1y_2(n)+a_2y_2(n)$

连续情况（信号的定义域为实数域 $\mathbb{R}$，值域为复数域 $\mathbb{C}$）：
- 对于任意的 $t_0\in\mathbb{R}$ 以及任意的信号 $x(t)$，假定该系统将信号 $x(t)$ 变换至 $y(t)$，那么该系统会将 $\tilde{x}(t):=x(t-t_0)$ 变换到 $\tilde{y}(t):=y(t-t_0)$；
- 对于任意的信号 $x_1(t), x_2(t)$，假定该系统会将其分别变换至 $y_1(t), y_2(t)$。那么对于任意的实数(复数) $a_1, a_2$，该系统会将 $\tilde{x}(t):=a_1x_1(t)+a_2x_2(t)$ 变换到 $\tilde{y}(t):=a_1y_2(t)+a_2y_2(t)$

可以证明，线性时不变系统一定为下述形式：

对于离散情况，系统对于 $\delta(n)$ 的输出假设为 $h(n)$，那么对于任意的信号 $x(n)$，输出信号为：
$$
y(n)=\sum_{k=-\infty}^{\infty}{x(k)h(n-k)}=x*h
$$

上述 $x*h$ 被称为离散卷积。

对于连续情况，系统对于 $\delta(t)$ 的输出假设为 $h(t)$，那么对于任意的信号 $x(t)$，输出信号为：
$$
y(t)=\int_{-\infty}^{\infty}{x(\tau)h(t-\tau)d\tau}=x*h
$$
上述 $x*h$ 被称为连续卷积。

对于线性时不变系统而言，如下信号是特别的：

离散情形下，对于任意的 $z\in\mathbb{C}$：

$$x(n)=z^n$$

其响应函数为：
$$
\begin{aligned}
y(n)&=\sum_{k=-\infty}^{\infty}{x(n-k)h(k)}\\
&=\sum_{k=-\infty}^{\infty}{z^{n-k}h(k)}\\
&=z^n\sum_{k=-\infty}^{\infty}{z^{-k}h(k)}\\
&:=H(z)x(n)
\end{aligned}
$$

即上述形式的输入信号为“特征”信号，即输出信号与输入信号只差一个常数倍

连续情形下，对于任意的 $s\in\mathbb{C}$，

$$
x(t)=e^{st}
$$

其响应函数为：

$$
\begin{aligned}
y(t)&=\int_{-\infty}^{\infty}{x(t-\tau)h(\tau)d\tau}\\
&=\int_{-\infty}^{\infty}{e^{s(t-\tau)}h(\tau)d\tau}\\
&=x(t)\int_{-\infty}^{\infty}{e^{-s\tau}h(\tau)d\tau}\\
&:=H(s)x(t)
\end{aligned}
$$

### A.3 连续周期信号的傅里叶级数


### A.4 离散周期信号的傅里叶级数

### A.5 连续信号的傅里叶变换

### A.6 离散信号的傅里叶变换

### 


## 附录 B：EM 算法

## 附录 C：CART 决策树

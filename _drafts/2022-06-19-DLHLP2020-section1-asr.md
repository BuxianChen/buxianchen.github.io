---
layout: post
title: "(WIP)DLHLP2020学习笔记1——语音识别"
date: 2022-06-19 22:31:04 +0800
---

<style>
hidden_block{
    color:red;
}
</style>

## 动机、参考资料、涉及内容

参考资料：

- 人类语言处理2020-李宏毅（语音识别部分、HMM based不详细做记录）

涉及内容

- 可能会增加最新（2020年之后）的一些端到端技术（不确定记不记得来）

---

## ASR的输入与输出
语音识别（Automatic Speech Recognition）是指将语音信号转为文字。在这个任务中，输入为语音信号，其呈现形式为一个一维的时间序列，存在一系列的声音信号前处理方式将其转换为一个 $(T, D)$ 的多维时间序列信号。在输出端，也存在多种选择：phoneme/grapheme/word/morpheme，而这些概念在不同的语种中可能有的适用有的不适用。

<details>
<summary>
声音信号及其前处理
</summary>
</details>


<details>
<summary>
<hidden_block>
语音识别的输出
</hidden_block>
</summary>
<h3>Part 1：端到端语音识别</h3>

<h3>Part 2：基于HMM的方法</h3>
</details>


## LAS

[LAS]() 是 xx 年的一篇论文，其结构上几乎就是 seq-to-seq model。


<details>
<summary>
<hidden_block>
seq-to-seq model
</hidden_block>
</summary>
举一个用 seq-to-seq 模型做翻译模型的例子：任务为将 A 语言翻译为 B 语言，以下用大写字母表示 A 语言的 token，用小写字母表示 B 语言的 token

假定词表分别为：${A_1,...,A_m,<BOS_A>,<EOS_A>}$，${B_1,...,B_n,<BOS_B>,<EOS_B>}$，现在有一个输入序列为：$(A_3, A_100, A_1)$，期望得到输出序列。具体计算过程为：

首先将


demo：

假定 A 语言的词表为大写字母A-F，B 语言为小写字母a-f。真实的翻译规则为：

- 若带翻译句子首尾两个字符为同一个字符，则翻译过程将此字符忽略，且此过程往复进行
- 给定如下词表具有对应关系
```
AB -> bc
...
DE -> ef
A -> a
...
F -> f
```
- 若出现如下词表则发生倒装
```
BC
DE
```
- 其余情况均按大小写对应关系进行翻译




</details>

## CTC



## RNN-T
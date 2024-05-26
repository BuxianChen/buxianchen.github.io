---
layout: post
title: "(P1) GitHub Action 简介"
date: 2024-05-25 10:05:04 +0800
labels: [github]
---

## 动机、参考资料、涉及内容

## `.github/workflows`

每个 yaml 文件都是一个独立的工作流, 独立触发互不干扰, 但是每个工作流内部的 jobs 是可以有依赖关系的, 每个 jobs 的 steps 是按顺序执行的.

## How To

### 怎样设置分支的 push 权限

问题描述: 只希望某个特定分支能被少数人直接 push, 其他人只能通过 PR 请求

解决方案: 去项目仓库 (只有管理员或拥有者有权限) 设置: `Branches` -> `Add branch protection rule` -> ...

### 怎样对 push/PR 做检查

问题描述: 只有当代码符合代码规范时才允许 push 或 merge

解决方案:

首先需要配置相应的 GitHub Action, 也就是在 `.github/workflows` 下面增加一个配置文件(文件内容是检查代码规范相关的), 注意将触发条件 `on` 设置为类似如下:

```yaml
on:
  push:
    branchs:
      - main
  pull_request:
```

然后去项目仓库 (只有管理员或拥有者有权限) 设置: `Branches` -> `Add branch protection rule` -> 填好分支名 -> `Require status checks to pass before merging`, 应该可以进一步只将某些 job 设置为 Required.

解释:

在提 PR 或者对 PR 更新代码时, 对应的事件都是 `pull_request`. 当直接往代码分支推送代码时, 对应的事件是 `push`. 而在 GitHub Action 中定义的每个 Action (即每一个 yaml 配置文件), 一旦被触发执行, 每个 job 都会产生一个 `status checks` (一个 Action 由多个 jobs 构成, 每个 `status checks` 代表一个 job 是否正确执行完成, 例如配置的所有 shell 脚本是否都以返回 0 退出). 而由于分支设置了需要检查这些状态才能被合并, 所以也就实现了我们检查代码规范的需求

`status checks` 主要有成功, 失败, 无权限执行等状态

## 附录

关于 yaml 文件多行字符串的语法: [https://yaml-multiline.info/](https://yaml-multiline.info/)

yaml 语法中关于多行的字符串有很多的写法, 例如像下面这种写法, 会被转换为 `ls\nmkdir app`

```yaml
# ls\nmkdir app
run1: |
  ls
  mkdir app
# ls mkdir app
run2: >
  ls
  mkdir app
```

相关的变体还有 `|+`, `|-`, 以及 `>`, `>+`, `>-`: `|` 表示保留换行符, `>` 表示将换行符替换为空格, 而 `+`, `-` 只在于对结尾是否保留换行符有些小差异
---
layout: post
title:  "Jekyll tutorial"
---

# Jekyll tutorial

简单来说，Jekyll 是一款能将标记语言文本（例如 markdown 文件）转换为静态网页的软件。本博客系列即是借助 Github Pages 这一平台，使用 Jekyll 来完成的。仅在此记录一下初次接触时的笔记。

## 安装

首先需要理清相关的概念。Ruby 是一种脚本语言，而 Jekyll 是基于该语言编写的一个包，因此要运行 Jekyll 需要有一个 Ruby 的解释器，并且安装好相关的依赖。

### Ruby 的包及包管理工具

Ruby 语言中包的呈现形式一般会是一个 `.gem` 文件，RubyGems 是安装 Ruby 时自带的一个包管理工具，用于将源代码打包为 `.gem` 文件，在 shell 中使用 `gem` 命令即为 RubyGems 工具。

### Rails
Rails 是一套基于 Ruby 的著名开发框架，Jekyll 也是基于此框架编写的。Rails 使用一个叫做 Bundle 的包管理工具，可以简单视为对原生 RubyGems 的一层封装，在使用 Bundle 进行包管理时，可以将一个项目的全部依赖关系写入到一个文本文件中（默认文件名为 Gemfile），这样就可以方便地使用 Bundle 进行依赖包的安装了n

### Ruby 类比 Python
|Ruby|Python|说明|
|---|---|---|
|Ruby|Python|都是脚本语言|
|RubyGems、Bundle|pip|包管理工具|
|Gemfile|requirements.txt|依赖包列表|
|Rails|Flask|Rails 是基于 Ruby 的框架，Flask 是基于 Python 的框架|

### 安装过程
参照 [jekyll 官网](https://jekyllrb.com/docs/installation/) 的安装步骤即可。

### GitHub Pages

在 Jekyll 安装好后，可以简单地使用如下命令生成一个 “HelloWorld” 静态网页。

```
# 新建一个名为 my-awesome-site 文件夹，并在这里面生成了一些文件
jekyll new my-awesome-site
cd my-awesome-site
bundle exec jekyll serve
# 使用浏览器打开 http://localhost:4000 即可看到网页内容
```

因此，至此为止，已经可以制作网页了，并且可以进行本地的浏览。如果有自己的服务器和域名的话，就可以让其他人也看到了，如果自己没有服务器或者域名的话，GitHub 网站的 Github Pages 功能则相当于提供了一个免费的服务器及域名。为了做到这一点，首先需要有一个 GitHub 账号，假定账号名为 `foo`，即进入自己主页后，其域名为 `https://github.com/foo`。之后，需要新建一个名为 `foo.github.io` 的仓库，之后将前面的 `my-awesome-site` 文件夹下的所有文件直接拷贝至这个仓库中，将代码提交到 Github 后，等待几分钟后，就可以用浏览器打开 `https://foo.github.io`，就可以浏览到生成的网页了。

## Jekyll 项目的目录结构（未完待续）

```
_drafts/
  - xxx.md
  - yyy.md
```
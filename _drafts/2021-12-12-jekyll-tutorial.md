---
layout: post
title: "(LTS) Jekyll tutorial"
date: 2021-12-12 22:30:04 +0800
---
# Jekyll tutorial

简单来说，Jekyll 是一款能将标记语言文本（例如 markdown 文件）转换为静态网页的软件。本博客系列即是借助 Github Pages 这一平台，使用 Jekyll 来完成的。仅在此记录一下初次接触时的笔记。

## 安装

首先需要理清相关的概念。Ruby 是一种脚本语言，而 Jekyll 是基于该语言编写的一个包，因此要运行 Jekyll 需要有一个 Ruby 的解释器，并且安装好相关的依赖。

### Ruby 的包及包管理工具

Ruby 语言中包的呈现形式一般会是一个 `.gem` 文件，RubyGems 是安装 Ruby 时自带的一个包管理工具，用于将源代码打包为 `.gem` 文件，在 shell 中使用 `gem` 命令即为 RubyGems 工具。

### Rails
Rails 是一套基于 Ruby 的著名开发框架，Jekyll 也是基于此框架编写的。Rails 使用一个叫做 Bundle 的包管理工具，可以简单视为对原生 RubyGems 的一层封装，在使用 Bundle 进行包管理时，可以将一个项目的全部依赖关系写入到一个文本文件中（默认文件名为 Gemfile），这样就可以方便地使用 Bundle 进行依赖包的安装了。

### Ruby 类比 Python
<table>
    <tr> <td>Ruby</td> <td>Python</td> <td>说明</td> </tr>
    <tr> <td>Ruby</td> <td>Python</td> <td>都是脚本语言</td> </tr>
    <tr> <td>RubyGems、Bundle</td> <td>pip</td> <td>包管理工具</td> </tr>
	<tr> <td>Gemfile</td> <td>requirements.txt</td> <td>依赖包列表</td> </tr>
    <tr> <td>Rails</td> <td>Flask</td> <td>Rails 是基于 Ruby 的框架，Flask 是基于 Python 的框架</td> </tr>
</table>

### 安装过程

方法一 (WSL2 上强烈不推荐): 参照 [jekyll 官网](https://jekyllrb.com/docs/installation/) 的安装步骤。

方法二 (推荐): 避免安装系统级 Ruby

参考:

- 不使用 apt-get: [stackoverflow](https://stackoverflow.com/questions/75452016/installation-messed-up-with-ruby-unable-to-install-jekyll), [https://dontusesystemruby.com/#/](https://dontusesystemruby.com/#/)
- RVM: [http://rvm.io/](http://rvm.io/)

RVM 应该相当于是 anaconda, 可以安装多个版本的 Ruby.

```bash
# 安装 RVM: http://rvm.io/
gpg2 --keyserver keyserver.ubuntu.com --recv-keys 409B6B1796C275462A1703113804BB82D39DC0E3 7D2BAF1CF37B13E2069D6956105BD0E739499BDB
curl -sSL https://get.rvm.io | bash -s stable

# 参考: https://stackoverflow.com/questions/75452016/installation-messed-up-with-ruby-unable-to-install-jekyll
rvm install 2.7
rvm use 2.7.2 --default
rvm -v
rvm gemset update
gem install jekyll -v 4.2.1
jekyll -v
cd /path/to/username.github.io

# 根据 Gemfile 安装依赖与运行
bundle install
bundle exec jekyll serve
```

gem 换源:

```bash
gem sources --add https://gems.ruby-china.com/ --remove https://rubygems.org/
gem sources -l
```

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

## Jekyll 项目的目录结构

```
_data/
_includes/
_layout/
_post/
_asset/
_sass/
_site/
_config.html
```

- `_site`：Jekyll 转换后的结果，即 `bundle exec jekyll serve` 命令的输出

- `_data` 文件夹：用于添加全局变量，例如在 `_data` 文件夹下建立一个名为 `navigation.yml` 的文件，那么 `site.data.navigation` 就指代的是这个文件里的数据，可以在 Liquid 模板语言中被引用

- `_includes` 文件夹：例如建立一个名为 `navigation.html` 的文件，那么它可以被 `_layout` 中的例如 `default.html` 用如下方式引入
{% raw %}
  ```text
  {% include navigation.html %}
  ```
{% endraw %}
- `_layout` 文件夹下的文件 `default.html`，可以使用在其他文件中，只要开头包含

  ```
  ---
  layout: default
  ---
  ```

  例如 `_post/2018-08-20-bananas.md` 的文件内容除了上述三行为

  ```markdown
  aaa
  bbb
  ```

  那么经过 Jekyll 转换后的 `_site/2018/08/20/bananas.html` 会是 `_layout/default.html` 将 `content` 替换为上述两行的结果，即自动转为

  ```html
  <p>aaa</p>
  <p>bbb</p>
  ```

- `_sass`：并非必要，例如 `_sass/main.scss` 可以被 `assets` 下的文件 `assets/css/styles.scss` 使用如下方式引入

  ```
  ---
  ---
  @import "main";
  ```

- `assets`：目录结构固定为

  ```
  assets/
    - css/
    - images/
    - js/
  ```

  在 `_site` 中体现在 `_site/assets` 文件夹中

- `_posts`：文件命名固定为 `YYYY-MM-DD-title.{ext}`，例如 `_post/2010-09-03-bananas.md`。最终由 Jekyll 生成的 html 文件路径为 `_site/YYYY/MM/DD/title.html`。

- 主目录下的 `index.html` 将会被映射为网站的 `/` 目录，映射关系如下

  ```
  ROOT
    - index.html  # -> _site/index.html -> ip:port/
    - about.html  # -> _site/index.html -> ip:port/about
    - hello.html  # -> _site/index.html -> ip:port/hello
  ```

## 预备知识：Liquid 模板语言

[Liquid 官方文档](http://shopify.github.io/liquid/)

### layout 的继承关系

`_layout/default.html`

```html
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>{{ page.title }}</title>
  </head>
  <body>
    {{ content }}
  </body>
</html>
```

`_layout/post.html`

```html
---
layout: default
---
<h1>{{ page.title }}</h1>
<p>{{ page.date | date_to_string }} - {{ page.author }}</p>

{{ content }}
```

`_post/2018-08-20-bananas.md`

```
---
layout: post
author: jill
---
first paragraph

second paragraph
```

生成的 `_post/2018/08/20/bananas.html` 文件内容如下：

```html
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Bananas</title>
  </head>
  <body>
    <h1>Bananas</h1>
    <p>20 Aug 2018 - jill</p>
    <p>first paragraph</p>
    <p>second paragraph</p>
  </body>
</html>
```

解释：继承关系使得 `_layout/post.html` 实际被替换为

```html
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>{{ page.title }}</title>
  </head>
  <body>
    <!--由于继承关系, 将post的实际内容替换掉default.html的{{content}}-->
    <h1>{{ page.title }}</h1>
	<p>{{ page.date | date_to_string }} - {{ page.author }}</p>
	{{ content }}
  </body>
</html>
```

## 附录：搭建 http 服务

最简单的操作方式如下，用虚拟机（ubuntu 18.04）做实验。

准备工作：

- 宿主机与虚拟机网络互通：参见 [CSDN 博客](https://blog.csdn.net/bifengmiaozhuan/article/details/79888516)

- 查询宿主机与虚拟机 IP 的命令如下：

  ```shell
  # 宿主机 Windows 10
  ipconfig  # 假定为 192.168.1.105
  # 虚拟机 Ubuntu 18.04
  hostname -I  # 假定为 192.168.1.102
  ```

操作过程参考 [CSDN 博客](https://blog.csdn.net/qq_38240926/article/details/99610158)。简述如下，在虚拟机中执行

```bash
sudo apt update
sudo apt install apache2
```

之后使用 `sudo` 将 `_site` 中的内容复制到 `/var/www/html` 目录下。之后即可在宿主机的浏览器通过访问虚拟机 IP 来访问网页，例如：

```
http://192.168.1.102
http://192.168.1.102/a
```


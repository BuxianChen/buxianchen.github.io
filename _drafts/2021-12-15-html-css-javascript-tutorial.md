---
layout: post
title: "(WIP)html/css/javascript tutorial"
date: 2021-12-15 22:30:04 +0800
---

{:toc}

## 概述

html 是一种标记语言，换句话说，是一种按照某种格式来书写的文本，这种格式可以被浏览器所解析。css 是对 html 的补充，用于规定文本的渲染方式，即呈现出来的视觉效果，如果不使用 css，html 可以直接对每个元素设定 `style` 来做到这一点，但将样式独立为一个单独的文件可以将内容与显示格式分离，起到解耦的效果，而 css 文件本身也不过是一种配置文件，用于指定显示格式。javascript 是对 html 的补充，使得用户可以对页面进行操作与交互。可以将 javascript 的内容直接写在 `html` 的 `script` 标签中，但与 css 类似，将其独立为一些单独的文件便于维护，而 javascript 本身是一种编程语言。

注释方式：

- html：`<!-- comment -->`
- css：`/* comment */`
- javascript：`// comment`

## HTML

### tutorial

```html
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <!-- This is comment -->
    <title> This is page's name</title>
  </head>
  <body>
      <h1>This is title showed in the page</h1>
      <p>This is a paragraph</p>
  </body>
</html>
```

规范的写法里没有内容的元素须使用 `<{tagname} />` 格式，例如：`<br />` 表示换行。标签的属性值必须用双引号括起来。

```html
<{tagname}>{content}</{tagname}>
<{tagname} />
<{tagname} {attrname1}="xxx" {attrname2}="yyy">{content}</{tagname}>
```

```
<img loading="lazy" src="/images/logo.png" width="258" height="39" />
```

通用的几个属性有：

- class：用于css中，一个元素的class标签中可以填入多个
- id：元素的唯一id
- style：指定文本样式，例如文字颜色
- title

标签罗列

- `<br />` 表示换行，`<hr />` 表示画一条水平线

- `h1` - `h6` 表示标题，`p` 表示段落，`a` 表示超链接，`cite` 表示引用

- `b` 表示加粗，`i` 表示斜体，`sup` 表示上标，`sub` 表示下标，`del` 表示删除字，`ins` 表示在文字加上底线

- `code`、`kbd`、`samp`、`var`、`tt` 这几个标签常用于显示计算机相关的文本

- `<abbr title="World Wide Web">WWW</abbr>`，鼠标悬停时会显示 title 中的内容

- 样式

  ```html
  <head>
  
  <!-- 直接使用style标签进行定义 -->
  <style type="text/css">
  	body {background-color:yellow}
  	p {color:blue}
  </style>
  
  <!-- 使用css文件 -->
  <link rel="stylesheet" type="text/css" href="mystyle.css">
  </head>
  ```

- `div` 为块级容器，`span` 为内联容器

注意事项

- HTML 源代码中多个连续的空格或空行都将被视为一个空格

### 代码规范

参考：[菜鸟教程](https://www.runoob.com/html/html5-syntax.html)

### Liquid

## CSS

### tutorial

放在 html 的 head 标签中

```css
<style type="text/css">
p {color:blue}
#para
{
	text-align:center;
    color:red;
}
/*comment*/
.center {text-align:center;}
p.center {text-align:center;}
</style>
```

`#para` 表示 html 标签中元素 `id` 属性为 `para` 的标签内容采用此样式

`.center` 表示 html 标签中元素 `class` 属性为 `center` 的标签内容采用此样式

`p` 表示 html 的 `p` 标签采用此样式

`p.center` 表示 html 的 `p` 标签且 `class` 属性为 `center` 的标签采用此样式

目录结构如下

```
ROOT/
  - main.html
  - style.css
```

`main.html` 文件内容如下

```html
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>abc</title>
    <link rel="stylesheet" type="text/css" href="styles.css" />
  </head>
  <body>
     <a href="/" class="current"> 111 </a>
  </body>
</html>
```

`style.css` 文件内容如下

```css
.current {
	color: red;
}
```

实际作用等效于 `main.html` 中的 `a` 标签变为

```html
<a href="/" style="color: red;"> 111 </a>
```

### SASS

由于 css 文件中可能会存在许多冗余，例如同样的样式对于多个标签具有公共性，这样需要做统一修改时会变得麻烦且容易出错，sass 可以解决这一问题。具体的做法是：sass 定义了一套”语言“，而 sass 是一个转换工具，可以将 sass 这套语言转换为标准的 css 文件。引用 sass 官方的介绍词：

> Sass is the most mature, stable, and powerful professional grade CSS extension language in the world.

#### hello world

sass 工具的输入是一个 scss 文件，输出是一个 css 文件，例如：

`test.scss` 的文件内容如下

```
$font-stack: Helvetica, sans-serif;
$primary-color: #333;

body {
  font: 100% $font-stack;
  color: $primary-color;
}
```

执行命令：`sass test.scss test.css`，将生成转换后的 `test.css` 文件。

```
body {
  font: 100% Helvetica, sans-serif;
  color: #333;
}
```

备注：旧版本定义的文件格式为 `.sass` 文件，与 `.scss` 文件定义的语法格式区别不大，主要区别在于 scss 文件使用了花括号与分号进行代码块的限定（有点像 C 语言风格），而 sass 文件则依赖于缩进（有点像 Python 语言风格）。就目前来说，推荐使用 scss 文件格式，因此转换命令会稍显诡异，例如：`sass xxx.scss xxx.css`。

#### tutorial

[Sass 官网](https://sass-lang.com/guide)

## JavaScript

### tutorial

```html
<!DOCTYPE html>
<html>
    <head> 
        <meta charset="utf-8"> 
        <title>菜鸟教程(runoob.com)</title> 
    </head>
    <body>

    	<h1>我的第一段 JavaScript</h1>
    	<p id="demo">
    		JavaScript 能改变 HTML 元素的内容。
    	</p>
        <script>
		// 注释
        function myFunction()
        {
            x=document.getElementById("demo");  // 找到元素
            x.innerHTML="Hello JavaScript!";    // 改变内容
        }
        </script>
        <button type="button" onclick="myFunction()">点击这里</button>
    </body>
</html>
```

javascript 指的是 `script` 标签中的内容，上例中 `botton` 标签相当于是调用者，当用户点击时，这个事件对应于 `botton` 的 `onclick` 属性，此时触发 `onclick` 的属性内容 `myFunction()`，调用 `script` 中定义的函数。

`script` 标签的位置：参考 [CSDN](https://www.cnblogs.com/xiangkejin/p/6411792.html)。


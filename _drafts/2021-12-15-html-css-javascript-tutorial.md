---
layout: post
title: "(LTS) html/css/javascript tutorial"
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
- title: 鼠标悬停显示 title 的属性值

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

### Liquid【待补充】

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

### 基本概念及参考资料

**长度单位**

我们看到的屏幕实际上的最小单位是像素, 例如一块分辨率为 2560x1440 的屏幕 (2K 屏幕), 指的是长与高方向被分为 2560 和 1440 小段, 共 2560x1440 个小色块. 而屏幕的实际大小譬如说长度为 54cm, 宽度为 30cm. 顺带补充一点, 平时买显示器所说的尺寸例如 24 寸指的是对角线的英寸数, 例如这个例子里按勾股定理算出的对角线长度就约为 24.3 英寸. 由于不同设备的一个像素点的实际尺寸不一样 (实际上像素点不一定是正方形, 也可以是矩形, 甚至可以是六边形等), 因此产生一个概念叫 dpi (dots per inch), 指一英寸长度上的像素点个数, 拿前面的例子来说, 在长度方向上, 实际长度为 54cm x 0.3937inch/cm = 21.26inch, 因此 dpi 为 2560 / 21.26 = 120.4

CSS 中的默认长度单位为像素 (px), 长度单位一般分为绝对长度单位与相对长度单位:

- 绝对长度单位: 这种长度单位进行度量是在任何设备上看都是一样的, cm, in
- 像素: px, 这个实际上是相对于屏幕的, 例如在 A 设备上像素点间距是 0.02cm, B 设备上是 0.01cm, 那么 12px 呈现出来的大小分别是 0.24cm 和 0.12 cm.
- 相对长度单位:
  - em: 假设当前字体是 12px, 那么 0.5em 就是 12px


**字体**

我们平时说的“字体”在广义上包含如下几个相互“垂直”的方面：

- 字体类别 (Font Family): 有如下几种通用字体族, 在写 CSS 时, 字体一般会设置多个备选, 一般最后会以这几种通用字体族结尾, 代表使用系统上默认的该字体族字体进行显示.
  - 衬线字体 (Serif), 例如: Times New Roman, Georgia
  - 无衬线字体 (Sans-serif), 例如: Arial
  - 等宽字体 (Monospace), 例如: Courier New, Lucida Console
  - 草书字体 (Cursive), 例如: Brush Script MT
  - 幻想字体 (Fantasy), 例如: Copperplate
- 样式 (Style): 包括正常 (normal) 与斜体 (italic)
- 粗细 (Weight): 包括正常 (normal, 即: 400)、粗体 (bold, 即: 700)、加粗 (bolder), 细体 (lighter). 用数字精确衡量的化, 字体粗细的取值范围一般是 1~1000 之间
- 大小/字号 (Font Size): 一般使用像素 (px) 来衡量, 一些预设的字号与像素的对应关系例如: 小四 (12px), 注意: 这里的大小实际上指的是字的高度, 我们平时所说的 2K 显示屏实际上指的是屏幕的长与高分别为 2560 和 1440 个像素 (2560x1440), 所以如果将字体设置为 20px, 且行间距设置为 0, 理论上应该能恰好看到 1440/20=72 行文字
- 颜色 (Color): 一般有几种表示方式: 
  - 预设的名称: red, green 等
  - RGB 值: 例如: `#FF0000` 从左到右两位一组, 分别表示红绿蓝的亮度值
  - RGBA 值: 另外增加一个 0~1 的值表示不透明度, 也就是值越小越透明

**行距**

字号 20px, 指的是在不包含任何顶端空白的情况下是 20px (即使在设置 1 倍行距时, 字的上下端实际上还是会留下一些空白), 假设实际上一行字真正占用 30px, 即上下各留 5px 空白, 于是 30px 代表着 1 倍行距, 现在将行距设置为 1.8 指的是, 总高度为 30x1.8=54px, 字占的高度为 20px, 上下各留 17px. 在使用鼠标选中高亮这一行字时, 一般是会选中总共的 54px (如果行间距小于 1, 那么鼠标选中一行仍然是高亮 30 px 的范围)

**CSS 框模型**

所有 HTML 元素都可以视为方框。在 CSS 中，在谈论设计和布局时，会使用术语“盒模型”或“框模型”。

CSS 框模型实质上是一个包围每个 HTML 元素的框。它包括：外边距、边框、内边距以及实际的内容。

注意: 上下边框在某些情况下会有合并规则

需要区分这几个概念: **元素高宽, 内边距, 边框, 轮廓, 外边距**

下图是两个紧邻的元素 (不考虑外边框合并规则), 以下面的元素为例, 即从内到外共 4 个矩形, 其中最内部的矩形为元素的实际内容, 其高和宽通过 (width/height) 来设定, 往外一层是内边距 (padding), 内边距所代表的矩形框是边框 border, 再外面一层是外边距 (margin), 再外面一层是轮廓 (outline), 注意:

- 相邻元素的外边 (margin) 相互紧挨着, 无视各自的轮廓 (outline)
- 除边框外, 其余的矩形框线实际上都是虚拟的, 显示时并不存在, 并且边框线的粗细是其自身的, 不含在内外边距里, 也就是说一个元素的总高度这么计算: 总高度=上外边距+上边框高度+上内边距+元素高度+下内边距+下边框高度+下外边框
- 实际上轮廓还能有偏移效果: 即不是在最外的矩形与第二外面的矩形间做填充颜色之类的效果, 而是可以在最外的矩形之外添加效果

![](../assets/figures/web/css-padding-border-margin-outline.png)

参考资料: [https://www.w3school.com.cn/css/css_boxmodel.asp](https://www.w3school.com.cn/css/css_boxmodel.asp)

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


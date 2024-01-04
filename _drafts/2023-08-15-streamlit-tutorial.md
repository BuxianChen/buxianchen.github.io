---
layout: post
title: "(P1) Streamlit Tutorial"
date: 2023-08-15 10:00:04 +0800
labels: [web]
---

## 动机、参考资料、涉及内容


## 记录

能制作的网页

- 机器学习模型简单演示: 文本/图像/语音分类, 对话机器人
- 标注平台

执行逻辑

脚本从上到下运行, 进行组件渲染. 当用户与页面组件(例如按钮,滚动条)交互时, 整个脚本会从上到下重新运行 (rerun). 一些函数调用可以加上装饰器进行缓存(当输入以及函数代码不发生变化时,不重复运行函数直接取缓存结果). 为了解决重新运行导致状态丢失的问题, 可以使用 `streamlit.session_state`, 在一个 session 中 (即浏览器的一个标签页), 重新运行时会共享 `streamlit.session_state`. 如果需要拆分页面, 则创建 `pages` 文件夹, `pages` 文件夹每一个 python 文件是一个页面, 写法与主页面相同, 所有页面共享一个 `streamlit.session_state`.

组件

详细 API 文档可直接参考: [https://docs.streamlit.io/library/api-reference](https://docs.streamlit.io/library/api-reference)

```python
# 普通文本

# 设置字体颜色

# 富文本(html/markdown)

# latex

# 侧边栏

# 表格

# 图形

# 图像

# 语音

# 视频

# 组件: 按钮

# 组件: 单选框

# 组件: 多选框

# 组件: 输入框

# 区域拆分

# 区域拆分: 可最小化隐藏与展开区域

# 标签页(页面内)

# page

# 可输入表格, 且带组件
```

`button`, `checkbox`, `radio`, `multiselect` 在交互时的区别

- `button` 在被点击时, 状态由 `False` 变为 `True`, 如果再进行点击, 状态维持为 `True`. 如果此时与其他组件进行交互, 发生 rerun, `button` 会恢复为 `False` 的状态
- `checkbox` 每次被点击时, 状态在 `True` 和 `False` 中切换, 与其他组件进行交互时, 发生 rerun, `checkbox` 的状态会维持不变
- `radio`: 与其他组件进行交互时, 发生 rerun, `radio` 的状态维持不变
- `multiselect`: 与其他组件进行交互时, 发生 rerun, `multiselect` 的状态维持不变

验证代码如下:

```python
import streamlit as st

st.button("button", key="button")
st.checkbox("checkbox", key="checkbox")
st.radio("radio", ["选项A", "选项B", "选项C"], key="radio")
st.multiselect("multiselect", ["选项A", "选项B", "选项C"], key="multiselect")

s = st.session_state

st.write(f"button: {s.button}")
st.write(f"checkbox: {s.checkbox}")
st.write(f"radio: {s.radio}")
st.write(f"multiselect: {s.multiselect}")
```


streamlit 的表格操作

总的来说, 功能上相比于excel, 还是有较多的欠缺的:

- 文字对齐方式:
- 文字颜色修改: excel 可以修改文字颜色, 但同一个单元格内的字体颜色似乎必须统一
- 筛选: excel 可以选中某列进行筛选, streamlit 似乎只能全局搜索
- 自动换行功能:
- 增加行: streamlit 只能在底部加行
- 删除行: streamlit 可用, 无明显缺陷
- 增加/删除列: streamlit 似乎不支持
- 公式计算: 不支持
- 筛选取值在一个列表内的数据: 不支持

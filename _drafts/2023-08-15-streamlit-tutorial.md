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

### dataframe

静态的表格展示用 `st.dataframe`, 动态的表格展示用 `sr.data_editor`, 更复杂可使用 `st_aggrid.AgGrid`

```python
import pandas as pd
import streamlit as st
from uuid import uuid4

df = pd.DataFrame(
    [
        {"command": "st.selectbox", "rating": 4, "is_widget": True, "uuid": str(uuid4())},
        {"command": "st.balloons", "rating": 5, "is_widget": False, "uuid": str(uuid4())},
        {"command": "st.time_input", "rating": 3, "is_widget": True, "uuid": str(uuid4())},
    ]
)

df = df[["command", "rating", "is_widget"]]

st.write("原始数据")
st.dataframe(df)
st.write("修改的数据")
# 注意: 每次对 data_editor 中的数据进行修改时
edited_df = st.data_editor(df, key="changed")  # edited_df 也是 dataframe 类型

favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]
st.markdown(f"Your favorite command is **{favorite_command}** 🎈")

st.write("原始数据")
st.write(df.to_dict("records"))
st.write("修改后的数据")
st.write(edited_df.to_dict("records"))

st.write(st.session_state["changed"])  # 仅包含被修改的行, 具体可参考官方文档
print(f"pass {uuid4()}")
```

#### streamlit-aggrid

版本

```
streamlit==1.32.2
streamlit-aggrid==0.3.4
```

代码

```python
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
import pandas as pd
from uuid import uuid4

print("run head")

df = pd.DataFrame(
    {
        "col1": ["1", "2", "3"],
        "col2": [4, 5, 6],
    }
)
def get_grid_options(df):
    options_builder = GridOptionsBuilder.from_dataframe(df)
    options_builder.configure_column('col1', editable=True)
    options_builder.configure_selection("single")
    grid_options = options_builder.build()
    return grid_options
grid_options = get_grid_options(df)
reload_data = False
flag = st.button("reset")  # 用于将修改的表格恢复为原始的数据
if flag:
    reload_data = True
else:
    reload_data = False

grid_return = AgGrid(
    df,
    grid_options,
    reload_data=reload_data,
)

st.write(grid_return.data)

print(f"button state {flag}")
print(f"run bottom: {uuid4()}")
```

前端与后端的交互逻辑:

```
# 打开网页 http://localhost:8501
run head
button state False
run bottom: 37f1501e-55d1-4de1-950e-30421f9194ce
# 单击表格的某一行
run head
button state False
run bottom: d6c89417-6006-42e2-86fe-ffee6732d618
# 双击修改之前单击的这一行的某个单元格并保存修改
run head
button state False
run bottom: a178a794-adda-4a57-a040-ff81ce47e660
# 单击 reset 按钮, 注意此次 rerun, flag 是 True
run head
button state True
run bottom: 7c8e5bcb-4b64-42c3-b9a8-e8bdb5222892
# 单机一个可修改按钮, 注意此次 rerun, flag 恢复为了 False
run head
button state False
run bottom: 54cd6c45-a645-40b0-bc1f-b1331e3ac158
```

如果需要把按钮放在后边, 实现如下 (TODO: 还有 BUG, 点击按钮复原后, 再选中某行时会触发两次 rerun 而不是一次)

```python
# ...
def reload_data_fn():
    st.session_state['reload'] = True
reload_data = st.session_state.get("reload", False)
print(f"reload_data: {reload_data}")

grid_return = AgGrid(df, grid_options, reload_data=reload_data)
st.session_state['reload'] = False

st.write(grid_return.data)
# 此处如果用下面这种方式的话
# if st.button("reset"):
#     st.session_state["reload"]=True
# 会直接触发一次 rerun, 而在这次 rerun 中, 上面的 reload_data 仍然是 False, 运行至下面时才会将 st.session_state["reload"] 置为 True, 导致无法进行复原 (除非再点击一次按钮)
st.button("reset", on_click=reload_data_fn)
```

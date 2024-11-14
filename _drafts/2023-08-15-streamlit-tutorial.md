---
layout: post
title: "(P1) Streamlit Tutorial"
date: 2023-08-15 10:00:04 +0800
labels: [web]
---

## 动机、参考资料、涉及内容


## 概要

能制作的网页

- 机器学习模型简单演示: 文本/图像/语音分类, 对话机器人
- 标注平台

执行逻辑

脚本从上到下运行, 进行组件渲染. 当用户与页面组件(例如按钮,滚动条)交互时, 整个脚本会从上到下重新运行 (rerun). 一些函数调用可以加上装饰器进行缓存(当输入以及函数代码不发生变化时,不重复运行函数直接取缓存结果). 为了解决重新运行导致状态丢失的问题, 可以使用 `streamlit.session_state`, 在一个 session 中 (即浏览器的一个标签页), 重新运行时会共享 `streamlit.session_state`. 如果需要拆分页面 (浏览器的标签页仍然是同一个, 但页面内的侧边栏供用户交互选择哪个子页面), 则创建 `pages` 文件夹, `pages` 文件夹每一个 python 文件是一个页面, 写法与主页面相同, 所有页面共享一个 `streamlit.session_state`.


## 组件生命周期 (TODO: 重新措辞)

主要参考: [https://docs.streamlit.io/library/advanced-features/widget-behavior](https://docs.streamlit.io/library/advanced-features/widget-behavior)

与组件交互时:

- 先修改 st.session_state
- 然后 callback 函数
- 最后 rerun, rerun 过程时会再依次确定每个组件是否重新构建, 是否使用 `st.session_state` 中的值.

在 rerun 时, 如果 rerun 时的 widget (组件) 与上次的是使用相同的 `label`, `key`, 以及参数 时 (这是通常情况, 例如: `st.slider("A", 1, 10, key="slider_a")`), 通常可以保留状态 (key 不设置也能保留状态, 只是不会被保留在 `st.session_state["slider_a"]` 中).

然而某些情况下, rerun 会改变 `label`, `key` 或参数 (最常见的是改变参数 `st.slider` 的最大值和最小值, 但下面的例子是改变 `label`), 保留状态会有些 tricky.

```python
import streamlit as st
from uuid import uuid4

def change_value():
    print(st.session_state["slider_value"])
    # 将以下注释掉的话: 每次与 slider 交互时, slider 的值都会被复原为默认值 1
    st.session_state["slider_value"] = st.session_state["slider_value"]

st.slider(
    str(uuid4()), 1, 10,
    on_change=change_value,
    key="slider_value",
)
```

- 打开页面时, 先构建前端的滑动条: 由于此时设置了 `key="slider_value`, 所以先搜索 `st.session_state['slider_value']`, 此时这个值没有被设置, 因此滑动条使用默认值 `min_value=1`, 并设置 `st.session_state['slider_value']=1`. (先设置 `st.session_state` 还是先构建完前端组件不清楚)
- 当前端与滑动条交互时 (例如将值修改为 4), 先将 `st.session_state['slider_value']=4`, 然后触发 callback 函数 `change_value`, 然后进行 rerun, 在执行至 `st.slider` 这一行时, 有如下比较 tricky 的地方
  - 由于本次 slider 的 ID 与上次的不同 (因为 label 不同导致 ID 不同, 只要 ID 不同就只能重新构造, ID 基于label, 参数例如 `min_value`, key), 所以会将上次的 slider 销毁掉, 同时会清除掉 `st.session_state["slider_value"]`, 由于此时 `slider_value` 被绑在了上一次运行时的 slider 上, 因此 `slider_value` 也会被销毁, 接下来在本次构造 slider 时重新生成新的 `slider_value`
  - 而如果取消上面的注释行, 行为会变成销毁并重新创建 slider, 但由于 `slider_value` 被重新赋了值, 因此解绑了, 所以这种情况下新构造的 slider 会使用之前的 `st.session_state["slider_value"]`

```python
# 另一种方式: 官方文档上的写法实际上稍有错误
import streamlit as st
from uuid import uuid4

rerun_id = uuid4()
print("start", rerun_id, st.session_state)

def save_value(key):
    st.session_state[key] = st.session_state["_"+key]
def get_value(key):
    st.session_state["_"+key] = st.session_state.get(key, 1)

get_value("slider_value")
st.slider(
    str(uuid4()), 1, 10,
    key="_slider_value",
    on_change=save_value,
    args=("slider_value",)
)

print("end", rerun_id, st.session_state)
```

关于以上, 这里引用官方文档的描述 [https://docs.streamlit.io/library/advanced-features/widget-behavior#widget-life-cycle](https://docs.streamlit.io/library/advanced-features/widget-behavior#widget-life-cycle), 并附注释

```
Calling a widget function when the widget doesn't already exist

If your script rerun calls a widget function with changed parameters or calls a widget function that wasn't used on the last script run:
(上面的例子中 st.slider 的 label 使用 uuid4 来生成, 就是这种情况)

1. Streamlit will build the frontend and backend parts of the widget.
(widget的前后端构造过程在执行 st.slider 这一行内发生的. 所谓后端, 应该是指前端实际上用的是 iframe, 实际上是需要先完成后端, 再嵌入至前端, 不太确定?)
2. If the widget has been assigned a key, Streamlit will check if that key already exists in Session State.
    a. If it exists and is not currently associated with another widget, Streamlit will attach to that key and take on its value for the widget.
    (假设在 callback 函数 change_value 中有 st.session_state["slider_value"] = st.session_state["slider_value"] 这一行, 那么 slider_value 就被 deattach 了, 那么这次构造 slider 时, 就会使用到当前的 st.session_state["slider_value"])
    b. Otherwise, it will assign the default value to the key in st.session_state (creating a new key-value pair or overwriting an existing one).
    (假设 st.session_state["slider_value"] 没有被重新赋值, 那么 slider_value 就还是被 attach 在之前的 slider 上, 那么此次构造 slider 时会覆盖掉之前的 slider_value)
3. If there are args or kwargs for a callback function, they are computed and saved at this point in time.
4. The default value is then returned by the function.

Step 2 can be tricky. If you have a widget:

st.number_input("Alpha", key="A")

and you change it on a page rerun to:

st.number_input("Beta", key="A")

Streamlit will see that as a new widget because of the label change. The key "A" will be considered part of the widget labeled "Alpha" and will not be attached as-is to the new widget labeled "Beta". Streamlit will destroy st.session_state.A and recreate it with the default value.

If a widget attaches to a pre-existing key when created and is also manually assigned a default value, you will get a warning if there is a disparity. If you want to control a widget's value through st.session_state, initialize the widget's value through st.session_state and avoid the default value argument to prevent conflict.
(
    这一句话实际上与组件的生命周期无关, 这里所谓的 warning 是指如果这么写代码:
    st.session_state["slider_value"] = 5
    st.slider("slider_label", 1, 10, value=3, key="slider_value")
    这里的 3 就是 default value, 它与手动赋值 5 有冲突, 这种情况下会以 5 为准, 前端界面上会报一次 warning, 官方建议如果是这种情况, 就要避免使用默认值 value=3.
)
```

## 组件记录

详细 API 文档可直接参考: [https://docs.streamlit.io/library/api-reference](https://docs.streamlit.io/library/api-reference)

### `st.button`, `st.checkbox`, `st.radio`, `st.multiselect`

`st.button` 和 `st.checkbox` 只有 True 和 False 两种状态, 但在与页面的其它组件交互而发生 rerun 时, `st.button` 会复位回 False 的状态, 而 `st.checkbox` 会维持当前的值. 而 `st.radio` 是单选框, `st.multiselect` 是复选框.

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

### `st.dataframe`, `st.data_editor`

静态的表格展示用 `st.dataframe`, 动态的表格展示用 `st.data_editor`, 更复杂可使用 `st_aggrid.AgGrid`

`st.data_editor` 的表格操作

总的来说, 功能上相比于excel, 还是有较多的欠缺的:

- 文字对齐方式:
- 文字颜色修改: excel 可以修改文字颜色, 但同一个单元格内的字体颜色也可以不同
- 筛选: excel 可以选中某列进行筛选, streamlit 似乎只能全局搜索
- 自动换行功能:
- 增加行: streamlit 只能在底部加行
- 删除行: streamlit 可用, 无明显缺陷
- 增加/删除列: streamlit 似乎不支持
- 公式计算: 不支持
- 筛选取值在一个列表内的数据: 不支持


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

### `st.file_uploader`

`file_uploader` 在交互层面上只允许两种操作: 上传一个或多个文件 (上传多个文件只触发一次 rerun), 删除一个上传的文件. 如果上传的文件与已有文件相同, 不做任何校验, 直接重复上传 (例如先上传了 3 个文件, 然后再一次性上传同样的 3 个文件, 那么上传列表将变成 6 个).

想实现这种效果做不到: 用户上传了 3 个文件时, 处理完其中一个文件 (例如将 3 个文件信息用 AgGrid 选中), 然后点击按钮希望从上传列表里删除这个文件, 使得上传列表只剩下 2 个文件. 原因是不能预先设置 `st.session_state` 用于 `file_uploader` 组件:

参考这个问答: [https://discuss.streamlit.io/t/streamlitapiexception-values-for-st-data-editor-cannot-be-set-using-st-session-state-using-data-editor-to-delete-rows/46759/4](https://discuss.streamlit.io/t/streamlitapiexception-values-for-st-data-editor-cannot-be-set-using-st-session-state-using-data-editor-to-delete-rows/46759/4):

```
Values for st.button, st.download_button, st.file_uploader, st.data_editor, st.chat_input, and st.form cannot be set using st.session_state.
```

但是可以做到清空上传的文件

```python
import streamlit as st
from uuid import uuid4

upload_key = st.session_state.get("upload_key", str(uuid4()))
st.session_state["upload_key"] = upload_key

files = st.file_uploader("上传文件", accept_multiple_files=True, key=upload_key)

def delete_on_click():
    st.session_state["upload_key"] = str(uuid4())

st.button("清空上传的文件", on_click=delete_on_click)
```

## 第三方插件

### 可编辑表格: `streamlit-aggrid`

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

如果需要把按钮放在后边, 实现如下 (TODO: 还有 BUG, 点击按钮复原后, 再选中某行时会触发触发一次 rerun, 但行不被选中, 再选中时, 会再触发一次 rerun, 行也被选中)

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

一个完美的解决方案如下 (`reload_data` 总是保持为 `False`, 但点击复原按钮时给 `AgGrid` 一个新的 `key`, 参考下一节):

```python
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from uuid import uuid4

rerun_id = str(uuid4())
print("start", rerun_id)

df = pd.DataFrame(
    {
        "col1": ["1", "2", "3"],
        "col2": ["4", "5", "6"],
    }
)

def get_grid_options(df):
    options_builder = GridOptionsBuilder.from_dataframe(df)
    options_builder.configure_column('col1', editable=True)
    options_builder.configure_selection("single")
    grid_options = options_builder.build()

    return grid_options

grid_options = get_grid_options(df)

key = st.session_state.get("key", str(uuid4()))
st.session_state["key"] = key

grid_return = AgGrid(
    df,
    grid_options,
    key=key,
)

st.write(grid_return.data)

def reload_data_fn():
    st.session_state['key'] = str(uuid4())

st.button("reset", on_click=reload_data_fn)
st.write(grid_return.selected_rows)

print("end", rerun_id)
```

### 用户登录: `streamlit-authenticator`

参考资料:

- 博客: [part-1](https://blog.streamlit.io/streamlit-authenticator-part-1-adding-an-authentication-component-to-your-app/), [part-2](https://blog.streamlit.io/streamlit-authenticator-part-2-adding-advanced-features-to-your-authentication-component/)


## 示例

### 常见错误: `session_state` 无法被修改

```python
import streamlit as st
from uuid import uuid4

RUN_ID = str(uuid4())
print(f"RERUN {RUN_ID}: {st.session_state}")
st.text_input("text", key="text")
button = st.button("save")
if button:
    print(f"点击按钮 {RUN_ID}: {st.session_state}")
    st.session_state["text"] = ""
print(f"FINISH_RUN {RUN_ID}: {st.session_state}")
```

将上述代码运行起来后, 与前端交互, 先在文本输入框里输入, 一切正常, 点击按钮, 则执行报错, 后端日志如下

```
# 第一次 rerun
RERUN bd8c35e4-60bc-4a62-b741-c1503ee37b86: {}
FINISH_RUN bd8c35e4-60bc-4a62-b741-c1503ee37b86: {'text': ''}

# 输入文本后触发 rerun
RERUN cfd7307c-6bd1-4368-b155-a9867f0d2d15: {'text': '12'}
FINISH_RUN cfd7307c-6bd1-4368-b155-a9867f0d2d15: {'text': '12'}

# 点击按钮后触发 rerun
RERUN 17935c8e-78af-4fa3-b28a-f807a571b21d: {'text': '12'}
点击按钮 17935c8e-78af-4fa3-b28a-f807a571b21d: {'text': '12'}
streamlit.errors.StreamlitAPIException: `st.session_state.text` cannot be modified after the widget with key `text` is instantiated.
```

原因是 streamlit 在组件被渲染了之后就不允许修改: 在我们点击按钮时, streamlit 会记录组件的现状, 并重新运行脚本, 当我们进入 if 分支后, 尝试对组件的值重新修改, 就会引发错误.

**修正方式**

```python
import streamlit as st
from uuid import uuid4

RUN_ID = str(uuid4())
print(f"RERUN {RUN_ID}: {st.session_state}")

def button_callback():
    print(f"点击按钮 {RUN_ID}: {st.session_state}")
    st.session_state["text"] = ""

st.text_input("text", key="text")
button = st.button("save", on_click=button_callback)
print(f"FINISH_RUN {RUN_ID}: {st.session_state}")
```

后端日志

```
# 第一次 rerun
RERUN 274c2950-a3ad-4a15-b992-7291958d3cfa: {}
FINISH_RUN 274c2950-a3ad-4a15-b992-7291958d3cfa: {'text': ''}

# 输入文本后触发 rerun
RERUN 1e371487-47cb-4daf-8e04-cfef94b09800: {'text': '12'}
FINISH_RUN 1e371487-47cb-4daf-8e04-cfef94b09800: {'text': '12'}

# 点击按钮后, 注意: 在 callback 里, RUN_ID 的值还是上一次的值, 但 rerun 之后, RUN_ID 被更新了
点击按钮 1e371487-47cb-4daf-8e04-cfef94b09800: {'text': '12'}
RERUN c62c4d41-213d-4d9f-aa6d-d37597360f3d: {'text': ''}
FINISH_RUN c62c4d41-213d-4d9f-aa6d-d37597360f3d: {'text': ''}
```
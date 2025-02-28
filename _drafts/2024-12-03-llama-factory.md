---
layout: post
title: "(P0) llama-factory 浅析"
date: 2024-12-03 13:00:04 +0800
labels: [llm]
---

yaml 配置文件里的各个参数代表的含义:




训练入口: `llamafactory/train/tuner.py:run_exp`, `run_exp` 首先会通过 `HfArgumentParser` 解析参数, 定义位于 `llamafactory/hparams/*.py`:

- `ModelArguments`
- `DataArguments`
- `Seq2SeqTrainingArguments`
- `FineTuningArguments`
- `GenerationArguments`

然后进入 `train/sft/workflow.py:run_sft` (或者其他的 `train/*/workflow:run_*`)

`data/loader.py:get_dataset` 得到一个普通字典:

```python
{
    "train_dataset": Dataset,  # hf datasets.Dataset
    "eval_dataset": Dataset
}
```

`get_dataset` 内部分为两步:

(1) `_get_merged_dataset`: 具体是先使用 `_load_single_dataset` 读取单个数据集(这里会涉及到 DataAttr, 并使用 `align_dataset` 根据数据集类型: `alpaca` 或者 `sharegpt` 最终转化为如下特定的 key: `_prompt`, `_response`, `_system`, `_tools`, `_images`, `_videos`), 然后 `merge_dataset` (基本上是 hf datasets 的功能)

此步骤之后, 数据被规范化为(这一步骤与 yaml 文件中的配置非常相关):

```python
{
    "_prompt": [{"role": "user", "content": "xxx"}, ...]
    "_response": [{"role": "assistant", "content": "yyy"}],
    "_system": "",
    "_tools": "",
    "_images": None,
    "_videos": None
}
```

(2) `_get_preprocessed_dataset`: 使用 Template, Tokenizer 将数据处理成模型输入


涉及到 DataAttr 的主入口是 `llamafactory/data/parser.py:get_dataset_list`:

```python
# dataset_names 的实参是 yaml 文件中的 dataset 和 eval_dataset
# dataset_dir 的实参是 yaml 文件中的 dataset_dir, 也就是 dataset_info.json 所在的文件目录
def get_dataset_list(dataset_names: Optional[Sequence[str]], dataset_dir: str) -> List["DatasetAttr"]: ...
```

`llamafactory/data/parser.py:DatasetAttr`

```python
@dataclass
class DatasetAttr:
    # basic configs
    load_from: Literal["hf_hub", "ms_hub", "om_hub", "script", "file"]
    dataset_name: str
    formatting: Literal["alpaca", "sharegpt"] = "alpaca"
    ranking: bool = False
    # extra configs
    subset: Optional[str] = None
    split: str = "train"
    folder: Optional[str] = None
    num_samples: Optional[int] = None
    # common columns
    system: Optional[str] = None
    tools: Optional[str] = None
    images: Optional[str] = None
    videos: Optional[str] = None
    # rlhf columns
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    kto_tag: Optional[str] = None
    # alpaca columns
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None
    # sharegpt columns
    messages: Optional[str] = "conversations"
    # sharegpt tags
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"
```


## 基础设施

### ToolUtils

`data/tool_utils.py:ToolUtils`: 抽象 dataclass 基类, 定义了 `get_function_slots` (期望模型输出的模板), `tool_formatter` (为了得到期望输出所需要的 system prompt), `tool_extractor` (假设模型按照期望的格式输出了,提取工具名和工具入参) 三个静态抽象方法, 最重要的两个具体类是 `DefaultToolUtils`, `GLM4ToolUtils`, 示例如下

```python
from llamafactory.data.tool_utils import ToolUtils, DefaultToolUtils, GLM4ToolUtils

# get_tool_utils("default"), get_tool_utils("glm4") 分别对应 DefaultToolUtils() 和 GLM4ToolUtils()
DefaultToolUtils().get_function_slots()

# 输出:
# ['Action: {{name}}\nAction Input: {{arguments}}\n']

DefaultToolUtils().tool_extractor('Action: test_tool\nAction Input: {"x": 1, "y": "abc"}\n')
# 输出:
# [('test_tool', '{"x": 1, "y": "abc"}')]

DefaultToolUtils().tool_formatter(
    [
        {
            "name": "test_tool",
            "description": "tool_desc",
            "parameters": {
                "type": "object",
                "properties": {
                    "foo": {"type": "string", "description": "foo_desc"},
                    "bar": {"type": "number", "description": "bar_desc"}
                },
                "required": ["foo"]
            }
        }
    ]
)

# 输出
# 'You have access to the following tools:\n> Tool Name: test_tool\nTool Description: tool_desc\nTool Args:\n  - foo (string, required): foo_desc\n  - bar (number): bar_desc\n\nUse the following format if using a tool:\n```\nAction: tool name (one of [test_tool])\nAction Input: the input to the tool, in a JSON format representing the kwargs (e.g. ```{"input": "hello world", "num_beams": 5}```)\n```\n'
```


### Formatter

`data/formatter.py:Formatter`: 抽象 dataclass 基类, 包含两个实例属性: `slots` (槽位) 和 `tool_format`, 包含 `apply` 和 `extract` 两个方法, 前者用于将槽位的实际值传入得到最终的字符串, 后者用于解析大模型的 FunctionCall 结果, 将其转化为格式化输出. Formatter 的重要子类有:

- `EmptyFormatter`: 无实际作用, 用于作为默认值
- `StringFormatter`: 参见下面的例子 (也可以参考 `tests/data/test_formatter.py` 中的例子)

```python
from llamafactory.data.formatter import Formatter, EmptyFormatter, StringFormatter, FunctionFormatter, ToolFormatter

# slots 需要传序列型数据
string_formatter = StringFormatter(slots=["This is {{x}}", "Here are {{y}}"])
string_formatter.apply(**{"x": "123"})
# ["This is 123", "Here are {{y}}"]
```
- `FunctionFormatter`, `ToolFormatter`:

```python
def test_multi_function_formatter():
    formatter = FunctionFormatter(slots=[], tool_format="default")
    tool_calls = json.dumps([{"name": "tool_name", "arguments": {"foo": "bar", "size": 10}}] * 2)
    assert formatter.apply(content=tool_calls) == [
        """Action: tool_name\nAction Input: {\"foo\": \"bar\", \"size\": 10}\n""",
        """Action: tool_name\nAction Input: {\"foo\": \"bar\", \"size\": 10}\n""",
    ]


def test_default_tool_formatter():
    formatter = ToolFormatter(tool_format="default")
    tools = [
        {
            "name": "test_tool",
            "description": "tool_desc",
            "parameters": {
                "type": "object",
                "properties": {
                    "foo": {"type": "string", "description": "foo_desc"},
                    "bar": {"type": "number", "description": "bar_desc"},
                },
                "required": ["foo"],
            },
        }
    ]
    assert formatter.apply(content=json.dumps(tools)) == [
        "You have access to the following tools:\n"
        "> Tool Name: test_tool\n"
        "Tool Description: tool_desc\n"
        "Tool Args:\n"
        "  - foo (string, required): foo_desc\n"
        "  - bar (number): bar_desc\n\n"
        "Use the following format if using a tool:\n"
        "```\n"
        "Action: tool name (one of [test_tool])\n"
        "Action Input: the input to the tool, in a JSON format representing the kwargs "
        """(e.g. ```{"input": "hello world", "num_beams": 5}```)\n"""
        "```\n"
    ]
```

### Template

```python
@dataclass
class Template:
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_function: "Formatter"
    format_observation: "Formatter"
    format_tools: "Formatter"
    format_separator: "Formatter"
    format_prefix: "Formatter"
    default_system: str
    stop_words: List[str]
    efficient_eos: bool
    replace_eos: bool
    replace_jinja_template: bool
    mm_plugin: "BasePlugin"

    # encode_oneturn 和 encode_multiturn 内部调用 _encode 转化为 input_ids
    def encode_oneturn(self, tokenizer, messages, system, tools) -> Tuple[List[int], List[int]]: ...
    def encode_multiturn(self, tokenizer, messages, system, tools) -> List[Tuple[List[int], List[int]]]: ...
    
    def extract_tool(self, content): ...
    def _encode(self, tokenizer, messages, system, tools) -> List[List[int]]: ...
    def _convert_elements_to_ids(self, tokenizer, elements) -> List[int]
```
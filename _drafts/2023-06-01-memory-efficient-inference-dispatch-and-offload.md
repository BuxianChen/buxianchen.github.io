---
layout: post
title: "(WIP) Memory efficient inference (🤗's dispatch and offload)"
date: 2023-06-01 17:20:04 +0800
labels: [transformers]
---

## 动机、参考资料、涉及内容

动机

- 🤗 transformer/accelerate 的 `init_empty_weight` 和 `dispatch_model` 等 API

参考资料

- [https://huggingface.co/docs/accelerate/v0.19.0/en/usage_guides/big_modeling](https://huggingface.co/docs/accelerate/v0.19.0/en/usage_guides/big_modeling)
- [https://huggingface.co/blog/accelerate-large-models](https://huggingface.co/blog/accelerate-large-models)

涉及内容

- 使用方法【样例】
- 文档补充：某些辅助函数官方文档缺失，某些对外API的参数说明不明确
- 源码分析

## 样例代码

```
# requirements.txt
torch >= 1.10.0
transformers==4.29.2
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausualLM, AutoConfig
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

pretrained_name_or_path = "./fnlp/moss-moon-003-sft"

# method 1: 调用相关的辅助函数(有利于了解整个过程)
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(
        config,
        # 在执行 cls(...) 之前, 会利用 torch.set_default_dtype(torch_dtype)全局设定默认浮点数类型
        torch_dtype=torch.float16,
        trust_remote_code=True)
# !!!这一步是必须的!!!
model.tie_weights()

max_memory = {
    0: "10GB",
    1: "10GB",
    "cpu": "20GB",
}

device_map = infer_auto_device_map(
    model,
    max_memory=max_memory,
    dtype=None,  # 使用model中的tensor类型估算所需的内存/显存
    no_split_module_classes=model._no_split_modules,  # 某些submodule必须放在同一个设备上
)

# 前面的两个步骤是为了获取合适的 device_map

# 写法1: 待确认
# model = AutoModelForCausalLM.from_pretrained(
#     pretrained_name_or_path,
#     trust_remote_code=True,
#     # 在执行 cls(...) 之前, 会利用 torch.set_default_dtype(torch_dtype)全局设定默认浮点数类型
#     torch_dtype=torch.float16,  
#     device_map=device_map,
#     offload_folder="offload",
#     offload_state_dict=True,  # ??
#     # low_cpu_memory_usage=True,  # 某些情况下会根据其他参数自动设定为True
#     # _fast_init=True,  # 默认值为True
# )

# 写法2: 官网推荐的写法(https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
model = load_checkpoint_and_dispatch(
    model,
    pretrained_name_or_path,
    device_map=device_map
)


# method 2: (直接一步到位), 【待确认】
model = AutoModelForCausalLM.from_pretrained(
    pretrained_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    # low_cpu_mem_usage=True,  # 设置了device_map后low_cpu_mem_usage会被默认设置为True
    max_memory=max_memory,
    device_map="sequential",  # 似乎是必须设置为这个, 才会让max_memory生效
    offload_folder="offload",
    # offload_state_dict=True
)


tokenizer = AutoTokenizer.from_pretrained(
    pretrained_name_or_path,
    trust_remote_code=True
)

inputs = tokenizer(["Alice and Bob"], return_tensors="pt")
output = model.generate(
    inputs=inputs["input_ids"],
    temperature=0.7,
    top_k=0,  # 0 表示不使用 top_k 对 logits 进行后处理
    top_p=0.8,
    repetition_penalty=1.02,
    max_length=50,
    do_sample=True,
    return_dict_in_generate=True,
    attention_mask=inputs["attention_mask"],
)

texts = tokenizer.batch_decode(output["sequences"])
```


备注：【以下API间的关系有些混乱，待研究】

- `PretrainedModel.from_pretrained(...)`
- `AutoModelForXXX.from_config(...)`
- `YYYModelForXXX._from_config(...)`
- `accelerate.init_empty_weight(...)`
- `accelerate.infer_auto_device_map(...)`
- `accelerate.get_balanced_memory(...)`
- `accelerate.dispatch_model(...)`
- `accelerate.load_checkpoint_and_dispatch(...)`

这些参数之间怎么配合的:

- `torch_dtype`, `model.config.torch_dtype`
- `device_map`: None, auto, balanced, balanced_low_0, sequential, dict(...)
- `offload_state_dict`: 
- `low_cpu_mem_usage`:
- `_fast_init`:


## 主要API及功能

```python
import torch.nn as nn
import torch

class FeedForward(nn.Module):
    def __init__(self, num_layer, in_dim, hidden_dim):
        super().__init__()
        assert num_layer > 2
        self.num_layer = num_layer
        layers = nn.Sequential()
        layers.append(nn.Linear(in_dim, hidden_dim))
        for i in range(num_layer - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, in_dim))
        self.layers = layers
        self.activate = nn.ReLU()
    def forward(self, x):
        x = self.layers(x)
        x = self.activate(x)
        return x
    
class Head(nn.Module):
    def __init__(self, in_dim, num_class):
        super().__init__()
        self.out = nn.Linear(in_dim, num_class)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.out(x)
        probs = self.softmax(x)
        return probs

class ExampleModel(nn.Module):
    def __init__(self, vocab_size, in_dim, num_layer, num_class):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, in_dim)
        self.feed_forward = FeedForward(num_layer, in_dim, 4*in_dim)
        self.head = Head(in_dim, num_class)

    def forward(self, x):
        # x: (B, L), out: (B, L, num_class)
        out = self.embed(x)
        out = self.feed_forward(out)
        out = self.head(out)
        return out
```

### accelerate.utils.modeling.infer_auto_device_map

**signature**

```python
device_map: Dict[str, torch.dtype] = infer_auto_device_map(
    model: nn.Module,
    max_memory: Optional[Dict]=None,
    no_split_module_classes: List[str]=None,
    dtype: Optional[Union[torch.dtype, str]]=None,
    special_dtypes: Optional[Dict] = None,
    verbose: bool = False
)
```

**入参**
- `model`: 只要求是 `nn.Module` 即可, 因此这个函数**不会**将 `model._no_split_module` 自动纳入 `no_split_module_classes` 中
- `max_memory`: 设定每个设备最大可使用的CPU内存/GPU显存
    - Dict:
        ```python
        max_memory={0: "10GB", 1: "20GB", "cpu": "200MB"}
        ```
    - None:
- `no_split_module_classes`: 不可分割的子模块名, 例如假设定义在 `transformers.models.t5.modeling_t5.py` 中的 `T5Block` 被认为是不可分割到不同设备上的基本单元, 则传入:
    ```python
    no_split_module_classes=["T5Block"]
    ```
- `dtype`: 
    - None: 表示用 model 本身的每个参数的 dtype 进行计算内存大小
    - torch.float32: 对于 model 的每个 tensor, tensor 的 dtype 与传入的 `dtype` 的最小值计算内存大小
- `special_dtypes`:
    ```python
    # 字典的 key 需要在 model.state_dict().keys() 中, 即它需要代表一个特定的张量
    special_dtypes = {"feed_forward.layers.0.weight": torch.float16}
    ```
**出参**

## 辅助API及功能


### accelerate.utils.modeling.compute_module_sizes

**signature**
```python
module_sizes: Dict[str, int] = compute_module_sizes(
    model: nn.Module,
    dtype: Optional[Union[torch.dtype, str]]=None,
    special_dtypes: Optional[Dict] = None
)
```

**入参**
参数类型与 `infer_auto_device_map` 相同, 注意: **既统计了 Parameter 也统计了 Buffer**

**出参**
- `module_sizes`: 各层级的 Module/Parameter/Buffer 的内存占用大小, 见示例

**例子**

```python
from accelerate.utils.modeling import compute_module_sizes
model = ExampleModel(100, 16, 4, 3)
compute_module_sizes(
    model.half(),
    dtype=torch.float32,
    special_dtypes={'feed_forward.layers.0.weight': torch.float32},
)

# 输出结果如下
{
    '': 26246,                              # Module
    'embed': 3200,                          # Module
    'embed.weight': 3200,                   # Parameter
    'feed_forward': 22944,                  # Module
    'feed_forward.layers': 22944,           # Module
    'feed_forward.layers.0': 4224,          # Module
    'feed_forward.layers.0.weight': 4096,   # Parameter: 4byte*16*64=4096byte
    'feed_forward.layers.0.bias': 128,      # Parameter: min(2, 4)=2byte*64=128byte
    'feed_forward.layers.1': 8320,          # Module
    'feed_forward.layers.1.weight': 8192,   # Parameter: min(2, 4)=2byte*64*64=8192byte
    'feed_forward.layers.1.bias': 128,      # Parameter
    'feed_forward.layers.2': 8320,          # Module
    'feed_forward.layers.2.weight': 8192,   # Parameter
    'feed_forward.layers.2.bias': 128,      # Parameter
    'feed_forward.layers.3': 2080,          # Module
    'feed_forward.layers.3.weight': 2048,   # Parameter
    'feed_forward.layers.3.bias': 32,       # Parameter
    'head': 102,                            # Module
    'head.out': 102,                        # Module
    'head.out.weight': 96,                  # Parameter
    'head.out.bias': 6                      # Parameter
}
```

### accelerate.utils.modeling.find_tied_parameters

首先区分几类“绑定”

- 变量绑定: `self.layer2=self.layer1`, 这种做法是对同一份内存绑定了两个实例变量名
    - `state_dict=self.state_dict()`: {"layer1.weight": xx, "layer2.weight": xx}, 两个key对应的value是相同的(对应同一份内存), 在使用 `torch.save(sate_dict)` 时内存仅占用一份
    - `self.modules()`: 迭代器会去重, **只迭代出一个**
    - `layer1.weight.grad` 等关于梯度的操作会同时作用于两者, 因此可以一直保持一致
    - `layer1.weight is layer2.weight`: True
    - `find_tied_parameters`: **无法检测**出这种情形, 返回值为空列表
- tied Parameter (**🤗 Transformer 中的 tie_weight 是指这种情形**): `self.layer2.weight=self.layer1.weight`
    - `state_dict=self.state_dict()`: {"layer1.weight": xx, "layer2.weight": xx}, 两个key对应的value是相同的(对应同一份内存), 在使用 `torch.save(sate_dict)` 时内存仅占用一份
    - `self.modules()`: 迭代器会去重, 但layer1与layer2是不同的, **迭代出两个**
    - `layer1.weight.grad` 等关于梯度的操作会同时作用于两者, 因此可以一直保持一致
    - `layer1.weight is layer2.weight`: True
    - `find_tied_parameters`: **可以检测**出这种情形, 返回值为 `[["layer1.weight", "layer2.weight"]]`
- copy Parameter data 【这两种似乎是完全的表现,有些奇怪】:
    - `layer2.weight.data=layer1.weight.data.clone()`
    - `layer2.weight.data=layer1.weight.data`

    例子【待研究清楚】
    ```python
    layer1 = nn.Linear(3, 3)
    layer2 = nn.Linear(3, 3)
    with torch.no_grad():
        layer2.weight.data = layer1.weight.data#.clone()
        layer2.bias.data = layer1.bias.data#.clone()
    import itertools
    opt=torch.optim.SGD(itertools.chain(layer1.parameters(), layer2.parameters()), lr=0.001)
    
    # clone: False, 不使用 clone：False
    layer2.weight.data is layer1.weight.data
    # 多次运行, 这两个值的id一直在变, 但无论使用clone还是不使用, 两者的值相同
    id(layer2.weight.data), id(layer1.weight.data)
    
    x = torch.rand(4, 3)
    z = layer2(layer1(x)).sum()
    y = torch.rand(())
    loss = z - y
    loss.backward()

    # clone: 不同, 不使用 clone：相同
    layer1.weight.grad, layer2.weight.grad

    opt.step()

    # clone: 不同, 不使用 clone: 相同
    layer1.weight, layer2.weight
    ```

**signature**
```python
tied_names: List[List[str]] = find_tied_parameters(model: nn.Module)
```

**入参**

**出参**

- `tied_names`:
    ```python
    tied_names=[
        ["layer1.weight", "layer2.weight", "layer3.weight"],  # 这三个weight被tied了
        ["layer1.bias", "layer2.bias", "layer3.bias"] # 这三个bias被tied了
    ]
    ```

**内部逻辑**

本质上, 是使用 `nn.Module.named_parameters()`, 然后两两使用 `parameter1 is parameter 2` 判断是否为 tied_weight. 因此完全相同的 submodule 由于 `nn.Module.named_parameters()` 会被预先排除掉, 因此不会出现在结果里。

**例子**

```python
import torch.nn as nn
import torch
from accelerate.utils.modeling import find_tied_parameters
class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 3)
        self.layer2 = self.layer1
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out
    
class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 3)
        self.layer2 = nn.Linear(3, 3)
        self.layer2.weight = self.layer1.weight
        self.layer2.bias = self.layer1.bias
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out

model_a = ModelA()
model_b = ModelB()
# 这里调用的是 nn.Module 的方法, 它会去除重复的module, 因此只会打印出 ModelA 和 ModelA.layer1
for module in model_a.modules():
    print(module)
find_tied_parameters(model_a)  # 返回为空列表!!!

# 这里调用的是 nn.Module 的方法, 因为layer1和layer2是分别定义的, 只是参数被绑定
for module in model_b.modules():
    print(module)
find_tied_parameters(model_a)  # 返回为[['layer1.weight', 'layer2.weight'], ['layer1.bias', 'layer2.bias']]
```


### get_max_layer_size


**内部逻辑**

递归遍历整个模型, 直至最小不可分割的 Module 或者单独的 Parameter/Buffer, 计算这些不可分割的部分所需要的内存的最大值, 判断不可分割的 Module 的依据为:

```python
module.__class__.__name__ in no_split_module_classes
# 假设 module 是一个 torch.nn.modules.linear.Linear 层
# module.__class__.__name__: "Linear"
# no_split_module_classes: ["Linear", "T5Block"]
```
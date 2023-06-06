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
accelerate==0.19.0
```

写法一: **注意: 本篇博客后续几乎全部的内容都是为了解释这一行初始化模型的代码**

```python
pretrained_name_or_path = "./fnlp/moss-moon-003-sft"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    # low_cpu_mem_usage=True,  # 设置了device_map后low_cpu_mem_usage会被默认设置为True
    max_memory=max_memory,
    device_map="sequential",  # 似乎是必须设置为这个, 才会让max_memory生效（触发infer_auto_device_map的调用）
    offload_folder="offload",
    offload_state_dict=True  # cpu上的参数暂时offload, 再重新加载回cpu, 仅用于减少加载模型时的内存占用, 但加载速度会变慢一些
)
```

写法二:


```python
import torch
from transformers import AutoTokenizer, AutoModelForCausualLM, AutoConfig
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

pretrained_name_or_path = "./fnlp/moss-moon-003-sft"
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
# https://huggingface.co/docs/accelerate/usage_guides/big_modeling
model = load_checkpoint_and_dispatch(
    model,
    pretrained_name_or_path,
    device_map=device_map
)
```

用法:

```python
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
- `accelerate.init_empty_weight(...)`: 上下文管理器, 在上下文范围内, 修改 `nn.Module` 的 `register_parameter` 与 `register_buffer` 函数, 使得 `nn.Module` 中的 tensor 总是在 `torch.device("meta")` 上
- `accelerate.get_balanced_memory(...)`: 在设定 `device_map` 为 `auto/balanced/balanced_low_0` 时, 得到 `max_memory` 字典
- `accelerate.infer_auto_device_map(...)`: 根据 `max_memory` 字典得到 `device_map` 字典, 确定每个 module 或 parameter/buffer 的设备
- `accelerate.dispatch_model(...)`: 根据 `device_map` 字典增加 `forward` 的 hook
- `accelerate.load_checkpoint_and_dispatch(...)`: 并未在 🤗 transformer 被使用 (与 from_pretrain 有些重复代码, 猜想应该是代码还需要重构好)

这些参数之间怎么配合的:

- `torch_dtype`, `model.config.torch_dtype`
- `device_map`: None, auto, balanced, balanced_low_0, sequential, dict(...)
- `offload_state_dict`: 如果为 False, 那么 device_map 为 cpu 的 tensor 会放在cpu上, 因此在加载过程中，cpu的内存要包含一份shard的state_dict与应该放在cpu上的model里的tensor。如果设置 `offload_state_dict` 为 True，那么对于 device_map 为 cpu 的 tensor 会先临时 offload 到 disk 上，这样子保证 cpu 的内存只要能装下最大的 shard 即可，在所有 shard 处理完成后，在将 offload 的 tensor 全部移回到 cpu 上（offload目录也将被删除）。由此 cpu 的内存只要超过最大shard、以及能装下放在 cpu 上的 model 的 tensor 即可，而不是需要能装下这两者之和。
- `low_cpu_mem_usage`:
- `_fast_init`:


## 主要API及功能

### `PretrainedModel.from_pretrained`

#### low_cpu_mem_usage 参数

按照 [🤗 blog](https://huggingface.co/blog/accelerate-large-models) 的描述, low_cpu_mem_usage 为 False 以及 True 分别对应如下两种流程

流程一：
> 1. Create the model
> 2. Load in memory its weights (in an object usually called state_dict)
> 3. Load those weights in the created model
> 4. Move the model on the device for inference

流程二：
> 1. Create an empty (e.g. without weights) model
> 2. Decide where each layer is going to go (when multiple devices are available)
> 3. Load in memory parts of its weights
> 4. Load those weights in the empty model
> 5. Move the weights on the device for inference
> 6. Repeat from step 3 for the next weights until all the weights are loaded

经过对源码的仔细分析, 将上述过程细化如下:

low_cpu_mem_usage 的默认值为 False, 但当其他一些参数被设定的情况下, low_cpu_mem_usage 会被自动设置为 True

- device_map 不为 None 时, low_cpu_mem_usage 会被自动设置为 True
- ...


**low_cpu_mem_usage=False**

当模型的参数文件只有一个时, 与流程一的描述一致，因此在流程一的第 2 步结束时，cpu 中会有着两份权重

- 初始化一个随机参数的模型（在CPU上）
- load `pytorch_model.bin`
- 将权重放入模型中（CPU上）

当模型的参数文件有多个时（即所谓的 "shard checkpoint"）

- 初始化一个随机参数的模型（在CPU上）
- 逐个 `pytorch_model-0000x-of-0000y.bin` 文件, 并用于加载至模型（在CPU上）中, 每个文件 load 进模型后 load 的参数字典

因此实际上只需要 “模型文件大小+1个shard参数” 的 CPU 内存。

**low_cpu_mem_usage=True**

当模型的参数文件只有一个时:

- 初始化一个空模型(device="meta")
- 如果模型的参数文件只有一个, 则通过 load `pytorch_model.bin` （load 在 cpu上）来获取保存的模型参数的 key 列表, 随后将 load 的参数字典先释放
- 根据 device_map, max_memory, torch_dtype 确定每个参数最终需要存放的设备
- 重新 load `pytorch_model.bin`（load 在 cpu 上）
- 将权重放入相应的设备中。注意，这一步骤可能会有如下几种情形导致需要更多的内存资源
    - 如果某个权重需要放置在 GPU 上，那么这份权重在 CPU 和 GPU 上都有一份
    - 如果某个权重需要的 dtype 与 load 出来的权重不相同，那么这个权重需要存储两份
    - 如果某个权重需要的 dtype 与 load 出来的权重相同，那么这个权重只会占用一份空间
    
    注意在将模型的权重初始化好之前，load 的权重都不会被释放
- 将 load 的权重一次性释放


当模型的参数文件有多个时

- 初始化一个空模型(device="meta")
- 如果模型的参数文件有多个, 则通过 load `pytorch_model.bin.index.json` 文件获取保存的模型参数的 key 列表
- 根据 device_map, max_memory, torch_dtype 确定每个参数最终需要存放的设备
- 每次 load 一个 `pytorch_model-0000x-of-0000y.bin` 文件, 初始化模型的一部分参数, 然后释放掉这一个 `pytorch_model-0000x-of-0000y.bin` 的参数文件


以下是简化后的伪代码: `low_cpu_mem_usage=True`，且为单个权重文件的情形

```python
assert model.device == torch.device("meta")
state_dict = torch.load("pytorch_model.bin")
device_map = {
    "layer1.weight": "cuda:0",
    "layer1.bias": "cuda:0",
    "layer2.weight": "cpu",
    "layer2.bias": "cpu",
}
for name, param in state_dict.items():
    # submodule = model.layer1 或 model.layer2
    submodule = get_submodule(model, name)
    tensor_name = name.split(".")[-1]
    with torch.no_grad():
        # 注意如果new_value的device与dtype与param相同时, 不需要额外占用内存空间
        new_value = torch.tensor(param.to(device=device_map[name], dtype=...))
        if is_buffer:
            submodule._buffers[tensor_name] = new_value
        else:
            submodule._parameters[tensor_name] = new_value
del state_dict
gc.collect()
```

#### _fast_init

这个参数似乎作用并不算太大

#### device_map, max_memory, torch_dtype

这两个参数一起决定每个参数的 device
- `device_map: str/Dict`
- `max_memory: None/Dict`

逻辑如下: 
- 如果 `device_map` 本身是 Dict 时, 则直接确定每个参数的 device
- 如果 `device_map` 取值为字符串, 而 `max_memory=None`, 那么就【待确认】假定所有的 GPU 与 CPU 的内存都可以使用, 由此得到字典形式的 `max_memory`, 然后再根据 `device_map` 的具体取值来确定每个参数的 device:
    - `auto/balanced`
    - `balanced_low_0`
    - `sequential`
- 如果 `max_memory` 为 Dict, `device_map` 必须取值为 `"sequential"`

这个参数以及config决定每个参数dtype
- `torch_dtype: None/torch.dtype`



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
    具体逻辑可参考 `get_max_layer_size` 中的解释
- `dtype`: 
    - None: 表示用 model 本身的每个参数的 dtype 进行计算内存大小
    - torch.float32: 对于 model 的每个 tensor, tensor 的 dtype 与传入的 `dtype` 的最小值计算内存大小
- `special_dtypes`:
    ```python
    # 字典的 key 需要在 model.state_dict().keys() 中, 即它需要代表一个特定的张量
    special_dtypes = {"feed_forward.layers.0.weight": torch.float16}
    ```
**出参**

- `device_map` 是一个字典, key 是子模块名或者是权重名(当 key 是子模块名时, 表明整个子模块都在同一设备上), value 是设备名(0/1/cpu/disk), 例子:
    ```python
    device_map = {
        'embed': 0                              # Module, 包含权重 embed.weight
        'feed_forward.layers.0': 0,             # Module, 包含权重 feed_forward.layers.0.weight 与 feed_forward.layers.0.bias
        'feed_forward.layers.1.weight': 0,      # Parameter
        'feed_forward.layers.1.bias': "cpu",    # Parameter
        'feed_forward.layers.2': "cpu",         # Module
        'feed_forward.layers.3': "disk",        # Module
        'head': "disk"                          # Module, 包含 head.out Module
    }
    ```

### accelerate.big_modeling.dispatch_model

依赖的辅助 API 如下:
- accelerate.utils.modeling.check_device_map

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

### accelerate.utils.modeling.check_device_map

**signature**
```python
check_device_map(model: nn.Module, device_map)
```

**输入**

略

**输出**

无, 只可能 raise 错误信息

**内部逻辑**

检查 `device_map` 是否会覆盖住 `model` 的所有参数, 注意: 这个函数**不检查** model 当前实际的参数位置是否与 device_map 一致。内部的实现逻辑是先获取 `all_names = model.state_dict().keys()`, 然后对于 `device_map` 中的每一个 key, 都在 `all_names` 将其排除(排除掉 key 以及以 `f"{key}."` 开头的键), 最后断言 `all_names` 为空列表。

备注: 这个检查其实有一定的缺陷，例如 `device_map` 本身有可能“不合法”，例如：
```python
device_map = {"fc": 0, "fc.weight": 0, "fc.bias": "cpu"}
```

### accelerate.utils.offload.OffloadedWeightsLoader

```python
from collections.abc import Mapping
class OffloadedWeightsLoader(Mapping):
    def __init__(self,
        state_dict,
        save_folder=None,
        index=None,
        device=None
    ):
        self.all_keys = ... # list(state_dict.keys()) + index 中的键
        self.device = device
        ...
    
    def __getitem__(self, key):
        if key in self.state_dict:
            return self.state_dict[key]
        else:
            weight = np.memmap(weight_file, dtype, shape, mode="r")
            weight = torch.tensor(weight)
            return weight
    
    ...
```

说明: 从 `dispatch_model` 中对这个函数的使用里:

- 当 main_device 为 gpu 时, 实例化参数 state_dict 为所有在 cpu 上的参数, save_folder 为 offload 文件夹, device 为 None
- 当 main_device 为 cpu 时, 实例化参数 state_dict 为 None, save_folder 为 offload 文件夹, device 为 None

`__getitem__` 方法:
- 当 main_device 为 gpu 时, 无论原本的参数 offload 到 disk 还是本就在 cpu 上, 都会将 tensor 转移到 cpu 上返回, 已确认，原本在 disk 上的参数，返回的tensor占用内存
- 当 main_device 为 cpu 时, 原本 offload 到 disk 上的参数会转移到 cpu 上返回, 已确认: 返回的tensor占用内存

```python
# 验证代码
import numpy as np
import psutil
import torch
nrows, ncols = 100000, 1000  # 需要占用大约 400MB

def show_memory():
    mem = psutil.virtual_memory()
    print("Used", mem.used / 1024 / 1024, "MB")
    print("Free", mem.free / 1024 / 1024, "MB")
show_memory()                # Used: 602MB, Free: 2586MB
# f = np.memmap('memmapped.dat', dtype=np.float32, mode='w+', shape=(nrows, ncols))
# for i in range(nrows):
#     f[i, :] = np.random.rand(ncols)
# f.flush()
# del f
# show_memory()

f = np.memmap('memmapped.dat', dtype=np.float32, mode='r', shape=(nrows, ncols))
show_memory()                # Used: 602MB, Free: 2586MB
tensor = torch.tensor(f)
show_memory()                # Used: 985MB, Free: 2204MB, 说明构建 tensor 需要占用大量内存
```

### accelerte.hooks.attach_align_device_hook_on_blocks

**signature**
```python
attach_align_device_hook_on_blocks(
    module: nn.Module,
    execution_device: Optional[Union[torch.device, Dict[str, torch.device]]] = None,
    offload: Union[bool, Dict[str, bool]] = False,
    weights_map: Mapping = None,
    offload_buffers: bool = False,
    module_name: str = "",
    skip_keys: Optional[Union[str, List[str]]] = None,
    preload_module_classes: Optional[List[str]] = None,
)  # 给 module 加上 hook, 此函数本身无返回
```

`PretrainedModel.from_pretrain` 中调用 `dispath_model` 方法, 在内部调用 `attach_align_device_hook_on_blocks` 所传入的参数有如下说明：

- main_device 为 gpu 时, 假设 `device_map` 如下
    ```python
    device_map = {
        'embed': 0                              # Module, 包含权重 embed.weight
        'feed_forward.layers.0': 0,             # Module, 包含权重 feed_forward.layers.0.weight 与 feed_forward.layers.0.bias
        'feed_forward.layers.1.weight': 1,      # Parameter
        'feed_forward.layers.1.bias': "cpu",    # Parameter
        'feed_forward.layers.2': "cpu",         # Module
        'feed_forward.layers.3': "disk",        # Module
        'head': "disk"                          # Module, 包含 head.out Module
    }
    ```
    offload_buffers、module_name、skip_keys、preload_module_classes 通常为默认值, 剩余其他参数为:
    ```python
    # 伪代码
    zip(execution_device, offload) = {
        'embed.weight': (0, False),  # 注意, key 必须是 tensor 的 name 
        'feed_forward.layers.0.weight': (0, False),
        'feed_forward.layers.0.bias': (0, False),
        'feed_forward.layers.1.weight': (1, False),
        # weight_map 包含以下的 tensor
        'feed_forward.layers.1.bias': (0, True),   # 原本device_map在cpu上, execution_device 将其映射到main_device上
        'feed_forward.layers.2.weight': (0, True), # 原本device_map在cpu上, execution_device 将其映射到main_device上
        'feed_forward.layers.2.bias': (0, True),
        'feed_forward.layers.3.weight': (0, True), # 原本device_map在disk上, execution_device 将其映射到main_device上
        'feed_forward.layers.3.bias': (0, True),
        "head.out.weight": (0, True),
        "head.out.bias": (0, True)
    }
    ```

- main_device 为 cpu 时, 假设 `device_map` 如下
    ```python
    device_map = {
        'embed': "cpu"                         # Module, 包含权重 embed.weight
        'feed_forward.layers.0': "cpu",        # Module, 包含权重 feed_forward.layers.0.weight 与 feed_forward.layers.0.bias
        'feed_forward.layers.1.weight': "cpu", # Parameter
        'feed_forward.layers.1.bias': "cpu",   # Parameter
        'feed_forward.layers.2': "cpu",        # Module
        'feed_forward.layers.3': "disk",       # Module
        'head': "disk"                         # Module, 包含 head.out Module
    }
    ```
    offload_buffers、module_name、skip_keys、preload_module_classes 通常为默认值, 剩余其他参数为:
    ```python
    # 伪代码
    zip(execution_device, offload) = {
        'embed.weight': ("cpu", False),  # 注意, key 必须是 tensor 的 name 
        'feed_forward.layers.0.weight': ("cpu", False),
        'feed_forward.layers.0.bias': ("cpu", False),
        'feed_forward.layers.1.weight': ("cpu", False),
        'feed_forward.layers.1.bias': ("cpu", False),
        'feed_forward.layers.2.weight': ("cpu", False),
        'feed_forward.layers.2.bias': ("cpu", False),
        # weight_map 包含以下的 tensor
        'feed_forward.layers.3.weight': ("cpu", True), # 原本device_map在disk上, execution_device 将其映射到main_device上
        'feed_forward.layers.3.bias': ("cpu", True),
        "head.out.weight": ("cpu", True),
        "head.out.bias": ("cpu", True)
    }
    ```

## 简化版实现

```python
def from_pretrain(cls, pretrained_name_or_path, device_map: Dict):
    # 只考虑device_map含有gpu,cpu,disk, 且模型文件为shard的情形, 并假定load的参数的key与模型完全匹配
    
    config = ...
    # step 1: 初始化空模型文件
    with init_empty_weights():
        model = cls(config, ...)
    # step 2: 逐个load, 并初始化模型文件
    ...
    # 此时, 模型的权重的device, 一部分在gpu上, 一部分在cpu上, 一部分仍为 meta
    # step 3:
    model.tie_weights()
    model.eval()
    # step 4: dispatch (add hook)
```
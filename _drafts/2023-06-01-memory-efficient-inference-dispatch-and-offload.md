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

- 使用方法
- 文档补充：某些辅助函数官方文档缺失，某些对外API的参数说明不明确
- 源码分析

## 简介

针对大模型的**推理过程** 🤗 Transformer/accelerate 引入了一些 API, 使得大模型能在资源有限的服务器(即显存有限,内存有限)上运行, 运行的唯一的要求是有足够的磁盘空间以及时间, 注意模型仍然是在 raw pytorch 的环境下运行的, 而非通过量化及并行技术进行节省显存以及加速

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
max_memory = {0: "12GB", 1: "20GB", "cpu": "20GB"}
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
# device_map 的实际内容可由 model.hf_device_map 获取
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
# !!!这一步是必须的!!!(empty weight model 也可以tie weight?)
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

## <font color=red>原理总结</font>

这里先对上述代码的原理做一个介绍, 不想太过深入地学习源码只需要看这里的描述或者参考 [官方文档]([https://huggingface.co/docs/accelerate/v0.19.0/en/usage_guides/big_modeling]) 和 [官方博客](https://huggingface.co/blog/accelerate-large-models) 即可, 在上述 `from_pretrained` 的过程中, 实际上发生了如下几件事情:

- 首先初始化一个空模型
  
  注意这个空模型没有随机初始化的权重, 技术上来说, 实现上本质来源于 `pytorch>=1.9.0` 可以将 tensor 的 device 设置为 `torch.device("meta")`, 🤗 accelerate 引入了一个上下文管理器 `init_empty_weights`, 在上下文范围内, 将 `nn.Module.register_parameter` 与 `nn.Module.register_buffer` 做了替换, 使得所有模型的所有 parameter 与 buffer 的设备都被设置为了 `torch.device("meta")`
  ```python
  with init_empty_weights():
    model = cls(config, *model_args, **model_kwargs)
  ```
- 然后根据 model 中每个参数的 dtype 以及 shape 可以估算每个参数所需要的存储空间, 依据给定的 max_memory 决定每个模型的每个参数应该放在哪个 device 上, 即得到 device_map
  
  这部分涉及到的 API 主要是 🤗 accelerate 的 `infer_auto_device_map`, 以及所依赖的 `compute_module_sizes`, `get_max_layer_size` 等

- 接下来将参数文件逐个加载进cpu, 并将参数逐个放置在上一步骤所确定的设备上

  这一步骤主要涉及到一些小细节, 首先要保证 cpu 的内存能装下最大的参数文件, 即 `state_dict=torch.load("pytorch_model-00001-of-00004.bin")`, 然后将 `state_dict` 放置到 gpu/cpu/disk 上: 如果需要放置在 gpu 上, 则需要拷贝一份 tensor 至 gpu, 如果需要放置在 cpu 上, 实际上不会发生拷贝, 内存不会增加, 仅仅是 tensor 的引用计数加 1, 如果需要放置在 disk 上, 则会利用到 `numpy.memmap` 的功能将数据保存在硬盘上. 为了进一步节省 cpu 内存(`offload_state_dict=True`), 可以先将本应该放置在 cpu 上的 tensor 也先放置在硬盘上, 并记录好每个 tensor 与相应的硬盘文件的映射关系, 等全部的参数文件处理完成后, 再重新将本应该放在 cpu 上的 tensor 加载回 cpu, 并删除掉这部分的硬盘文件

- 修改 model 及其子模块的 forward 函数, 保证每个子模块在使用 forward 函数进行运算时, 相应的 tensor 操作数(输入及模型的权重)都在相同的设备上

  这一步骤的实现所使用的细节在官方文档中被称为 hook, 但个人认为这与 torch.nn.Module 中的 hook 还是有一定的区别, 称为修改 forward 函数可能更为准确. 需要注意的细节有: 首先会确定一个所谓的 main_device, main_device 一般为某一个 gpu.

  - 在进行运算前: 对于模型中原本在 disk 上保存的 tensor, 在进行运算时, 会首先将 disk 上的数据读入 cpu, 然后将其转移到 main_device 上; 对于模型中原本在 cpu 上保存的 tensor, 则将其转移到 main_device 上; 对于模型中原本就在 gpu 上的 tensor, 则不进行 tensor 的设备转移. 然后将输入的 tensor 也转移到相匹配的 gpu 上
  - 运算: 即普通的 CUDA 上的算子运算
  - 运算结束后: 对于模型中原本在 disk 上保存的 tensor, 则重新将 tensor 从 main_device 转移回 cpu 再保存会 disk, 对于模型中原本在 cpu 上保存的 tensor, 则 tensor 从 main_device 转移回 cpu, 对于模型中原本就在 gpu 上的 tensor, 则不进行 tensor 的设备转移.

  另一个细节是: 上述的 tensor 转移过程每次的转移范围是每次转移一个 submodule 下所有直接的 paramter 与 buffer. submodule 里的 submodule 的 paramter 和 buffer 则会在触发它本身的 forward 函数时被转移设备

  上述步骤所涉及的 API 主要是 🤗 accelerate 的 `dispatch_model`, 以及所依赖的 `OffloadedWeightsLoader`, `AlignDeviceHook` 等

通过以上步骤, 🤗 实现了将大模型的**推理过程**运行在资源有限的设备的目标

其他细节: 对于存在 tie_weight 情况的模型, 对这部分参数做了一些额外的特殊处理

备注：【以下API间的关系有些混乱，待研究】

- `PretrainedModel.from_pretrained(...)`
- `AutoModelForXXX.from_config(...)`
- `YYYModelForXXX._from_config(...)`
- `accelerate.init_empty_weight(...)`: 上下文管理器, 在上下文范围内, 修改 `nn.Module` 的 `register_parameter` 与 `register_buffer` 函数, 使得 `nn.Module` 中的 tensor 总是在 `torch.device("meta")` 上
- `accelerate.get_balanced_memory(...)`: 在设定 `device_map` 为 `auto/balanced/balanced_low_0` 时, 得到 `max_memory` 字典
- `accelerate.infer_auto_device_map(...)`: 根据 `max_memory` 字典得到 `device_map` 字典, 确定每个 module 或 parameter/buffer 的设备
- `accelerate.dispatch_model(...)`: 根据 `device_map` 字典增加 `forward` 的 hook
- `accelerate.load_checkpoint_and_dispatch(...)`: 并未在 🤗 Transformer 被使用 (与 from_pretrain 有些重复代码, 猜想应该是代码还需要重构好)

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

### accelerate.big_modeling.init_empty_weights

上下文管理器

```python
init_empty_weights(include_buffers: bool = False)
```

### accelerate.utils.modeling.infer_auto_device_map【逻辑比较复杂, 且有一些bug】

依赖的辅助 API 如下:

- accelerate.utils.modeling.compute_module_sizes
- accelerate.utils.modeling.get_max_layer_size
- accelerate.utils.modeling.find_tied_parameters, retie_parameters


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

**例子**

```python
class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1000, 1000))
        self.b = nn.Parameter(torch.rand(1000, 1000))
        self.layer = nn.Linear(1000, 1000)
    def forward(self, x):
        pass

print(infer_auto_device_map(ModelA(), max_memory={"cpu": 1000*1000*6}))  # {'': 'disk'}
print(infer_auto_device_map(ModelA(), max_memory={"cpu": 1000*1000*8+3999}))  # {'': 'disk'}
print(infer_auto_device_map(ModelA(), max_memory={"cpu": 1000*1000*8+4000}))  # {'a': 'cpu', 'b': 'disk', 'layer': 'disk'}
print(infer_auto_device_map(ModelA(), max_memory={"cpu": 1000*1000*10}))  # {'a': 'cpu', 'b': 'disk', 'layer': 'disk'}
```

注意这里第一个例子里, 按通常的想法, 至少 `self.a` 应该可以放在 cpu 上, 但这里要考虑 offload 的参数 `self.b` 和 `self.layer` 在运算时需要重新 load 回 cpu, 因此如果需要将 `self.a` 保留在 cpu 上, 应该至少要这么多内存:

```python
1000*1000*4 # 存放 self.a
+ max(
    1000*1000*4,  # 运算时需要临时加载 self.b
    1000*1000*4 + 1000*4  # 运算时需要临时加载 self.layer
)
```

### accelerate.big_modeling.dispatch_model

依赖的辅助 API 如下:
- accelerate.utils.modeling.check_device_map
- accelerate.utils.offload.OffloadedWeightsLoader, accelerate.utils.offload.PrefixDataset
- accelerate.hooks.attach_align_device_hook_on_blocks
    - accelerate.hooks.ModelHook, SequentialHook, AlignDeviceHook
    - accelerate.hooks.add_hook_to_module, (remove_hook_from_module)
    - accelerate.hooks.attach_align_device_hook, attach_execution_device_hook


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


### accelerate.utils.modeling.get_max_layer_size


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

### accelerte.hooks.attach_align_device_hook_on_blocks【里面的递归逻辑后面再补充】

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
    zip(execution_device, offload) = {
        "": 0,                                  # 这个是 dispath 函数在调用 attach_align_device_hook_on_blocks 之前追加的一个键
        'embed': (0, False)                         # Module, 包含权重 embed.weight
        'feed_forward.layers.0': (0, False),        # Module, 包含权重 feed_forward.layers.0.weight 与 feed_forward.layers.0.bias
        'feed_forward.layers.1.weight': (1, False), # Parameter
        # weight_map 包含以下模块里的所有 tensor, 注意 weight_map 的键是 tensor name
        'feed_forward.layers.1.bias': (0, True),    # Parameter, 原本device_map在cpu上, execution_device 将其映射到main_device上
        'feed_forward.layers.2': (0, True),         # Module
        'feed_forward.layers.3': (0, True),         # Module
        'head': (0, True)                           # Module, 原本device_map在disk上, execution_device 将其映射到main_device上
    }
    weights_map = OffloadedWeightsLoader(cpu_state_dict, offload_folder)
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
    zip(execution_device, offload) = {
        "": "cpu",                                  # 这个是 dispath 函数在调用 attach_align_device_hook_on_blocks 之前追加的一个键
        'embed': ("cpu", False)                         # Module, 包含权重 embed.weight
        'feed_forward.layers.0': ("cpu", False),        # Module, 包含权重 feed_forward.layers.0.weight 与 feed_forward.layers.0.bias
        'feed_forward.layers.1.weight': ("cpu", False), # Parameter
        'feed_forward.layers.1.bias': ("cpu", True),    # Parameter, 原本device_map在cpu上, execution_device 将其映射到main_device上
        # weight_map 包含以下模块里的所有 tensor, 注意 weight_map 的键是 tensor name
        'feed_forward.layers.2': ("cpu", True),         # Module
        'feed_forward.layers.3': ("cpu", True),         # Module
        'head': ("cpu", True)                           # Module, 原本device_map在disk上, execution_device 将其映射到main_device上
    }
    weights_map = OffloadedWeightsLoader(offload_folder)
    ```

### `torch.nn.Module` 的 hook 机制与 `torch.nn.Module.__call__`


[torch.nn.Module 源码](https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html), 由于这里我们只关注 forward hook, 所以简化如下:

```python
class Module:
    def __call__(self, *args, **kwargs):
        for hook_id, hook in (*_global_forward_pre_hooks.items(), *self._forward_pre_hooks.items()):
            if hook_id in self._forward_pre_hooks_with_kwargs:
                result = hook(self, args, kwargs)  # type: ignore[misc]
                if result is not None:
                    args, kwargs = result
            else:
                result = hook(self, args)
                if result is not None:
                    if not isinstance(result, tuple):
                        result = (result,)
                    args = result
        result = self.forward(*args, **kwargs)
        for hook_id, hook in (*_global_forward_hooks.items(), *self._forward_hooks.items()):
            if hook_id in self._forward_hooks_with_kwargs:
                hook_result = hook(self, args, kwargs, result)
            else:
                hook_result = hook(self, args, result)

            if hook_result is not None:
                result = hook_result
        return result
```

简言之, 对于 `torch.nn.Module` 的两类 forward hook 而言, 有如下说明, 假设 `result=forward(*args, **kwargs)`

- 只有在调用 `__call__` 时, hook 才会起作用, 直接调用 `forward` hook 不会起作用
- forward_pre_hook 的出入参可以是如下（注意 `args` 和 `kwargs` **可能会被修改**）:
  - 入参: **除去module本身外**, 只接收一个元组形式的参数 `args`, 出参可以是 None, tuple, 其他(最终转化为单个元素的tuple), 如果出参为 tuple, 则下一个 hook 的入参将使用当前 hook 的出参
  - 入参: **除去module本身外**, 接收元组形式的参数 `args` 和字典形式参数 `kwargs`, 出参可以是 None, (tuple, dict), 如果出参为 (tuple, dict), 则下一个 hook 的入参将使用当前 hook 的出参
- forward_hook 的出入参可以是如下（注意 `args` 和 `kwargs` **不会被修改**）:
  - 入参: **除去module本身外**, 一个元组形式的参数 `args` 和 `result`, 出参可以是 None, 其他, 如果出参不为 None, 则下一个 hook 所使用的 `result` 入参将使用当前 hook 的出参
  - 入参: **除去module本身外**, 一个元组形式的参数 `args`、字典形式参数 `kwargs` 和 `result`, 出参可以是 None, 其他, 如果出参不为 None, 则下一个 hook 所使用的 `result` 入参将使用当前 hook 的出参


注册 hook 的方式为:
```python
# 注意: 老版本的pytorch(例如torch==1.12.0), 没有with_kwargs参数, 即不能使用关键字参数
register_forward_pre_hook(
    self,
    hook: Union[
        Callable[[T, Tuple[Any, ...]], Optional[Any]],
        Callable[[T, Tuple[Any, ...], Dict[str, Any]], Optional[Tuple[Any, Dict[str, Any]]]],
    ],
    *,
    prepend: bool = False,  # 如果为 True, 则把优先级提到最高, 最先执行
    with_kwargs: bool = False  # hook的入参是否有关键字参数
)

def register_forward_hook(
    self,
    hook: Union[
        Callable[[T, Tuple[Any, ...], Any], Optional[Any]],
        Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
    ],
    *,
    prepend: bool = False,
    with_kwargs: bool = False,
)
```

示例

```python
# 这个例子适合新老版本torch
import torch.nn as nn
import torch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(20, 10)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(10, 5)
    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        return x
model = MLP()
for name, module in model.named_modules():
    module.register_forward_pre_hook(
        lambda self, args: print(self.__class__.__name__, len(args), type(args), args[0].shape),
        )
x = torch.rand((4, 20))
y = model(x)
# 输出:
# MLP 1 <class 'tuple'> torch.Size([4, 20])
# Linear 1 <class 'tuple'> torch.Size([4, 20])
# ReLU 1 <class 'tuple'> torch.Size([4, 10])
# Linear 1 <class 'tuple'> torch.Size([4, 10])
```


### accelerate.hooks.ModelHook, SequentialHook


### accelerate.hooks.AlignDevicesHook


```python
class AlignDevicesHook:
    def __init__(
        self,
        execution_device: Optional[Union[int, str, torch.device]] = None,
        offload: bool = False,
        io_same_device: bool = False,
        weights_map: Optional[Mapping] = None,  # 通常情况会指定
        offload_buffers: bool = False,
        place_submodules: bool = False,
        # skip_keys: Optional[Union[str, List[str]]] = None,  # accelerate 0.20 增加的参数, 为简单起见忽略
    ):

    def init_hook(self, module):
        pass
```

首先重复一下 `OffloadedWeightsLoader` 的内容: `self.weights_map` 只包含需要 offload 的参数: 如果 main_device 为 gpu, `self.weights_map` 参数既包含 cpu 上的参数, 也包含 disk 上的参数; 如果 main_device 为 cpu, 则 `self.device_map` 只包含 disk 上的参数. 在执行 `self.weights_map["tensor_name"]` 时, 如果原本的参数在 cpu 上保存, 则直接取出进行返回, 如果原本的参数是使用 `np.memmap` 保存在 disk 上, 则通过 `np.memmap` load 回 cpu, 然后再转化为 cpu 上的 tensor 进行返回, 总之返回的总是 cpu 上的 tensor.


要理解其余参数的含义, 这里先对 `nn.Module` 中的“tensor”进行澄清：

```python
class SubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.rand(40, 40))
        # 这种表示不持久化保存的buffer, 使用通常torch.save(model.state_dict(), "x.pth")将不会保存这部分buffer
        self.register_buffer("d", torch.ones(10, 10), persistent=False)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(100, 100))
        self.register_buffer("b", torch.rand(50, 50))
        self.layer = SubModule()

model = Model()
model.state_dict()                          # 包含 a, b, layer.c
list(model.named_parameters())              # 包含 a, layer.c
list(model.named_parameters(recurce=False)) # 包含 a
list(model.named_buffers())                 # 包含 b, layer.d
list(model.named_buffers(recurse=False))    # 包含 b
```

总的来说, 一个module的参数包含如下几类(后面的描述约定采用这些术语)

- parameter
- buffer
- child

**init_hook**

`init_hook` 的入参 `module` 应该与 `self.weights_map`、`self.offload_buffers`、`self.offload`、`self.place_submodules` 参数匹配 (感觉这里的API设计确实不太合理), 也就是说 `module` 里包含不存放在 meta device 上的参数与 `self.weights_map` 合并起来应该要包含全部的参数:

`self.offload=False`: 则将 parameter/buffer 移动到 `self.execution_device` 上, 如果 `self.place_submodules=True`, 则递归将 child 的 parameter/buffer 也移动到 `self.execution_device` 上

`self.offload=True`: `self.place_submodules=True` 表示如下操作也对 child 递归进行
    - 对buffer的处理: 如果 `self.offload_buffers=True`, 则表明 buffer 也被包含在 `self.device_map` 里, 同时意味着 `module` 里的 buffer 在 meta device 上, 此时将 `module` 的 buffer 移至 meta device 上; 如果 `self.offload_buffers=False`, 则表明 buffer 不被包含在 `self.device_map` 里, 同时意味着 `module` 里的 buffer 存放了实际数据, 此时将 `module` 的 buffer 移至 `self.execution_device` 上
    - 对parameter的处理: 将 `module` 的 parameter 移至 meta device 上

**pre_forward/post_forward**

`self.io_same_device=True`: 将入参先转移到 `self.execution_device` 上, 然后递归/不递归地将 `module` 的 parameter/buffer 全部转移到 `self.execution_device`, 然后正常进行 `forward`, 然后将所有需要 offload 的参数全部转移到 meta device 上, 最后将出参转移到跟入参相同的 device 上

`self.io_same_device=False`: 将入参先转移到 `self.execution_device` 上, 然后递归/不递归地将 `module` 的 parameter/buffer 全部转移到 `self.execution_device`, 然后正常进行 `forward`, 然后将所有需要 offload 的参数全部转移到 meta device 上, 最后出参保留在 `self.execution_device` 上

**detach_hook**

将 `module` 的参数的 device 全部恢复至 `init_hook` 之前的状态


### accelerate.hooks.add_hook_to_module, remove_hook_from_module

使用 accelerate 的 hook 机制, 会操纵原本的 `nn.Module` 的一些属性/方法

- `model._hf_hook: ModelHook`
- `model._old_forward: Callable`
- `model.forward`

### accelerate.hooks.attach_align_device_hook, attach_execution_device_hook


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

## 其他有用的 API

### `accelerate.big_modeling.cpu_offload_with_hook`, `accelerate.hooks.CpuOffload`, `accelerate.hooks.UserCpuOffloadHook`

注意前面提到的扮演重要角色的 `accelerate.hooks.AlignDevicesHook`, 当它作为某个 `nn.Module` 的 hook 时, 假设它在不执行 `forward` 函数时, 权重存储在 CPU 上 (这种情况称为 CPU offload). 在执行 `forward` 函数之前, hook 被触发, 将所有的入参及其本身的权重参数都被转移到了 `execution_device` 上 (通常是某一块 GPU 上), 而在执行完 `forward` 后, hook 会再次被触发, 将其本身的权重全部转移到 CPU 上 (根据配置的不同, 出参也许会被转移到与入参相同的设备上).

但在某些特殊情况下, 上述执行逻辑效率比较低, 例如:

```python
# module_a, module_b 均被加上了 AlignDevicesHook, 在不运行时, 权重保存在 CPU 上, 而运行设备为 GPU
for i in range(3):
    x = module_a(x)
for i in range(4):
    x = module_b(x)
```

在上述情况下, `module_a` 本身的权重会经历 3 次 CPU 到 GPU 的拷贝以及 3 次 GPU 到 CPU 的拷贝, `module_b` 本身的权重会经历 4 次 CPU 到 GPU 的拷贝以及 4 次 GPU 到 CPU 的拷贝, 然而更为优秀的策略是, 在进入第一个 `for` 循环之时, `module_a` 的权重进行一次 CPU 到 GPU 的拷贝, 而后权重一直保持在 GPU 上, 直至第二个 `for` 循环开始时, 首先将 `module_a` 的权重从 GPU 拷贝回 CPU, 而后将 `module_b` 的权重从 CPU 转移至 GPU, 而后权重一直保持在 GPU 上, 在第二个循环结束时, 将 `module_b` 的权重从 GPU 拷贝回 CPU. 这样我们就最大限度地降低了 CPU 与 GPU 之间地内存拷贝次数, 从而提升了运行效率.

那么如何实现呢? 首先我们需要定义一个半自动化的 `AlignDevicesHook`: 它只保证在 `forward` 函数执行之前, 模型参数被正确地转移到运行设备上, 并提供一个 `offload` 方法用以将参数重新从运行设备转移回 CPU, 但 `offload` 应该交由用户手动触发, 这样以来, 我们可能可以用类似于如下的伪代码实现之前提到的优化逻辑:

```python
attach_hook_to_module(module_a, hook_a)
attach_hook_to_moduel(module_b, hook_b)
for i in range(3):
    x = module_a(x)
hook_a.offload()
for i in range(4):
    x = module_b(x)
hook_b.offload()
```

现在来看 🤗 accelerate 的实现方案, 先看使用例子(完全借用自源码 [docstring](https://github.com/huggingface/accelerate/blob/v0.20.3/src/accelerate/big_modeling.py#L194)):

```python
model_1, hook_1 = cpu_offload_with_hook(model_1, cuda_device)
model_2, hook_2 = cpu_offload_with_hook(model_2, cuda_device, prev_module_hook=hook_1)
model_3, hook_3 = cpu_offload_with_hook(model_3, cuda_device, prev_module_hook=hook_2)

hid_1 = model_1(input)
for i in range(50):
    # model1 is offloaded on the CPU at the first iteration, model 2 stays on the GPU for this whole loop.
    hid_2 = model_2(hid_1)
# model2 is offloaded to the CPU just before this forward.
hid_3 = model_3(hid_3)

# For model3, you need to manually call the hook offload method.
hook_3.offload()
```

注意到, 这里的使用方式与我们最初假想的使用方式的最大区别在于 `model_1` 与 `model_2` 的 `hook_1` 与 `hook_2` 不需要手动用 `offload` 函数触发. 实现逻辑如下, `cpu_offload_with_hook` 函数的源码为:

```python
def cpu_offload_with_hook(
    model: torch.nn.Module,
    execution_device: Optional[Union[int, str, torch.device]] = None,
    prev_module_hook: Optional[UserCpuOffloadHook] = None,
):
    hook = CpuOffload(execution_device=execution_device, prev_module_hook=prev_module_hook)
    add_hook_to_module(model, hook, append=True)
    user_hook = UserCpuOffloadHook(model, hook)
    return model, user_hook
```

此处的 `CpuOffload` 是前面章节提到的 `accelerate.hooks.ModelHook` 的子类, 其仅包含 `pre_forward`, 用于保证执行 `forward` 函数之前, 入参及模型参数被转移到执行设备上. 而 `UserCpuOffloadHook` **不是** `accelerate.hooks.ModelHook` 的子类, 它仅包含 `offload` 函数. 注意, 这里的 trick 在于 `CpuOffload` 在执行 `forward` 函数被触发时, 除了转移本模型的参数, 还自动触发了 `prev_module_hook.offload` 函数, 由此实现自动化, 唯一需要手动触发 `offload` 的是最后一个 `nn.Module`. 这两个类的实现源码如下:

```python
class CpuOffload(ModelHook):
    def __init__(
        self,
        execution_device: Optional[Union[str, int, torch.device]] = None,
        prev_module_hook: Optional["UserCpuOffloadHook"] = None,
    ):
        self.prev_module_hook = prev_module_hook

        if execution_device is not None:
            self.execution_device = execution_device
        elif is_mps_available():
            self.execution_device = torch.device("mps")
        elif torch.cuda.is_available():
            self.execution_device = torch.device(0)
        else:
            self.execution_device = torch.device("cpu")

    def init_hook(self, module):
        return module.to("cpu")

    def pre_forward(self, module, *args, **kwargs):
        module.to(self.execution_device)
        if self.prev_module_hook is not None:
            self.prev_module_hook.offload()
        return send_to_device(args, self.execution_device), send_to_device(kwargs, self.execution_device)


class UserCpuOffloadHook:
    def __init__(self, model, hook):
        self.model = model
        self.hook = hook

    def offload(self):
        self.hook.init_hook(self.model)

    def remove(self):
        remove_hook_from_module(self.model)
```
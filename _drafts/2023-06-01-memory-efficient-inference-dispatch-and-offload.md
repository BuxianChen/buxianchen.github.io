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
    # low_cpu_mem_usage=True,
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


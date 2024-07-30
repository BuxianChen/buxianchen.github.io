---
layout: post
title: "(P1) huggingface cheatsheet"
date: 2023-11-01 11:12:04 +0800
labels: [huggingface]
---

## DataCollator

在 huggingface 的 trainer 里, dataset 一般是使用 datasets 包的 `load_dataset` 得到的 Dataset 经过 map 等方式处理. 而 dataloader 则一般是通过普通的 torch.utils.data.DataLoader 将 dataset 和这里的 datacollator 作为参数传入得到最终的 dataloader, 而 transformers 的 Model 的 forward 的输入就是 dataloader 的一个 batch

### transformers.DataCollatorForLanguageModeling

```python
import torch
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

tokenizer = AutoTokenizer.from_pretrained("./gpt-2/")
tokenizer.pad_token = tokenizer.eos_token

import torch
data = [
    {
        "input_ids": torch.tensor([1000, 1234, 5679]),
        "attention_mask": torch.tensor([1, 1, 1]),
        # "other_text": "text-A",  # 不允许, 必须预先删除
    },
    {
        "input_ids": torch.tensor([1000, 11111, 2345, 1234, 5679]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        # "other_text": "text-B",  # 不允许, 必须预先删除
    }
]
data_collator(data)

# {
#     'input_ids': tensor([[ 1000,  1234,  5679, 50256, 50256],
#         [ 1000, 11111,  2345,  1234,  5679]]),
#     'attention_mask': tensor([[1, 1, 1, 0, 0],
#         [1, 1, 1, 1, 1]]),
#     'labels': tensor([[ 1000,  1234,  5679,  -100,  -100],
#         [ 1000, 11111,  2345,  1234,  5679]])
# }
```

## 下载模型/数据相关

**验证下载的正确性(主要适用于手动点击下载)**

以下载 `bert-base-uncased/pytorch_model.bin` 文件为例

```python
from huggingface_hub import model_info
model_info("bert-base-uncased", revision="main", files_metadata=True)

# 输出结果里包含如下输出
# RepoFile: { 
#     {'blob_id': 'ba5d19791be1dd7992e33bd61f20207b0f7f50a5',
#      'lfs': {'pointerSize': 134,
#              'sha256': '097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a',
#              'size': 440473133},
#      'rfilename': 'pytorch_model.bin',
#      'size': 440473133}
```

检验本地下载的数据是否与上面的信息一致

```
sha256sum pytorch_model.bin  # 097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a
```

再举一例, 下载一个大规模的数据集, 下载方式采用手动点击链接的方式, 首先生成 lfs 文件的期望 hash 值

```python
# get_hash.py
from huggingface_hub import dataset_info
import json
repo_id = "Skywork/SkyPile-150B"
validate_filepath = "SkyPile-150B_sha256.json"
info = dataset_info(repo_id, revision="main", files_metadata=True)
sha256_info = {}
for file_info in info.siblings:
    filename = file_info.rfilename
    if file_info.lfs:
        sha256 = file_info.lfs['sha256']
        sha256_info[filename] = sha256
with open(validate_filepath, "w") as fw:
    json.dump(sha256_info, fw, ensure_ascii=False, indent=4)
```

验证 hash 值的代码: 每次下载完一些新的内容时, 执行一次此脚本进行验证

```python
# validate.py
import hashlib
import json
import os

def get_sha256(filepath):
    with open(filepath, "rb") as f:
        chunksize = 50 * 1024 * 1024
        m = hashlib.sha256()
        while True:
            chunk = f.read(chunksize)
            if not chunk:
                break
            m.update(chunk)
    return m.hexdigest()


validate_filepath = "SkyPile-150B_sha256.json"
state_filepath = "local_download_state.json"
root_path = "./"

with open(validate_filepath, "r") as fr:
    sha256_info = json.load(fr)
    
if os.path.exists(state_filepath):
    with open(state_filepath) as fr:
        state_dict = json.load(fr)
else:
    state_dict = {key: "undownload" for key in sha256_info}


for prefix, folders, filenames in os.walk(root_path):
    for filename in filenames:
        filepath = os.path.join(prefix, filename)
        relpath = os.path.relpath(filepath, root_path).replace("\\", "/")  # Windows
        if relpath in sha256_info:
            if state_dict[relpath] != "matched":
                expected_sha256_value = sha256_info[relpath]
                local_sha256_value = get_sha256(filepath)
                if expected_sha256_value == local_sha256_value:
                    print(relpath, "\033[0;32m matched \033[0m")  # green
                    state_dict[relpath] = "matched"
                else:
                    print(relpath, "\033[0;31m unmatched, please redownload !! \033[0m")  # red
                    state_dict[relpath] = "unmatched"
            else:
                print(relpath, "\033[0;32m skip validate \033[0m")
            sha256_info.pop(relpath)

print("="*40)
print(f"There are {len(sha256_info)} files should be download")

for relpath in sha256_info:
    print(relpath, "\033[0;33m should download !! \033[0m")
    
with open(state_filepath, "w") as fw:
    json.dump(state_dict, fw, ensure_ascii=False, indent=4)
```
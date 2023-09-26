---
layout: post
title: "(WIP) 🤗 Hub API"
date: 2023-09-20 07:20:24 +0800
labels: [huggingface, hub]
---

## 动机、参考资料、涉及内容

动机

- 🤗 Transformers `AutoModel.from_pretrained` 过程中的下载与缓存机制: 模型脚本下载后怎么动态加载, 脚本文件怎么缓存, 模型文件怎么缓存. 提前使用 git clone 将 Hub 中的模型库下载后再执行 `from_pretrained`, 跟不手动 git clone 之间的差别在哪 (在文件缓存方面)
- 🤗 Datasets `load_dataset` 过程里下载脚本, 动态 import, 下载数据, 将数据转化为 arrow 格式的具体逻辑
- 🤗 缓存目录有没有可能跟 git clone 的方式下载能做某种“相互转换”
- “玩转” huggingface 提供的 Hub 服务

涉及内容

- 🤗 Hub API
- 🤗 Transformers 中怎么利用 🤗 Hub API

参考资料

- 🤗 Hub 官方文档[https://huggingface.co/docs/huggingface_hub/guides/manage-cache](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)

## 产品逻辑

本篇博客的内容属于 🤗 的基础设施范畴, 这里想将 Huggingface 作为一个产品而言做一些解读, 以读者的认知范围, Huggingface 由这几部分构成

- Hub (服务): 包含 Models, Datasets, Spaces, 这三者首先是作为 git 远程仓库存在的, 因此 🤗 提供了一个 Git 仓库的托管平台, 而且类似于 GitHub, 而这个平台还具备一些额外功能, 例如: 权限管理, 每个 Dataset 仓库还有数据预览功能, 每个 Model 仓库一般都有模型卡片页, 帮助读者快速上手, Space 仓库还免费提供了将仓库内的代码部署的功能
- 软件与开发工具: 首先是 Hub API, 然后是各种下游库, 最知名的是 transformers 库

## Huggingface Hub

### 按图索骥

通读官方文档后, 感觉对下游库或者对基于 Huggingface Hub 进行开发比较有作用的 API


两类 API: `Repository`, `HfApi`

`HfApi` 的大致原理如下: 直接利用本地的单个文件或者单个版本对远程仓库进行操作

- `create_repo`
- `delete_repo`
- `create_branch`, `create_tag`, `delete_branch`, `delete_tag`: 远程创建/删除branch/tag
- `create_commit`: 底层接口, 下面四个底层都调用了 create_commit 方法, 除此之外, `metadata_update` 也使用了此方法
- `upload_file`, `upload_folder`, `delete_file`, `delete_folder`
- `hf_hub_download`:
  - 广泛用于 transformers 库中各种模型的权重转换脚本中, 例如 `transformers/models/videomae/convert_videomae_to_pytorch.py`
- `snapshot_download`


```
```

```python
from huggingface_hub import login, create_repo
token = "hf_xxxyyy"
login(token)
create_repo("Buxian/test-model")
```



```python
# huggingface-cli lfs-enable-largefiles
# 底层实际干的事:
lfs_config = "git config lfs.customtransfer.multipart"
LFS_MULTIPART_UPLOAD_COMMAND = "lfs-multipart-upload"
run_subprocess(f"{lfs_config}.path huggingface-cli", self.local_dir)
run_subprocess(
    f"{lfs_config}.args {LFS_MULTIPART_UPLOAD_COMMAND}",
    self.local_dir,
)

git config lfs.customtransfer.multipart.path huggingface-cli <local_dir>
git config lfs.customtransfer.multipart.args lfs-multipart-upload <local_dir>
```


### Upload


#### `upload_file`

```python
def upload_file(
  path_or_fileobj,
  path_in_repo,
  repo_id,
  token=None,
  repo_type=None,  # 只能是 model, dataset, space, 默认是 model
  revision=None,
  commit_message=None,
  commit_description=None,
  create_pr=None,
  parent_commit=None,
  run_as_future=False
):
  pass

# 希望对远程仓库的特定分支提交一个【增加一个文件的提交】: 分支必须已存在, 有可能会产生一个“空提交”
upload_file("hello.c", "c/hello.c", "Buxian/test-model", revision="main", commit_message="add hello.c")

# 建立一个 PR 请求:
# 方式1: 基于远程分支名建立: 在main分支的基础上创建一个增加一个文件的提交, 以此建立PR请求
upload_file(
  path_or_fileobj="hello.c", path_in_repo="c/hello.c", repo_id="Buxian/test-model", commit_message="(pr branch) add hello.c",
  create_pr=True, revision="main"
)

# 方式2: 基于特定的提交建立: 在parent_commit的基础上, 建立一个提交, 并请求合并至main
upload_file(
  path_or_fileobj="hello.c", path_in_repo="c/hello.c", repo_id="Buxian/test-model", commit_message="(pr parent commit) add hello.c",
  create_pr=True, revision="main", parent_commit="ea9c8da4cda73fb6456cef85627d789394354a29"
)
```


备注: git 中的 commit message 与 commit description 有什么不同: 本质上可以认为 commit message 是标题, commit description 是详细内容, 这里有一篇关于写好 commit message 的[博客](https://cbea.ms/git-commit/)

使用原生的 git cli 工具可以用这两种方式区分 commit message 和 commit description.
```bash
git commit -m "This is Title" -m "This is Description"
git commit  # 在弹出的文本编辑器中, 第一行是 commit message, 其余均为 commit description
```

在 GitHub/GitLab 的网页界面上, 一般在浏览文件夹时, 显示的是 commit message, 而查看某个版本的详细信息时可以看到 commit description.

备注: 如果在没有修改文件的情况下使用 `upload_file` 时, 仍然会为远端仓库增加一个“没有实际意义”的提交, 作用等效于
```bash
git commit --allow-empty -m "no file changed"
```

备注: 在使用 `create_pr=True` 的时候产生了两个疑问:
- 怎么持续为一个 pr 增加提交
- 怎么解决 pr 与需要合并的分支的冲突 (似乎只有用 Repository API 来做?)


#### `upload_folder`

类似于 `upload_file`, 不赘述太多

#### `create_commit` / `create_commit_on_pr`

`create_commit` 是 `upload_folder` 与 `upload_file` 在内部调用的方法. 本小节实质上是对 `uploader_folder` 的原理/源码进行解析, 本质上: 本地发送 HTTP 请求给 Hub 服务器, 本地已经打包了创建的 commit 相关的信息以及上传文件, Hub 服务器接收到请求后更新远端仓库


### Download

根据官方文档[https://huggingface.co/docs/huggingface_hub/guides/download](https://huggingface.co/docs/huggingface_hub/guides/download) 中描述的, 最主要的就是这两个函数

- `hf_hub_download`: 下载单个文件
- `snapshot_download`: 下载一个版本的多个文件

#### `hf_hub_download`

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="huggingface/label-files", filename="kinetics400-id2label.json", repo_type="dataset")
```

按照缓存目录结构下载单个文件
```
~/.cache/huggingface/hub/
├── datasets--huggingface--label-files
│   ├── blobs
│   │   └── 32cb9c6d5f5fe544580663ec11808e15c0ae2080
│   ├── refs
│   │   └── main
│   └── snapshots
│       └── 9462154cba99c3c7f569d3b4f1ba26614afd558c
│           └── kinetics400-id2label.json -> ../../blobs/32cb9c6d5f5fe544580663ec11808e15c0ae2080
└── version.txt
```

```python
@validate_hf_hub_args
def hf_hub_download(...)
```

`validate_hf_hub_args` 装饰器用于检查被装饰的函数的入参:

- 如果 `repo_id`, `from_id`, `to_id` 是函数的入参, 检查其传入的实参的值是满足条件的字符串: 至多只包含一个 `/`, 不包含 `--` 与 `__`, 以 `/` 分隔的两部分只能由 数字/字母/`.-_` 构成, 不能以 `.git` 结尾. 简单来说就是检查入参是一个合法的 repo_id
- 关于 `use_auth_token` 与 `token` 参数的兼容性检查, 具体细节不深究, 只需记住一点, 旧版本的参数一般是 `use_auth_token`, 未来版本最终计划弃用这个参数, 使用 `token` 作为入参


### 缓存目录

参考官方文档: [https://huggingface.co/docs/huggingface_hub/guides/manage-cache](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)

在官方文档中, 有两个术语:

- cache: 从 huggingface.co 下载的文件, 缓存地址默认在 `~/.cache/huggingface/hub`
- asset (asset cache): 一些下游库除了下载原始文件之外, 可能还需要做些后处理, 例如: 【待理清】
  - 🤗 Dataset 的 `load_dataset` 方法下载脚本后, 会执行脚本将数据以 arrow 的格式默认缓存在 `~/.cache/huggingface/dataset` 目录下
  - 🤗 Transformer 使用 `AutoConfig.from_pretrained(trust_remote_code=True)` 时, 会将 Hub 中的脚本缓存在 `~/.cache/huggingface/modules/transformers_modules` 目录下

从 Huggingface 各个项目之间的组织方式来考虑问题的话, Huggingface Hub 库其本身的定位, 以及与下游库 (例如: Huggingface transformers, huggingface datasets) 的关系, 在缓存目录的问题上, 主要是做这几件事

- 提供从 Hub 下载文件的 API, 下游库可复用这些接口


```python
from huggingface_hub import cached_assets_path

assets_path = cached_assets_path(library_name="datasets", namespace="SQuAD", subfolder="download")
something_path = assets_path / "something.json" # Do anything you like in your assets folder !
```


cache 文件结构目录, 也可参考官方示例: [https://huggingface.co/docs/huggingface_hub/guides/manage-cache#in-practice](https://huggingface.co/docs/huggingface_hub/guides/manage-cache#in-practice)
```
~/.cache/huggingface/hub
  - models--username--projectname/
    - refs/                  # 包含的是分支名对应的最新 commit-id
      - main                 # 文本文件, 实际存储的是对应的 commit-id, 例如: eee
      - dev                  # 文本文件, 实际存储的是对应的 commit-id, 例如: fff
    - blobs/
      - aaaaaaaaaaaaaaaaaaaaaaaaa
      - bbbbbbbbbbbbbbbbbbbbbbbbb
      - ccccccccccccccccccccccccc
      - ddddddddddddddddddddddddd
    - snapshots/  # 假设dev分支历史版本有fff和ggg
      - eee/
        - pytorch_model.bin  # 软连接至 blobs/aaaaaaaaaaaaaaaaaaaaaaaaa
        - README.md          # 软连接至 blobs/bbbbbbbbbbbbbbbbbbbbbbbbb
      - fff/
        - pytorch_model.bin  # 软连接至 blobs/aaaaaaaaaaaaaaaaaaaaaaaaa
        - README.md          # 软连接至 blobs/ccccccccccccccccccccccccc
      - ggg/
        - README.md
```

asset 文件结构示例: [https://huggingface.co/docs/huggingface_hub/guides/manage-cache#assets-in-practice](https://huggingface.co/docs/huggingface_hub/guides/manage-cache#assets-in-practice)

```
~/.cache/huggingface
    assets/
        └── datasets/
        │   ├── SQuAD/
        │   │   ├── downloaded/
        │   │   ├── extracted/
        │   │   └── processed/
        │   ├── Helsinki-NLP--tatoeba_mt/
        │       ├── downloaded/
        │       ├── extracted/
        │       └── processed/
        └── transformers/
            ├── default/
            │   ├── something/
            ├── bert-base-cased/
            │   ├── default/
            │   └── training/
    hub/
    └── models--julien-c--EsperBERTo-small/
        ├── blobs/
        │   ├── (...)
        │   ├── (...)
        ├── refs/
        │   └── (...)
        └── [ 128]  snapshots/
            ├── 2439f60ef33a0d46d85da5001d52aeda5b00ce9f/
            │   ├── (...)
            └── bbc77c8132af1cc5cf678da3f1ddf2de43606d48/
                └── (...)
    datasets/
    modules/
```


## 🤗 Transformers


### `transformers.utils.hub.try_to_load_from_cache`

输入: 🤗 Hub 的 repo-id; repo 的 commit-id/分支名; 文件名
输出: 检查本地的缓存目录中是否有满足输入条件的缓存, 如果有则返回实际路径, 没有则返回None或特殊值


```python
try_to_load_from_cache("username/projectname", "README.md", revision="dev")
# 返回结果是: "~/.cache/huggingface/models--username--projectname/snapshots/dev/README.md"
```


```python
def try_to_load_from_cache(
    repo_id: str,  # 一级或两级, 例如: bert-base, fnlp/bart-base-chinese
    filename: str,  # 文件名, 例如 pytorch_model.bin
    cache_dir: Union[str, Path, None] = None,  # 默认为 TRANSFORMERS_CACHE=~/.cache/huggingface/hub
    revision: Optional[str] = None,  # 分支名/commit-id, 注意不是文件本身的hash-id
    repo_type: Optional[str] = None,  # model/dataset/space
) -> Optional[str]:
    """
    Returns:
        `Optional[str]` or `_CACHED_NO_EXIST`:
            Will return `None` if the file was not cached. Otherwise:
            - The exact path to the cached file if it's found in the cache
            - A special value `_CACHED_NO_EXIST` if the file does not exist at the given commit hash and this fact was
              cached.
    """
    if revision is None:
        revision = "main"

    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE

    object_id = repo_id.replace("/", "--")
    if repo_type is None:
        repo_type = "model"
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    if not os.path.isdir(repo_cache):
        # No cache for this model
        return None
    for subfolder in ["refs", "snapshots"]:
        if not os.path.isdir(os.path.join(repo_cache, subfolder)):
            return None

    # Resolve refs (for instance to convert main to the associated commit sha)
    cached_refs = os.listdir(os.path.join(repo_cache, "refs"))
    if revision in cached_refs:
        with open(os.path.join(repo_cache, "refs", revision)) as f:
            revision = f.read()

    if os.path.isfile(os.path.join(repo_cache, ".no_exist", revision, filename)):
        return _CACHED_NO_EXIST

    cached_shas = os.listdir(os.path.join(repo_cache, "snapshots"))
    if revision not in cached_shas:
        # No cache for this revision and we won't try to return a random revision
        return None

    cached_file = os.path.join(repo_cache, "snapshots", revision, filename)
    return cached_file if os.path.isfile(cached_file) else None
```
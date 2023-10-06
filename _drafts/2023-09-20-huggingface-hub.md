---
layout: post
title: "(WIP) 🤗 Hub"
date: 2023-09-20 07:20:24 +0800
labels: [huggingface, hub]
---

## 动机、参考资料、涉及内容

动机

- 🤗 Transformers `AutoModel.from_pretrained` 过程中的下载与缓存机制: 模型脚本下载后怎么动态加载, 脚本文件怎么缓存, 模型文件怎么缓存. 提前使用 git clone 将 Hub 中的模型库下载后再执行 `from_pretrained`, 跟不手动 git clone 之间的差别在哪 (在文件缓存方面)
- 🤗 Datasets `load_dataset` 过程里下载脚本, 动态 import, 下载数据, 将数据转化为 arrow 格式的具体逻辑
- 🤗 缓存目录有没有可能跟 git clone 的方式下载能做某种“相互转换”
- 怎么在 🤗 Hub Python Library 之上开发新的项目: [https://huggingface.co/docs/huggingface_hub/guides/integrations](https://huggingface.co/docs/huggingface_hub/guides/integrations)
- “玩转” huggingface 提供的 Hub 服务

涉及内容

- 🤗 Hub
- 🤗 Hub Python Library
- 🤗 Transformers 中怎么利用 🤗 Hub Python Library

参考资料

- 官方文档

## Huggingface Hub

本篇博客的内容属于 🤗 的基础设施范畴, 这里想将 Huggingface 作为一个产品而言做一些解读, 以笔者的认知范围, Huggingface 由这几部分构成

- Hub (服务): 包含 Models, Datasets, Spaces, 这三者首先是作为 git 远程仓库存在的, 因此 🤗 提供了一个 Git 仓库的托管平台, 而且类似于 GitHub, 而这个平台还具备一些额外功能, 例如: 权限管理, 每个 Dataset 仓库还有数据预览功能, 每个 Model 仓库一般都有模型卡片页, 帮助读者快速上手, Space 仓库还免费提供了将仓库内的代码部署的功能
- 软件与开发工具: 首先是 🤗 Hub Python Library, 然后是各种下游库, 最知名的是 🤗 Transformers 库

下载模型

```bash
# 代理设置: ~/.bashrc
# WSL2
hostip=$(cat /etc/resolv.conf |grep -oP '(?<=nameserver\ ).*')
# Windows 本机
# hostip="127.0.0.1"
export HTTP_PROXY="http://${hostip}:7890"
export HTTPS_PROXY="http://${hostip}:7890"

GIT_LFS_SKIP_SMUDGE=0 git clone --no-checkout https://huggingface.co/Qwen/Qwen-14B-Chat-Int4
cd Qwen-14B-Chat-Int4
git lfs fetch --all
```

## Huggingface Hub Python Library

### 主要 API

通读官方文档后, 感觉对下游库或者对基于 Huggingface Hub 进行开发比较有作用的 API

两类 API: `Repository`, `HfApi`. 更推荐 `HfApi` 接口.

`HfApi` 的大致原理如下: 

- 对于文件上传操作, 直接利用本地的单个文件或者单个版本对远程仓库发送 HTTP 请求, 服务端 (即 Huggingface Hub) 处理请求 (例如: 操作远程仓库), 因此无需保存完整的 git 仓库备份.
- 对于下载文件操作, 这个库的作者设计了一个缓存目录结构来对下载的文件进行保存, 这种目录结构与传统的 git 仓库的 `.git` 目录略有不同, 算是借用了 git 中底层的一些内容进行按需简化.

具体比较重要的 API 如下:

- `create_repo`, `delete_repo`
- `create_branch`, `create_tag`, `delete_branch`, `delete_tag`: 远程创建/删除branch/tag
- `create_commit`, `create_commits_on_pr`: 底层接口, 下面四个底层都调用了 create_commit 方法, 除此之外, `metadata_update` 也使用了此方法
- `upload_file`, `upload_folder`, `delete_file`, `delete_folder`
- `hf_hub_download`:
  - 广泛用于 transformers 库中各种模型的权重转换脚本中, 例如 `transformers/models/videomae/convert_videomae_to_pytorch.py`
- `snapshot_download`

`Repository` 的大致原理

由于

原生 git 命令



### `HfApi` 接口列表以及对应的 🤗 Hub API

下面许多方法实际上是 `HfApi` 这个类的实例方法, 而 huggingface_hub python 库中示例化了一个 `HfApi` 实例, 并将这个实例的方法绑定为了顶级方法:

```python
api = HfApi()
upload_file = api.upload_file
create_commit = api.create_commit
# ...
```

大多数 🤗 Hub Python Library 中的 API 实质上都是对 HTTP 请求的封装, 请求 URL 一般类似于: `https://huggingface.co/api/{repo_type}s/{repo_id}/...`. 然而, 🤗 Hub 的官方文档 [https://huggingface.co/docs/hub/api](https://huggingface.co/docs/hub/api) 中所提供的 API 列表实际上并不全, 🤗 Hub Python Library 使用了许多未被记录在前面文档上的 API, 例如稍后会进行深入分析的 `huggingface_hub.create_commit`. 因此有必要对每个接口涉及到的 🤗 Hub API 做一个简单的梳理

- `endpoint`: 默认是 `https://huggingface.co`
- `model_type`: `model`/`dataset`/`space` 其中之一
- `repo_id`: 至多两级, 代表 `username/projectname`
- `revision`: 版本, 分支或者是commit id

以下是对应关系:

<table>
<tr>
  <td>作用</td>
  <td>🤗 Hub Python Library API</td>
  <td>🤗 Hub API</td>
  <td>备注</td>
</tr>
<tr>
  <td>创建 repo</td>
  <td>HfApi.create_repo</td>
  <td>{endpoint}/api/repos/create POST</td>
  <td></td>
</tr>
<tr>
  <td>移动 repo</td>
  <td>HfApi.move_repo</td>
  <td>{endpoint}/api/repos/move POST</td>
  <td></td>
</tr>
<tr>
  <td>删除 repo</td>
  <td>HfApi.delete_repo</td>
  <td>{endpoint}/api/repos/delete DELETE</td>
  <td></td>
</tr>
<tr>
  <td>修改仓库的可见性</td>
  <td>HfApi.update_repo_visibility</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/settings PUT</td>
  <td></td>
</tr>
<tr>
  <td>复制 space</td>
  <td>HfApi.duplicate_space</td>
  <td>{endpoint}/api/spaces/{from_repo_id}/duplicate POST</td>
  <td></td>
</tr>
<tr>
  <td>创建 tag</td>
  <td>HfApi.create_tag</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/tag/{revision} POST</td>
  <td></td>
</tr>
<tr>
  <td>删除 tag</td>
  <td>HfApi.delete_tag</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/tag/{tag} DELETE</td>
  <td></td>
</tr>
<tr>
  <td>创建 branch</td>
  <td>HfApi.create_branch</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/branch/{branch} POST</td>
  <td></td>
</tr>
<tr>
  <td>删除 branch</td>
  <td>HfApi.delete_branch</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/branch/{branch} DELETE</td>
  <td></td>
</tr>
<tr>
  <td>下载单个文件</td>
  <td>HfApi.hf_hub_download</td>
  <td>{endpoint}/{repo_id}/resolve/{revision}/{filename} HEAD</br>{endpoint}/{repo_id}/resolve/{revision}/{filename} GET</td>
  <td>前一个请求是为了获取需要下载的文件的准确信息, 第二个请求的 URL 是在第一个请求的响应结果里, 可能与第一个请求相同, 也可能不同, 参考后文简化版源码实现</td>
</tr>
<tr>
  <td>下载版本快照</td>
  <td>HfApi.snapshot_download</td>
  <td>{endpoint}/{repo_id}/resolve/{revision}/{filename} HEAD</br>{endpoint}/{repo_id}/resolve/{revision}/{filename} GET</td>
  <td>实质上是 repo_info 查询后, 对版本里的文件使用 hf_hub_download 方法来完成的</td>
</tr>
<tr>
  <td>创建 commit</td>
  <td>HfApi.create_commit</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/commit/{revision} POST</td>
  <td>还涉及到更多的 HTTP API 调用, 例如 lfs 文件查询及上传</td>
</tr>
<tr>
  <td>创建包含多次 commit 的 PR</td>
  <td>HfApi.create_commits_on_pr</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/commit/{revision} POST</td>
  <td>底层调用了 HfApi.create_commit, 另外还涉及到更多的 HTTP API 调用</td>
</tr>
<tr>
  <td>上传文件夹</td>
  <td>HfApi.upload_folder</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/commit/{revision} POST</td>
  <td>底层是 HfApi.create_commit</td>
</tr>
<tr>
  <td>上传文件</td>
  <td>HfApi.upload_file</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/commit/{revision} POST</td>
  <td>底层是 HfApi.create_commit</td>
</tr>
<tr>
  <td>删除文件</td>
  <td>HfApi.delete_file</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/commit/{revision} POST</td>
  <td>底层是 HfApi.create_commit</td>
</tr>
<tr>
  <td>删除文件夹</td>
  <td>HfApi.delete_folder</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/commit/{revision} POST</td>
  <td>底层是 HfApi.create_commit</td>
</tr>
<tr>
  <td>更新 card</td>
  <td>huggingface_hub.metadata_update</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/commit/{revision} POST</td>
  <td>实际上就是提交对 README 的修改, 因此实质上是调用 create_commit</td>
</tr>
<tr>
  <td>squash 历史 (将多个提交合并为一个提交)</td>
  <td>HfApi.super_squash_history</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/super-squash/{branch} POST</td>
  <td></td>
</tr>
<tr>
  <td>创建 discussion</td>
  <td>HfApi.create_discussion</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/discussions POST</td>
  <td></td>
</tr>
<tr>
  <td>创建 Pull Request</td>
  <td>HfApi.create_pull_request</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/discussions POST</td>
  <td>创建之后 PR 的状态是 draft</td>
</tr>
<tr>
  <td>修改 discussion/Pull Request 的状态</td>
  <td>HfApi.change_discussion_status</td>
  <td>{endpoint}/api/{repo_id}/discussions/{discussion_num}/status POST</td>
  <td>只能修改为 open 或 closed 两种状态</td>
</tr>
<tr>
  <td>在 discussion/Pull Request 上增加评论</td>
  <td>HfApi.comment_discussion</td>
  <td>{endpoint}/api/{repo_id}/discussions/{discussion_num}/comment POST</td>
  <td></td>
</tr>
<tr>
  <td>在 discussion/Pull Request 上编辑评论</td>
  <td>HfApi.edit_discussion_comment</td>
  <td>{endpoint}/api/{repo_id}/discussions/{discussion_num}/comment/{comment_id}/edit POST</td>
  <td>comment_id 可由 get_discussion_details 方法获取到</td>
</tr>
<tr>
  <td>在 discussion/Pull Request 上隐藏评论</td>
  <td>HfApi.hide_discussion_comment</td>
  <td>{endpoint}/api/{repo_id}/discussions/{discussion_num}/comment/{comment_id}/hide POST</td>
  <td>一旦评论被隐藏, 那么这条评论不能再被修改内容, 原始的内容也不会对外展示</td>
</tr>
<tr>
  <td>将 Pull Request 合并</td>
  <td>HfApi.merge_pull_request</td>
  <td>{endpoint}/api/{repo_id}/discussions/{discussion_num}/merge POST</td>
  <td></td>
</tr>
<tr>
  <td>查询仓库信息</td>
  <td>HfApi.repo_info</td>
  <td></td>
  <td>根据 repo_type 确定调用 model_info, dataset_info, space_info 其中之一</td>
</tr>
<tr>
  <td>查询 model 类型的仓库信息</td>
  <td>HfApi.model_info</td>
  <td>{endpoint}/api/models/{repo_id}/revision/{revision} GET<br/>{endpoint}/api/models/{repo_id} GET</td>
  <td>猜测: 不指定 revision 时会重定向至默认的 revision</td>
</tr>
<tr>
  <td>查询 dataset 类型的仓库信息</td>
  <td>HfApi.dataset_info</td>
  <td>{endpoint}/api/datasets/{repo_id}/revision/{revision} GET<br/>{endpoint}/api/datasets/{repo_id} GET</td>
  <td>猜测: 不指定 revision 时会重定向至默认的 revision</td>
</tr>
<tr>
  <td>查询 space 类型的仓库信息</td>
  <td>HfApi.space_info</td>
  <td>{endpoint}/api/spaces/{repo_id}/revision/{revision} GET<br/>{endpoint}/api/spaces/{repo_id} GET</td>
  <td>猜测: 不指定 revision 时会重定向至默认的 revision</td>
</tr>
<tr>
  <td>给一个 repo 进行 star</td>
  <td>HfApi.like</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/like POST</td>
  <td></td>
</tr>
<tr>
  <td>取消对一个 repo 的 star</td>
  <td>HfApi.unlike</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/like DELETE</td>
  <td></td>
</tr>
<tr>
  <td>列举所有 star 了的 repo</td>
  <td>HfApi.list_liked_repos</td>
  <td>{endpoint}/api/users/{user}/likes GET</td>
  <td></td>
</tr>
<tr>
  <td>列举所有满足条件的 dataset 仓库</td>
  <td>HfApi.list_datasets</td>
  <td>{endpoint}/api/datasets GET</td>
  <td></td>
</tr>
<tr>
  <td>列举所有满足条件的 model 仓库</td>
  <td>HfApi.list_models</td>
  <td>{endpoint}/api/models GET</td>
  <td></td>
</tr>
<tr>
  <td>列举所有满足条件的 space 仓库</td>
  <td>HfApi.list_spaces</td>
  <td>{endpoint}/api/spaces GET</td>
  <td></td>
</tr>
<tr>
  <td>获取某个revision的所有历史提交信息</td>
  <td>HfApi.list_repo_commits</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/commits/{revision} GET</td>
  <td></td>
</tr>
<tr>
  <td>获取仓库的所有 ref (tag 和 branch)</td>
  <td>HfApi.list_repo_refs</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/refs GET</td>
  <td>不包含 pr 所创建的分支</td>
</tr>
<tr>
  <td>获取某个 revision 下的所有文件信息</td>
  <td>HfApi.list_files_info</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/paths-info/{revision} POST</br>{self.endpoint}/api/{repo_type}s/{repo_id}/tree/{revision}/{encoded_path} GET</td>
  <td>首先通过第一个接口找出顶级目录下的文件和文件夹, 然后再通过第二个接口找出非顶级目录下的文件</td>
</tr>
<tr>
  <td>获取某个 revision 下的所有文件路径</td>
  <td>HfApi.list_repo_files</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/paths-info/{revision} POST</br>{self.endpoint}/api/{repo_type}s/{repo_id}/tree/{revision}/{encoded_path} GET</td>
  <td>调用 list_files_info, 只返回文件路径, 不返回其余信息</td>
</tr>
<tr>
  <td></td>
  <td>HfApi.list_metrics</td>
  <td></td>
  <td></td>
</tr>
<tr>
  <td>文件是否存在</td>
  <td>HfApi.file_exists</td>
  <td>{endpoint}/{repo_id}/resolve/{revision}/{filename} HEAD</td>
  <td></td>
</tr>
<tr>
  <td>仓库是否存在</td>
  <td>HfApi.repo_exists</td>
  <td></td>
  <td>调用 repo_info 实现</td>
</tr>
<tr>
  <td>查看已登录身份信息</td>
  <td>HfApi.whoami</td>
  <td>{endpoint}/api/whoami-v2 GET</td>
  <td></td>
</tr>
<tr>
  <td>获取 dataset 类型仓库的标签</td>
  <td>HfApi.get_dataset_tags</td>
  <td>{endpoint}/api/datasets-tags-by-type GET</td>
  <td>这里的标签是 language, task_categories 这类的标签 </td>
</tr>
<tr>
  <td>获取 model 类型仓库的标签</td>
  <td>HfApi.get_model_tags</td>
  <td>{endpoint}/api/models-tags-by-type GET</td>
  <td>这里的标签是 language, pipeline_tag 这类的标签</td>
</tr>
<tr>
  <td>获取仓库的所有 discussion 信息</td>
  <td>HfApi.get_repo_discussions</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/discussions?p={page_index} GET</td>
  <td></td>
</tr>
<tr>
  <td>获取一个 discussion 的详细信息</td>
  <td>HfApi.get_discussion_details</td>
  <td>{endpoint}/api/{repo_type}s/{repo_id}/discussions/{discussion_num} GET</td>
  <td></td>
</tr>
<tr>
  <td>查看 token 的权限信息</td>
  <td>HfApi.get_token_permission</td>
  <td>{endpoint}/api/whoami-v2 GET</td>
  <td>调用 whoami 接口实现, 返回结果是 write/read/None</td>
</tr>
<tr>
  <td>获取 space 的运行信息</td>
  <td>HfApi.get_space_runtime</td>
  <td>{endpoint}/api/spaces/{repo_id}/runtime GET</td>
  <td>包含运行状态, 运行资源等</td>
</tr>
<tr>
  <td>获取 space 的环境变量信息</td>
  <td>HfApi.get_space_variables</td>
  <td>{endpoint}/api/spaces/{repo_id}/variables GET</td>
  <td></td>
</tr>
<tr>
  <td>为 space 增加/修改环境变量</td>
  <td>HfApi.add_space_variable</td>
  <td>{endpoint}/api/spaces/{repo_id}/variables POST</td>
  <td></td>
</tr>
<tr>
  <td>为 space 删除环境变量</td>
  <td>HfApi.delete_space_variable</td>
  <td>{endpoint}/api/spaces/{repo_id}/variables DELETE</td>
  <td></td>
</tr>
<tr>
  <td>为 space 增加/修改秘密环境变量</td>
  <td>HfApi.add_space_secret</td>
  <td>{endpoint}/api/spaces/{repo_id}/secrets POST</td>
  <td></td>
</tr>
<tr>
  <td>为 space 删除秘密环境变量</td>
  <td>HfApi.delete_space_secret</td>
  <td>{endpoint}/api/spaces/{repo_id}/variables DELETE</td>
  <td></td>
</tr>
<tr>
  <td>删除 space 的持久化保存文件</td>
  <td>HfApi.delete_space_storage</td>
  <td>{endpoint}/api/spaces/{repo_id}/storage DELETE</td>
  <td></td>
</tr>
<tr>
  <td>为 space 请求持久化保存硬盘容量</td>
  <td>HfApi.request_space_storage</td>
  <td>{endpoint}/api/spaces/{repo_id}/storage POST</td>
  <td></td>
</tr>
<tr>
  <td>为 space 请求硬件资源</td>
  <td>HfApi.request_space_hardware</td>
  <td>{endpoint}/api/spaces/{repo_id}/hardware POST</td>
  <td></td>
</tr>
<tr>
  <td>重启 space</td>
  <td>HfApi.restart_space</td>
  <td>{endpoint}/api/spaces/{repo_id}/restart POST</td>
  <td></td>
</tr>
<tr>
  <td>停止 space</td>
  <td>HfApi.pause_space</td>
  <td>{endpoint}/api/spaces/{repo_id}/pause POST</td>
  <td></td>
</tr>
<tr>
  <td>设置 space 的睡眠时间</td>
  <td>HfApi.set_space_sleep_time</td>
  <td>{endpoint}/api/spaces/{repo_id}/sleeptime</td>
  <td></td>
</tr>
</table>


### Discussion/PR 相关

与 GitHub 相比, huggingface hub 中的 discussion (类似于 Github 中的 issue) 与 PR 没有明显分别:

huggingface hub: URL 前缀是相同的, 例如: `https://huggingface.co/Buxian/test-model/discussions/{discussion_num}` 既可能是一个 discussion, 也可能是一个 Pull Request. 并且在 huggingface hub 的 Web 页面上, 也没有将 discussion 和 PR 分别作为一个 Tab 页, 而是将它们统一编排在 Community 的 Tab 页下

Github: URL 前缀是不同的, 例如: `https://github.com/pytorch/pytorch/pull/{num}` 代表的是 PR, 而 `https://github.com/pytorch/pytorch/issues/{num}` 代表的是 issue, 需要注意的是, 这里的 `num` 的自增也是 PR 和 issue 混在一起的, 也就是说假设 `num=10`, 说明在这个 PR 或 issue 之前, 还有 9 个 PR 或 issue 已被创建. 在 Github 的 Web 页面上, PR 与 issue 分别是一个 Tab 页.

huggingface hub 中, Discussion / Pull Request 的状态分为四种: `["open", "closed", "merged", "draft"]`, 其中 `merged` 和 `draft` 仅适用于 PR, 状态变更如下:

- Discussion: `open` 与 `closed` 状态可相互切换
- Pull Request: `draft` -> `open` 或 `closed` 相互切换 -> `merged`
- Discussion 与 Pull Request 不能相互转换? 【待确认】


以下是一些接口 (全部都是 `HfApi` 的方法) 的使用说明, 同时适用于 model/dataset/space

```python
# create_discussion
# 默认pull_request 为 False, 而当取值为 True 时, 会在远程仓库建立类似refs/pr/6这种分支名, 然后创建的 discussion 会被标记为 Draft PR, 网页界面上会有操作指引:
# git clone https://huggingface.co/Buxian/test-model
# cd test-model && git fetch origin refs/pr/6:pr/6
# git checkout pr/6
# huggingface-cli login
# git push origin pr/6:refs/pr/6
# 在网页上点按钮将PR转换为正式状态
# 
# 具体可参考:
# https://huggingface.co/docs/hub/repositories-pull-requests-discussions
create_discussion(repo_id, title="title", description="content", pull_request=True)

# git clone 时不会 clone refs/pr/6 这个分支, 执行git fetch origin refs/pr/6:xxyy时, 目录结构会增加一个
# .git/refs/
# ├── heads
# │   ├── main  # 保存着 commit-id
# │   └── xxyy  # 保存着 commit-id


# create_pull_request
# 本质上, 就是调用 create_discussion 设定参数 pull_request=True 实现的
create_pull_request(repo_id, title="title", description="content")


# change_discussion_status
# 注意PR与Discussion的编号是混在一起的, 序号从1开始, 例如可能是这样
# https://huggingface.co/Buxian/test-model/discussions/1    PR
# https://huggingface.co/Buxian/test-model/discussions/2    Discussion
# https://huggingface.co/Buxian/test-model/discussions/3    PR
# 如果状态本身就是 closed, 那么会报错
change_discussion_status(repo_id, discussion_num=2, new_status='closed', comment='finish the discussion')

# comment_discussion
comment_discussion(repo_id,  discussion_num=2, comment="add comment")

# edit_discussion_comment

# hide_discussion_comment

# rename_discussion

# merge_pull_request
```

对 PR 继续提交代码: [官方文档](https://huggingface.co/docs/huggingface_hub/guides/community#push-changes-to-a-pull-request) 中描述暂无, 但实际上通过分析 `create_commits_on_pr` 的源码后, 发现其实很简单:

```python
discussion_num = 6
# from huggingface_hub import get_discussion_details
# discussion = get_discussion_details(repo_id=repo_id, discussion_num=6, repo_type=repo_type, token=token)
# revision = discussion.git_reference
create_commit(
    repo_id=repo_id,
    repo_type=repo_type,
    token=token,
    commit_message="update pr 2",
    revision=f"refs/pr/{discussion_num}",
    operations=operations,
    create_pr=False,
)
```

#### `create_discussion`



### 上传文件至 🤗 Hub 仓库

#### `upload_file/upload_folder/delete_file/delete_folder`

**<span style="color:red">使用说明</span>**

以下是 `upload_file` 的详细使用说明:

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
- 怎么解决 pr 与需要合并的分支的冲突 (似乎只有用 Repository API 来做? 可能也做不了, 只能用 git CLI)


`upload_folder/delete_file/delete_folder` 类似, 不赘述太多


**<span style="color:red">源码分析</span>**

以下几个方法最终的返回值都是一个 URL, 格式如下

```python
from huggingface_hub import hf_hub_url, upload_file, upload_folder
_staging_mode = _is_true(os.environ.get("HUGGINGFACE_CO_STAGING"))
ENDPOINT = os.getenv("HF_ENDPOINT") or ("https://hub-ci.huggingface.co" if _staging_mode else "https://huggingface.co")
endpoint = ENDPOINT  # 因此默认是: "https://huggingface.co"
f"{endpoint}/{repo_id}/resolve/{revision}/{filename}"   # hf_hub_url
f"{endpoint}/{repo_id}/tree/{revision}/{path_in_repo}"  # upload_file
f"{endpoint}/{repo_id}/blob/{revision}/{path_in_repo}"  # upload_folder
```

由于 `upload_file` 与 `upload_folder` 在本质上是调用 `create_commit` 和 `create_commits_on_pr` 进行实现的 (`delete_file` 与 `delete_folder` 类似, 此处不赘述), 大致的伪代码如下:

```python
def upload_file(...):
  operations = [CommitOperationAdd(...)]  # 每个文件一个operation, upload_file只涉及一个文件, 且只能是 CommitOperationAdd
  commit_info = create_commit(operations, ...)
  return f"{endpoint}/{repo_id}/blob/{revision}/{path_in_repo}"

def upload_folder(
  ...,
  delete_patterns, allow_patterns, ignore_patterns,
  multi_commits: bool = False, create_pr: bool = False
):
  delete_operations = self._prepare_upload_folder_deletions(..., delete_patterns)  # List[CommitOperationDelete]
  add_operations = self._prepare_upload_folder_additions(..., allow_patterns, ignore_patterns)  # List[CommitOperationAdd]
  commit_operations = delete_operations + add_operations
  
  # multi_commits 为 True, 则创建一个 Draft PR, 并可能进行多次提交
  if multi_commits:
    addition_commits, deletion_commits = plan_multi_commits(operations=commit_operations)
    pr_url = self.create_commits_on_pr(addition_commits, deletion_commits)
  else:
    commit_info = create_commit(operations, ...)
    pr_url = commit_info.pr_url
  return f"{endpoint}/{repo_id}/tree/{revision}/{path_in_repo}"
```

所以 `upload_file` 和 `upload_folder` 本质上只是构造了 `create_commit` 或 `create_commits_on_pr` 的入参 `operations`, 所有可能的 `operations` 在 huggingface_hub 中一共有三种:

```python
CommitOperationAdd     # upload_file/upload_folder, 可以是lfs文件或普通文件
CommitOperationDelete  # upload_folder/delete_file/delete_folder, 可以是lfs文件或普通文件
CommitOperationCopy    # 只有直接调用 create_commit 方法时才触发, 只能对lfs文件能进行此操作
```


#### `create_commit`

`create_commit` 是 `upload_folder/upload_file/delete_folder/delete_file` 在内部调用的方法.

`create_commit` 方法也是**对外接口**, 例如希望做一个类似如下的提交:

```
# 为远程仓库的main分支增加一个提交, 提交内容如下
1. 将本地的 test/exp.py 添加到远程仓库内的 test/exp.py
2. 将远程仓库 dev 分支的 pytorch_mode.bin 复制到 main 分支
3. 删除远程仓库的 deploy/app.py 文件
4. 删除远程仓库的 docker/Dockerfile 文件
```

这个提交不能使用 `upload_folder/upload_file/delete_folder/delete_file`, 只能调用 `create_commit` 来实现. 原因在于:

- 涉及到多个目录, 没有办法用 `uploader_folder` 实现
- 涉及到lfs的拷贝操作(`CommitOperationCopy`), 四个高阶 API 都没法处理

而 `create_commit` 本质上的执行逻辑是: 本地发送 HTTP 请求给 Hub 服务器, 本地已经打包了创建的 commit 相关的信息以及上传文件, Hub 服务器接收到请求后更新远端仓库

`create_commits_on_pr` 目前处于**实验阶段**, 个人认为不是**不是对外接口**, 仅在 `upload_folder` 中可能被调用, 用于分批进行文件提交(每次提交具体提交哪些文件由 huggingface_hub 内部方法决定: `plan_multi_commits` 方法).

**<span style="color:red">源码分析</span>**

一个简化(其实基本是抄源码)的实现见

[https://github.com/BuxianChen/snippet/blob/master/huggingface_hub/simple_hf_hub_download.py](https://github.com/BuxianChen/snippet/blob/master/huggingface_hub/simple_create_commit.py)

需要指出的是实际上涉及到的 http 请求的 API 文档我没有在[这里](https://huggingface.co/docs/hub/api)找到, 可能这些都是内部的 API 接口, 这里对执行步骤总结如下:

- 确定类型是 `CommitOperationAdd` 的文件是以普通文件还是 lfs 文件方式上传, 请求方式为:
  ```
  请求:
  url: f"{endpoint}/api/{repo_type}s/{repo_id}/preupload/{revision}"
  method: POST
  headers: {
    "user-agent": "mylib/v1.0; hf_hub/0.17.2; python/3.9.16; torch/1.12.1+cu113;",
    "authorization": f"Bearer {token}"
  }
  json: {
    "files":{
      [
        {"path": op.path_in_repo, "sample": op.upload_info.sample, "size": 234, "sha": op.upload_info.sha256}
      ]
    }
  }

  响应:
  {'files': [{'path': op.path_in_repo, 'uploadMode': 'regular'}]}
  ```
- 确定类型是 `CommitOperationCopy` 的文件的相关信息
  ```
  请求:
  通过 HfApi的list_file_repo

  响应:
  确定需要复制的文件确实存在, 并得到相关信息
  ```
- 将类型是 `CommitOperationAdd` 且为 lfs 的文件进行上传(如果需要的话), 并验证上传是否成功
  - STEP 1: 获取上传方式及验证方式
    ```
    请求:
    url: f"{endpoint}/{repo_id}.git/info/lfs/objects/batch"
    method: POST
    headers: {
      "Accept": "application/vnd.git-lfs+json",
      "Content-Type": "application/vnd.git-lfs+json",
    }
    json: {
      "operation": "upload",
      "transfers": ["basic", "multipart"],
      "objects": [
        {
          "oid": upload.upload_info.sha256.hex(),
          "size": upload.upload_info.size,
        }
        for upload in lfs_additions
      ],
      "hash_algo": "sha256",
    }

    响应:
    {
      "transfer": "basic"
      "objects": [
        {
          "oid": upload.upload_info.sha256.hex(),
          "size": upload.upload_info.size,
          "authenticated": True,
          'actions': {
            'upload': {
              'href': 'https://s3.us-east-1.amazonaws.com/lfs.huggingface.co/repos/...'
            },
            'verify': {
              'href': 'https://huggingface.co/Buxian/test-model.git/info/lfs/objects/verify',
              'header': {'Authorization': 'Basic xyzdef'}
            }
          }
        }
      ]
    }
    备注: xyzdef 是 token='hf_xyzdef'
    ```
  - STEP 2: 上传lfs文件
    ```
    # 方式一: 一次将单个文件上传完毕
    url: actions.upload.href
    method: PUT
    data: op.asfile()

    # 方式二: 一次只能上传一个文件一定大小的数据, 多次上传
    # 先分块上传
    url: actions.upload.header.values()[i]
    method: PUT
    data: op.asfile()[part_start:part_end]

    # 最后告知上传完成
    url: actions.upload.href
    method: POST
    headers: {
      "Accept": "application/vnd.git-lfs+json",
      "Content-Type": "application/vnd.git-lfs+json"
    }
    ```

  - STEP 3: 验证上传成功
    ```
    请求:
    url: actions.verify.href
    method: POST
    json: {"oid": operation.upload_info.sha256.hex(), "size": operation.upload_info.size}
    ```

- 创建提交并上传
  ```
  请求:
  url: f"{self.endpoint}/api/{repo_type}s/{repo_id}/commit/{revision}"
  method: POST
  headers: {
    "Content-Type": "application/x-ndjson",
    "user-agent": "mylib/v1.0; hf_hub/0.17.2; python/3.9.16; torch/1.12.1+cu113;",
    "authorization": f"Bearer {token}"
  }
  data: bytes
  params: {"create_pr": "1"} if create_pr else None
  # 备注: data 中的字节是由提交项的各个文件拼接起来的

  响应:
  {
    'success': True,
    'commitOid': '003e9ffb13bdb747b8a128abbcb5841964c1a054',
    'commitUrl': 'https://huggingface.co/Buxian/test-model/commit/003e9ffb13bdb747b8a128abbcb5841964c1a054',
    'hookOutput': ''
  }
  ```


#### `plan_multi_commits` 与 `create_commits_on_pr`

**<span style="color:red">源码分析</span>**

首先, 在目前版本 (huggingface_hub==0.17.3) 里, `multi_commits` 的特性还处于实验阶段:

- `create_commits_on_pr` 是一个对外接口 (默认 `HfApi` 实例的方法)
- `plan_multi_commits` 只在 `upload_folder` 中被使用, 并非对外接口
- [官方文档](https://huggingface.co/docs/huggingface_hub/v0.17.3/en/guides/upload#upload-a-folder-by-chunks) 里将这两个接口都处于实验阶段

先回顾一下 `upload_folder` 的源码:

```python
def upload_folder(
  ...,
  delete_patterns, allow_patterns, ignore_patterns,
  multi_commits: bool = False, create_pr: bool = False
):
  delete_operations = self._prepare_upload_folder_deletions(..., delete_patterns)  # List[CommitOperationDelete]
  add_operations = self._prepare_upload_folder_additions(..., allow_patterns, ignore_patterns)  # List[CommitOperationAdd]
  commit_operations = delete_operations + add_operations
  
  # multi_commits 为 True, 则创建一个 Draft PR, 并可能进行多次提交
  if multi_commits:
    addition_commits, deletion_commits = plan_multi_commits(operations=commit_operations)
    pr_url = self.create_commits_on_pr(addition_commits, deletion_commits)
  else:
    commit_info = create_commit(operations, ...)
    pr_url = commit_info.pr_url
  return f"{endpoint}/{repo_id}/tree/{revision}/{path_in_repo}"
```

目前版本的 `plan_multi_commits` 的函数定义为:

```python
def plan_multi_commits(
    operations: Iterable[Union[CommitOperationAdd, CommitOperationDelete]],
    max_operations_per_commit: int = 50,                        # 一个提交涉及的最大文件数
    max_upload_size_per_commit: int = 2 * 1024 * 1024 * 1024,   # 一个提交涉及的最大合计文件大小(仅适用于add操作)
) -> Tuple[List[List[CommitOperationAdd]], List[List[CommitOperationDelete]]]:
    ...
    return addition_commits, deletion_commits
```

其具体算法实际上在[docstring](https://huggingface.co/docs/huggingface_hub/v0.17.3/en/package_reference/hf_api#huggingface_hub.plan_multi_commits)中已经解释的比较明白, 此处再赘述一些要点:

- 首先 `plan_multi_commits` 入参中的 operation 只能是 `CommitOperationAdd` 和 `CommitOperationDelete` 类型, 例如有 5 个 add 操作与 55 个 delete 操作, 最终可能会拆成:
  ```python
  addition_commits = [[CommitOperationAdd(...), CommitOperationAdd(...)], [...]]        # 列表长度分别为: 2, 3
  deletion_commits = [[CommitOperationDelete(...), CommitOperationDelete(...)], [...]]  # 列表长度分别为: 50, 5
  ```
- 对于 delete 操作, 只是简单按每组最多删除 `max_operations_per_commit` 个文件进行分组
- 对于 add 操作, 如果单个文件就超过了 `max_upload_size_per_commit`, 那么这个文件单独做一次提交, 否则在提交数量不超过 `max_operations_per_commit` 以及提交文件大小合计不超过 `max_upload_size_per_commit` 的前提下进行分组

对于 `create_commits_on_pr` 的源码分析, 实质上是通过多次调用 `create_commit` 来完成的, 伪代码如下:

```python
pr = self.create_pull_request(...)  # Draft PR
# step: List[CommitOperationAdd] 或 List[CommitOperationDelete]
for step in list(remaining_deletions.values()) + list(remaining_additions.values()):
    # Push new commit
    self.create_commit(repo_id=repo_id, repo_type=repo_type, token=token, commit_message=step.id,
        revision=pr.git_reference, num_threads=num_threads, operations=step.operations, create_pr=False)
    # Update PR description
    self.edit_discussion_comment(repo_id=repo_id, repo_type=repo_type, token=token, discussion_num=pr.num,
        comment_id=pr_comment.id, new_content=...)
self.rename_discussion(repo_id=repo_id, repo_type=repo_type, token=token,
    discussion_num=pr.num, new_title=commit_message)
# 将 PR 状态修改为 open 状态
self.change_discussion_status(repo_id=repo_id, repo_type=repo_type, token=token,
    discussion_num=pr.num, new_status="open", comment=MULTI_COMMIT_PR_COMPLETION_COMMENT_TEMPLATE)
if merge_pr:  # User don't want a PR => merge it
    self.merge_pull_request(repo_id=repo_id, repo_type=repo_type, token=token,
        discussion_num=pr.num, comment=MULTI_COMMIT_PR_CLOSING_COMMENT_TEMPLATE)
```

### 从 🤗 Hub 仓库下载文件

根据官方文档[https://huggingface.co/docs/huggingface_hub/guides/download](https://huggingface.co/docs/huggingface_hub/guides/download) 中描述的, 最主要的就是这两个函数

- `hf_hub_download`: 下载单个文件
- `snapshot_download`: 下载一个版本的多个文件

`snapshot_download` 实际上是多次(实际实现时可以利用多线程/多进程加速这个过程)调用 `hf_hub_download` 来完成的, 而 `hf_hub_download` 本质上只是一个 stream 形式的 GET 请求, 然而, 理解 `hf_hub_download` 的实际行为的关键点在于理解 huggingface hub 的对下载下来的文件怎么存放(缓存目录结构设计)

#### 缓存目录

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

注意: 例如 `datasets` 库就没有使用 cached_assets_path 来确定默认的缓存目录, 而是用 `~/.cache/huggingface/dataset`


cache 文件结构目录, 也可参考官方示例: [https://huggingface.co/docs/huggingface_hub/guides/manage-cache#in-practice](https://huggingface.co/docs/huggingface_hub/guides/manage-cache#in-practice)
```
~/.cache/huggingface/hub
  - models--username--projectname/
    - refs/                  # 包含的是分支名对应的最新 commit-id
      - main                 # 文本文件, 实际存储的是对应的 commit-id, 例如: eeeeeeeee
      - dev                  # 文本文件, 实际存储的是对应的 commit-id, 例如: fffffffff
      - eeeeeee              # 文本文件, 实际存储的是对应的 commit-id, 例如: eeeeeeeee, 注意文件名是截断的 commit-id
    - blobs/
      - aaaaaaaaaaaaaaaaaaaaaaaaa
      - bbbbbbbbbbbbbbbbbbbbbbbbb
      - ccccccccccccccccccccccccc
      - ddddddddddddddddddddddddd
    - snapshots/  # 假设dev分支历史版本有fff和ggg
      - eeeeeeeee/
        - pytorch_model.bin  # 软连接至 blobs/aaaaaaaaaaaaaaaaaaaaaaaaa
        - README.md          # 软连接至 blobs/bbbbbbbbbbbbbbbbbbbbbbbbb
      - fffffffff/
        - pytorch_model.bin  # 软连接至 blobs/aaaaaaaaaaaaaaaaaaaaaaaaa
        - README.md          # 软连接至 blobs/ccccccccccccccccccccccccc
      - ggggggggg/
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

#### `hf_hub_download`

**<span style="color:red">使用说明</span>**

`hf_hub_download` 方法的定义如下:

```python
@validate_hf_hub_args
def hf_hub_download(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    local_dir: Union[str, Path, None] = None,
    local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
    user_agent: Union[Dict, str, None] = None,
    force_download: bool = False,
    force_filename: Optional[str] = None,
    proxies: Optional[Dict] = None,
    etag_timeout: float = 10,
    resume_download: bool = False,
    token: Union[bool, str, None] = None,
    local_files_only: bool = False,
    legacy_cache_layout: bool = False,
) -> str:
    ...
    return pointer_path  # 一般来说是类似 ~/.cache/huggingface/hub/{repo_type}s--{username}--{project-name}/snapshots/{commit-id}/{filename}
```

就笔者的观察, huggingface 代码库里很多函数的定义参数众多, 更有甚者有些参数直接是一个字典, 例如 `xx_kwargs`, 导致很难理解一个函数/类的准确行为, 时常令人有这样的困惑: 哪几个参数是配套用的? 同时设定 A 参数与 B 参数会怎样? 某个参数弃用了? 即使 huggingface 本身的 docstring 以及文档已经算是比较详细的情况下, 还是有许多参数是比较难以理解的. `hf_hub_download` 函数就是这种情况 (一共有 20 个参数), 因此需要首先解释其用法.

- `repo_id`, `filename`, `repo_type`, `revision`, `subfolder` 这几个参数自然不必多说, 是为了指定待下载的文件. 稍作解释的是 `subfolder` 参数, 实际上这个参数没有必要, 使用时如果文件位于 `a/b/c.txt`, 直接传入:
  ```python
  filename, subfolder="a/b/c.txt", None
  # 等价于
  filename, subfolder="c.txt", "a/b"
  # 等价于
  filename, subfolder="b/c.txt", "a"
  ```
- `endpoint`, `library_name`, `library_version`, `user_agent`, `proxies`, `token`, `etag_timeout`: 这几个参数是与发送的 HTTP 请求相关的, 除了 `token` 外其余均可暂时不管
- `cache_dir`: 指定缓存目录, 默认是 `~/.cache/huggingface/hub`
- `local_dir`, `local_dir_use_symlinks`: 是否将cache_dir中的内容“拷贝/软连接”出来一份, 方便查看和修改(如果是拷贝则可以不影响缓存目录), 与 `cache_dir` 参数有些“互动”, 因此一般会将 cache_dir 设置在一个不常手动打开查阅的统一位置, 而 `local_dir` 会设置在一些经常打开查看的地方, 以方便做些临时的修改.
- `force_download`, `resume_download`, `local_files_only`: 用于控制下载行为, 即强制重新下载/使用“断点续传”/只使用本地的缓存
- `force_filename`, `legacy_cache_layout`: 弃用参数, 不必理会

这里是上面的一些参数的可能取值:

- `revision`: 使用 branch/tag 名指定; 使用 commit-id 指定
- `local_dir`: 被设定时, `local_dir_use_symlinks` 取值为 `"auto"`/`True`/`False`
- `force_download`, `resume_download`, `local_files_only` 取值可以是 `True`/`False`
- 网络是否通畅

其中 `local_dir` 和 `local_dir_use_symlinks` 的逻辑如下: 在完成文件的下载后, 如果 `local_dir_use_symlinks` 默认被设置为了 `"auto"`, 如果目标文件是大文件(文件大小超过5MB, 由 `HF_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD` 环境变量决定), 则在 `local_dir` 保存 `cache_dir` 中该文件的软连接, 如果是小文件, 则 `local_dir` 中保存一份 `cache_dir` 中该文件的复制. 如果 `local_dir_use_symlinks=True`, 则无论文件大小, 都采用软连接, 如果 `local_dir_use_symlinks=False`, 则无论文件大小, 都从 `cache_dir` 中复制一份到 `local_dir` 中. 并且如果一旦指定了 `local_dir`, `hf_hub_download` 返回的文件路径会是 `local_dir` 内的文件路径, 以下所有情况都在最后执行前述逻辑.

情况1: 假设 `revision` 使用 commit-id 进行指定, 且本地已有该 commit-id 对应的缓存, 则直接返回 (注意 huggingface_hub 并不检查此文件是否被修改过).

情况2: 假设 `revision` 通过 branch/tag 进行指定, 且本地已有一份该 branch/tag 对应的缓存
- 情况2.1: 如果使用了 `local_files_only=True` 或者网络不通畅, 则使用本地的缓存文件(注意: 这样得到的文件可能不是最新的)
- 情况2.2: 首先发送一个 HTTP 请求查询远程的 branch/tag 是否被更新, 如果被更新, 则需要先修改本地的 `{cache_dir}/{sub_path}/refs/{branch}` 文件里的 commit-id 值, 然后执行下载文件的逻辑

情况3: 假设本地不存在指定的 `revision` 对应的缓存, 则先在 `{cache_dir}/{sub_path}/refs/{revision}` 中保存 commit-id 值 (除非 `revision` 是完整 commit-id), 然后执行下载文件下载逻辑


**<span style="color:red">源码分析</span>**

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

而 `hf_hub_download` 的主体部分可参考:

[https://github.com/BuxianChen/snippet/blob/master/huggingface_hub/simple_hf_hub_download.py](https://github.com/BuxianChen/snippet/blob/master/huggingface_hub/simple_hf_hub_download.py)

备注: 仅包含上一节的情况 3: 即本地完全没有缓存, 且不包含 `local_dir` 参数的逻辑

#### `snapshot_download`

`snapshot_download` 在源码实现的大致逻辑是:

- 调用 `HfApi.repo_info` 方法找到所有该 revision 的文件
- 逐个(可以使用多进程加速)文件使用 `hf_hub_download` 方法进行下载

### HfFileSystem

```python
# huggingface_hub.HfFileSystem (仅仅是对HfApi的一点封装)
# pip install pandas huggingface_hub
import pandas as pd
df = pd.read_csv("hf://Buxian/test-model/.gitattributes", sep=" ")
```

### Inference API

这个适用于 model 类型的仓库, 无需代码自动部署

```python
import json
import requests
# 只要这个model类型的仓库存在即可: https://huggingface.co/gpt2
API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": f"Bearer {token}"}
def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))
data = query("Can you please let us know more details about your ")
```

这看起来像是魔法, 怎么确定它是一个文本生成模型? 入参出参怎么确定的呢? 

任务类型怎么确定(上面的例子中是文本生成): 可具体参考[官方文档](https://huggingface.co/docs/hub/models-widgets)推荐的这份[伪代码](https://gist.github.com/julien-c/857ba86a6c6a895ecd90e7f7cab48046), 以下是一些发现:

```
# https://huggingface.co/bert-base-uncased/blob/main/config.json
# https://huggingface.co/bert-base-uncased
# 页面上 Inference API 上显示的是 Fill-Mask
{
    "architectures": ["BertForMaskedLM"]
}

# https://huggingface.co/internlm/internlm-chat-7b/blob/main/config.json
# https://huggingface.co/internlm/internlm-chat-7b
# 页面上 Inference API 上显示的是 Text Generation
{
  "architectures": [
    "InternLMForCausalLM"
  ],
  "auto_map": {
    "AutoConfig": "configuration_internlm.InternLMConfig",
    "AutoModel": "modeling_internlm.InternLMForCausalLM",
    "AutoModelForCausalLM": "modeling_internlm.InternLMForCausalLM"
  },
}
```

任务类型与请求出入参对应关系: [https://huggingface.co/docs/api-inference/detailed_parameters](https://huggingface.co/docs/api-inference/detailed_parameters)

### Inference Endpoint

这种适用于 Space 类型的仓库, 可完全控制部署的服务

### 杂项

#### 大文件处理


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

### 开发者工具及思考

本质上, 我们是需要与平台 (🤗 Hub) 进行交互: 这些交互包括

- 🤗 Hub 作为一个 Git 远程仓库, 版本管理功能: 上传文件, 下载文件, 合并请求, 评论
- 🤗 Hub 提供了将代码/模型实际运行起来的功能: Inference API (零额外代码情形) 和 Inference Endpoint (用户自己编写 server 代码)

本质上来说, 第二种交互实际上就是在第一种交互的基础上搭载上运行环境的机器, 因此核心还是第一种. 而第一种交互实质上有如下几种方法；

- 通过原生的 Git CLI 命令进行 `git clone`, `git add`, `git commit`, `git push` 等操作
- 通过 🤗 Hub Python Library 的 Repository 接口进行操作 (受限的 Git CLI, 并且不被官方推荐), 其实质上调用了 Git CLI 命令来完成操作
- 通过 🤗 Hub Python Library 的 HfApi 接口进行操作, 其实质上是调用了 🤗 Hub Server 端的功能, 而 Server 端的功能实现在底层应该也只是对 Git 的简单包装

使用 HfApi 与 Git CLI 的区别在于: 
- 首先如果使用 Git CLI, 则需要本地安装了 Git 这个软件, 而 HfApi 没有这个依赖项
- 其次, 使用 Git CLI 一般来说会保存远程仓库的一份完整备份, 这有利于在本地看到所有的仓库信息, 但比较占空间
- 从下载文件来说, HfApi 使用缓存结构, 可以不需要总是要自己决定某个文件下载到某个地方, 也没有这种困扰: 某个文件是不是之前下载过, 但不知道放到哪去了, 于是只能重新下载. 从文件上传来看, HfApi 直接将本地的某些文件上传, 本地不需要有 git 仓库.

假设情况如下: 主要需求是下载模型文件和数据集到一台不能联网的机器上, 希望保存的历史版本信息, 以应对被误删, 并且希望在需要的时候, 可以对不能联网的机器上保存的仓库文件拉取远程仓库最新的提交, 涉及到:
- bare repository (不需要checkout到workspace)
- bundle (只获取需要更新的内容)


怎么在这个库的基础上构建自己的项目, 官方文档中有一篇针对上传和下载文件的[集成指南](https://huggingface.co/docs/huggingface_hub/v0.17.3/en/guides/integrations), 里面提到了两种方案:

- 使用 `HfApi` 提供的接口 (例如: `upload_file`, `hf_hub_download`) 自己写一些辅助函数 (例如命名为: `load_from_hub`, `push_to_hub`)
- 继承 `huggingface_hub.ModelHubMixin` 类

而 🤗 Transformers 本质上是前一种实现方式:

- `PreTrainedModel`, `PratrainedConfig`, `PreTrainedTokenizerBase` 各自实现了一个 `from_pretrained` 方法, 此方法在底层调用了 `HfApi.hf_hub_download` 方法
- `PreTrainedModel`, `PratrainedConfig`, `PreTrainedTokenizerBase` 都继承了 `PushToHubMixin` 这个类, 这个类本质上主要就是定义 `push_to_hub` 这一个方法, 但这个类并没有继承 `huggingface_hub.ModelHubMixin`, 而 `push_to_hub` 在底层调用了 `HfApi.create_commit` 方法 (可能还会涉及到 `HfApi.create_repo` 和 `HfApi.create_branch` 方法的调用)

另外, 关于 assets 目录, 🤗 datasets 实际上也没有使用 huggingface_hub 里[推荐](https://huggingface.co/docs/huggingface_hub/v0.17.3/en/guides/manage-cache#caching-assets)的 `huggingface_hub.cached_assets_path` 接口做缓存根目录: `~/.cache/huggingface/assets/datasets`, 而是使用了 `~/.cache/huggingface/datasets` 这个目录作为缓存根目录


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

### `transformers.utils.hub.cached_file`

此方法是各种 `from_pretrained` 方法时触发从 Huggingface Hub 自动下载文件这一过程的核心方法之一

- `PreTrainedModel`, `PratrainedConfig`, `PreTrainedTokenizerBase` 各自实现了一个 `from_pretrained` 方法, 而它们最终都会落到对 `cached_file` 方法的调用
- auto-class 的 `from_pretrained` 方法实际上最终都是调用具体类 (例如: `BertConfig`) 的 `from_pretrained` 方法, 因此本质上还是对基类 `from_pretrained` 方法的调用

### `transformers.utils.hub.PushToHubMixin`

`PreTrainedModel`, `PratrainedConfig`, `PreTrainedTokenizerBase` 都继承了 `PushToHubMixin`, 以复用其实现的 `push_to_hub` 方法
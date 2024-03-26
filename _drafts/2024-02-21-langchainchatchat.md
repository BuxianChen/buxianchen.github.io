---
layout: post
title: "(P1) Langchain-ChatChat 浅析"
date: 2024-02-21 12:10:04 +0800
labels: [llm]
---

## 动机、参考资料、涉及内容

langchain-chatchat 浅析, 目录结构

```python
chains/  # 不确定用途, 仅有一个chain
configs/ 
  - basic_config.py
  - kb_config.py
  - model_config.py
  - prompt_config.py
  - server_config.py
document_loader/
  - FilteredCSVLoader.py  # 继承自 langchain.document_loaders.CSVLoader, 读取特定一列作为 content, 指定一列作为 source, 若干列为 metadata
  - mydocloader.py  # word 文档, 提取文字(docx模块) + ocr识别图片(rapidocr模块), 然后使用 unstructured.partition.text.partition_text 切分
  - myimgloader.py  # 图片, ocr识别图片(rapidocr模块), 然后使用 unstructured.partition.text.partition_text 切分
  - mypdfloader.py  # pdf 文件, 提取文字(pymupdf模块) + ocr识别图片(rapidocr模块), 然后使用 unstructured.partition.text.partition_text 切分
  - mypptloader.py  # ppt 文档, 提取文字(pptx模块) + ocr识别图片(rapidocr模块), 然后使用 unstructured.partition.text.partition_text 切分
  - ocr.py
embeddings/  # 不确定用途, 仅包含一点点内容
knowledge_base/
  - samples/
    - vector_store/  # 使用 faiss 做向量库时生成的文件夹
      - bge-large-zh-v1.5/
        - index.faiss
        - index.pkl
    - content/  # 知识库数据目录
      - llm/img/*.jpg
      - llm/*.md
      - test_files/*.{txt,pdf,csv,xlsx,jsonl}
      - wiki/*.md
  - info.db     # sqlite3 数据库, 不确定是啥
nltk_data/   # nltk 包需要的文件
server/      # 待探索
text_splitter/
  - ali_text_splitter.py                # 阿里的一个模型, 先根据标点符号换行符切为子句, 然后每个子句打一个二分类标签, 根据子句然后切割原始句子
  - chinese_recursive_text_splitter.py  # 继承自 langchain 的 RecursiveTextSplitter, 补充设置了一些标点符号作为默认分割符, 然后可能做了些不是太要紧的小逻辑修改
  - chinese_text_splitter.py            # 纯粹手写的一些正则规则
  - zh_title_enhanced.py                # 一些启发式的正则规则, 判断文字是否是标题, 此函数用在 server 目录里的一些地方
web_pages/   # 应该是 streamlit 页面
copy_config_example.py
init_database.py
startup.py
webui.py     # 不确定用法
shutdown_all.sh  # 不确定用法
release.py   # 不确定用途
```

## 入口: `startup.py`

```python
python copy_config_example.py
python init_database.py --recreate-vs
python startup.py -a  # 也即 --all-webui
```

`python startup.py` 的选项有:

<table>
<th>
    <td>all_webui</td>
    <td>all_api</td>
    <td>llm_api</td>
</th>
<tr>
    <td>openai_api</td>
    <td>True</td>
    <td>True</td>
    <td>True</td>
</tr>
<tr>
    <td>model_worker</td>
    <td>True</td>
    <td>True</td>
    <td>True</td>
</tr>
<tr>
    <td>api_worker</td>
    <td>True</td>
    <td>True</td>
    <td>True</td>
</tr>
<tr>
    <td>api</td>
    <td>True</td>
    <td>True</td>
    <td>False</td>
</tr>
<tr>
    <td>webui</td>
    <td>True</td>
    <td>False</td>
    <td>False</td>
</tr>
</table>

其他参数

- `--lite`: 将 `model_worker` 指定为 `False`, 表示不需要加载本地模型, 参考[文档](https://github.com/chatchat-space/Langchain-Chatchat/wiki/%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E9%83%A8%E7%BD%B2)通常用法为
  ```bash
  python startup.py -a --lite  # 启动 controller, openai-api 接口格式的接口, 外部模型(需要API KEY的那种模型), 向量库/对话管理/知识管理等, streamlit 前端 UI 服务
  ```
- `--model-name`: 指定模型名
- `--controller`: 跟 fastchat 有关
- `--quiet`: 减少 fastchat 的日志输出

`startup.py` 重点 (如果不关注本地模型启动的话) 只在于 `run_api_server` 与 `run_webui` 方法, 前者定义了许多 chain 的接口, 后者实际上起了个 `streamlit run webui.py ...` 的子进程.


```python
# 以下不完全遵循原始实现, 仅说明其架构
# startup.py
def run_controller(...):
    app = create_controller_app(...)
    uvicorn.run(port=FSCHAT_CONTROLLER["post"], ...)  # 20001 端口

def run_model_worker(model_name, ...):
    app = create_worker_app(model_name)
    # 本地模型占同一个端口: 20002, 其他线上的模型是不同的端口, 例如: qwen_api: 21006, gemini-api: 21010
    uvicorn.run(port=port, ...)

def run_openai_api():
    app = create_openai_app()
    uvicorn.run(port=FACHAT_OPENAI_API["port"], ...)  # 20000 端口

def run_api_server():
    from api import create_app
    app = create_app()
    uvicorn.run(port=API_SERVER["port"], ...)   # 7861 端口, 包含知识库管理/LangChain对话的后台接口

def run_webui():
    subprocess.Popen("streamlit run webui.py")

processes = {"online_api": [], "model_worker": []}

if args.openai_api:
    processes["controller"] = Process(target=run_controller, ...)
    processes["openai_api"] = Process(target=run_openai_api, ...)

if args.model_worker:  # 设置为lite时会关闭这个
    for model_name in args.model_name:
        if model_name in local_models:
            process = Process(target=run_model_worker, kwargs={"model_name": model_name})
            processes["model_worker"].append(process)
if args.api_worker:
    for model_name in args.model_name:
        if model_name not in local_models:
            process = Process(target=run_model_worker, kwargs={"model_name": model_name})
            processes["online_api"].append(process)
if args.api:
    processes["api"] = Process(run_api_server, ...)

if args.webui:
    processes["webui"] = Process(run_webui, ...)

# start all Process in processes
```

## `web_pages/`

## `server/`

### ThreadSafeObject, CachePool

关系如下:

```
        ThreadSafeObject    --------- Contains ------->      CachePool    ------- Contains ----->  _cache: OrderedDict(str, ThreadSafeObject)
                |                                                  /  \
                |                                                 /    \
          ThreadSafeFaiss                             EmbeddingPool    _FaissPool
                                                                          / \
                                                                         /   \
                                                            MemoFaissPool   KBFaissPool
                                                      (memo_faiss_pool)    (kb_faiss_pool)
```

- `server/knowledge_base/kb_cache/faiss_cache.py`:  全局变量 `memo_faiss_pool`, `kb_faiss_pool`, 类: `ThreadSafeFaiss`, `MemoFaissPool`, `KBFaissPool`
- `server/knowledge_base/`



以下是一个简化版的 `ThreadSafeObject`

```python
import threading
from contextlib import contextmanager
import time

class ThreadSafeObj:
    def __init__(self, obj):
        self.obj = obj
        self.lock = threading.RLock()

    @contextmanager
    def acquire(self):
        try:
            self.lock.acquire()
            yield self.obj
        finally:
            self.lock.release()

def do_op(op, name):
    global thread_safe_obj
    with thread_safe_obj.acquire():
        if op == "add":
            temp = thread_safe_obj.obj + 1
        elif op == "sub":
            temp = thread_safe_obj.obj - 1
        else:
            raise ValueError("")
        time.sleep(0.01)
        thread_safe_obj.obj = temp
        k = int(name.split("_")[1]) + 1
        if k % 20 == 0:
            print(thread_safe_obj.obj, name)

if __name__ == "__main__":
    thread_safe_obj = ThreadSafeObj(0)
    threads = []
    for i in range(200):
        if i % 2 == 0:
            threads.append(threading.Thread(target=do_op, args=("add", f"add_{i//2}")))
        else:
            threads.append(threading.Thread(target=do_op, args=("sub", f"sub_{i//2}")))

    for t in threads:
        t.start()

    for t in threads:
        t.join()
    print("finished:", thread_safe_obj.obj)
```

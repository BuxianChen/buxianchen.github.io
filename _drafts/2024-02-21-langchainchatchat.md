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

- `--lite`: 待研究
- `--model-name`: 指定模型名
- `--controller`: 大概跟 fastchat 有关
- `--quiet`: 减少 fastchat 的日志输出

`startup.py` 重点 (如果不关注本地模型启动的话) 只在于 `run_api_server` 与 `run_webui` 方法, 前者定义了许多 chain 的接口, 后者实际上起了个 `streamlit run webui.py ...` 的子进程. 可以先从 webui 看起

## `web_pages/`

## `server/`

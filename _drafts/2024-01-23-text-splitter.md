---
layout: post
title: "(Alpha) Text Splitter for LLMs"
date: 2024-01-23 17:10:04 +0800
labels: [llm]
---

## 动机、参考资料、涉及内容

LLM 在应用时通常会使用到 RAG 技术, 其核心组件是文档检索模块, 以向量检索为例, 需要使用用户问检索文档库, 检索质量极为重要 (尤其是召回率, 如果答案不在召回结果里, LLM 的回复结果是可预想地差). 通常的设定是我们手头上有一个开源的 Embedding 模型, 并且也不打算对它做微调, 因此提升检索质量只能从文档切分下手. `langchain`, `llama_index` 等 LLM 应用框架里内置了一些文档切分方法, 虽然在特定场景和数据场景下, 可能需要自己编写切分文档脚本, 但这些内置的切分方法也是一个不错的默认选项, 参考它们也有利于帮助写出特定场景下的切分文档脚本.

## `llama_index.node_parser.text.sentence.SentenceSplitter`

**分析入口**

```python
# 准备工作
# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'
from llama_index import SimpleDirectoryReader
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
idx = documents[0].text[:2000].rfind(".")
text = documents[0].text[:idx+1]
documents[0].set_content(text)
print(text)

from llama_index.node_parser.text.sentence import SentenceSplitter
from llama_index.schema import TextNode
from typing import List

splitter = SentenceSplitter.from_defaults(chunk_size=150, chunk_overlap=50)
text_nodes: List[TextNode] = splitter(documents)
```

**注意事项**

从下面的实现分析可以看出, `SentenceSplitter` 的默认参数 `chunking_tokenizer_fn` 使用的是 `nltk.tokenize.PunktSentenceTokenizer`, 用来将文档切分为句子, 而它对中文并不合适 (实际体验上看尤其是对中文的句号不敏感, 而对英文的句号过于敏感). 好在后续这个正则 `"[^,.;。？！]+[,.;。？！]?"` 对中文进行了考虑, 能适当缓解这一问题.

例子:

```python
from llama_index.schema import Document
text = "一、" + "你" * 50 + "。" + "二、" + "好" * 50 + "。" + "1." + "啊" * 50 + "2." + "哦" * 50

splitter = SentenceSplitter.from_defaults(chunk_size=60, chunk_overlap=30, tokenizer=lambda x: [1]*len(x))
doc = Document(text=text)
text_nodes = splitter([doc])
print([text_node.text for text_node in text_nodes])
```

输出:

```
['一、你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你你。',
 '二、好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好好。1.',
 '1.啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊2.',
 '哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦']
```


**框架说明**

`llama_index==0.9.36`

`SentenceSplitter` 通常会是在调用 `*Index.from_documents` 函数时被触发:

```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.schema import Document
from typing import List
data_dir = "./data"  # 目录下包含一堆 txt 文件
documents: List[Document] = SimpleDirectoryReader(data_dir).load_data()
index = VectorStoreIndex.from_documents(documents=documents)
```

`VectorStoreIndex.from_documents` 方法包含一个入参 `service_context`, 而 `service_context.tranformations` 是一个 `List[Callable[List[TextNode], List[TextNode]]]`, `VectorStoreIndex.from_documents` 会对传入的 `documents` 做处理:

```python
nodes = documents
for fn in service_context.transformations:
    nodes = fn(nodes)
```

而这个 `service_context.from_defaults` 函数在不传入 `transformations` 参数时会自动设为 `[SentenceSplitter(...)]`


**实现分析**

对于一个字符串 `text`, 以及给定的 `chunk_size`, 首先将 `text` 切分为每个都不超过 `chunk_size` 的小块 (tokenize 之后不超过 `chunk_size`, 并且这个过程各个小块是无重叠的, 且不损失原始 `text` 字符), 切分按如下顺序递归进行 (`SentenceSplitter._split`):

- 按 `paragraph_separator="\n\n\n"` 切分, 如果切分的小块满足长度限制, 则标记这一小块 `is_sentence=True`
- 按 `self._chunking_tokenizer_fn` 切分, 实质上是利用 `nltk.tokenize.PunktSentenceTokenizer`, 如果切分的小块满足长度限制, 则标记这一小块 `is_sentence=True`
- 按正则 `secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?"` 切分, 如果切分的小块满足长度限制, 则标记这一小块 `is_sentence=False`
- 按 `separator=" "` 即空格进行切分, 如果切分的小块满足长度限制, 则标记这一小块 `is_sentence=False`
- 最后用 `list(s)` 的方式, 即单个字符切分兜底, 标记 `is_sentence=False`

随后对前面切分的小块进行合并: 从左到右开始合并至尽量接近 `chunk_size`, 重叠 token 数也尽量接近 `chunk_overlap`

合并后的每个块的 token 数不超过 `chunk_size`, 重叠的 token 数不超过 `chunk_overlap`. 但目前 [0.9.36 版本](https://github.com/run-llama/llama_index/blob/v0.9.36/llama_index/node_parser/text/sentence.py#L259-L279) 的实现里有一种例外情况:

```python
# 例外情况示意
split_lengths = [90, 20, 110]
chunk_size = 120
chunk_overlap = 30
# _merge 的结果会是
result = [[90, 20], [20, 110]]  # 注意第2个chunk的长度超过了chunk_size=120的限制
```

```python
# 一个完整的例子说明上述例外情况
from llama_index.node_parser.text.sentence import SentenceSplitter
from llama_index.schema import Document
splitter = SentenceSplitter.from_defaults(chunk_size=120, chunk_overlap=30, tokenizer=lambda x: [1]*len(x))
doc = Document(text="你"*89+"."+"好"*19+"."+"啊"*110)
text_nodes = splitter([doc])
print([len(text_node.text) for text_node in text_nodes])  # [110, 130]
```

具体分析:

```python
import nltk
from typing import List

tokenizer = nltk.tokenize.PunktSentenceTokenizer()

def split(text: str) -> List[str]:
    spans = list(tokenizer.span_tokenize(text))
    sentences = []
    for i, span in enumerate(spans):
        start = span[0]
        if i < len(spans) - 1:
            end = spans[i + 1][0]
        else:
            end = len(text)
        sentences.append(text[start:end])

    return sentences
```

## `langchain.text_splitter.RecursiveCharacterTextSplitter`

**分析入口**

参考文档: [https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf)

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
pages = loader.load_and_split()  # RecursiveCharacterTextSplitter 发生在此处, 实际上最终只需关心 split_text 方法

# 以下为检索过程, 不是重点
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
docs = faiss_index.similarity_search("How will the community be engaged?", k=2)
for doc in docs:
    print(str(doc.metadata["page"]) + ":", doc.page_content[:300])
```

入口函数为 `RecursiveCharacterTextSplitter.split_text`, 其函数签名如下:

```python
class RecursiveCharacterTextSplitter(...):
    ...
    def split_text(self, text: str) -> List[str]: ...
```

**注意事项**

纯粹基于正则的切分, 可配合使用 `tokenizer` 来确定切分点

**框架说明**

```python
class BaseLoader(ABC):
    ...
    def load_and_split(self, text_splitter=None):
        _text_splitter = RecursiveCharacterTextSplitter()
        docs = self.load()
        return _text_splitter.split_documents(docs)  # split_documents 是重点
```

继承关系:

```
BaseDocumentTransformer: ABC  # 仅包含 transform_documents 抽象方法
TextSplitter: ABC
RecursiveCharacterTextSplitter
```

```
BaseLoader: ABC
BasePDFLoader
PyPDFLoader
```

**实现分析**

与 `llama_index.node_parser.text.sentence.SentenceSplitter` 类似, 分为 split 和 merge 过程, 按 `separators` 逐个切分后再尽可能长地 merge 至 `chunk_size`, 也尽可能长地将重合长度达到 `chunk_overlap`, 并且与 `llama_index` 不同, `langchain` 切分地 chunk 能更精确地保证不超过 `chunk_size`

## 比较 llama_index `SentenceSplitter` 和 langchain `RecursiveCharaterTextSplitter`

结论:

- 都包含 `split` 和 `merge` 两个过程, 并且 `split` 算法应该是相同的, 而 `merge` 算法 `llama_index` 更简化一些, 导致切分的长度可能会超出设定的 `chunk_size`
- `langchain` 对字符数的控制比 `llama_index` 更精确: `langchain` 切割后的片段长度总是不会超过 `chunk_size`, 而 `llama_index` 有时会超过 `chunk_size`
- `langchain` 在字符分割方面灵活度更高, `RecursiveCharaterTextSplitter` 参数的 `seperators` 可以按需指定, 并且还内置了一些关于各种编程语言的 `seperators`, 而 `SentenceSplitter` 只能通过 `paragraph_separator`, `secondary_chunking_regex`, `separator` 进行指定. 备注: `SentenceSplitter` 实际上可以通过自定义 `chunking_tokenizer_fn` 来插入更多分割符:
    ```python
    # langchain
    seperators = ["\n\n", "\n", " ", ""]  # 默认值
    # llama_index 示意
    paragraph_separator, secondary_chunking_regex, separator = ["\n\n\n", "[^,.;。？！]+[,.;。？！]?", " "]  # 默认值
    seperators = [paragraph_separator, chunking_tokenizer_fn, secondary_chunking_regex, separator, ""]
    ```
- `llama_index` 的自定义参数 `chunking_tokenizer_fn` 可以不只是一个正则切分, 而 `langchain` 的本质是纯字符的切分


比较代码:

```python
# llama_index==0.9.36, langchain==0.1.0

from llama_index.node_parser.text.sentence import SentenceSplitter
from llama_index.schema import Document

text = "xxx"
chunk_size = 200
chunk_overlap = 40

llama_index_splitter = SentenceSplitter.from_defaults(
    paragraph_separator="\n\n",
    chunking_tokenizer_fn=lambda x: [x],
    secondary_chunking_regex="[^\n]+[\n]?",
    separator=" ",
    tokenizer=lambda x: [1]*len(x),
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

splits = llama_index_splitter([Document(text=text)])
splits = [split.text for split in splits]
print(len(splits), max([len(split) for split in splits]))

from langchain.text_splitter import RecursiveCharacterTextSplitter
langchain_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],  # 默认值
    length_function=lambda x: len(x),    # 默认值
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
splits = langchain_splitter.split_text(text)
print(len(splits), max([len(split) for split in splits]))
```

## 阿里文本分割模型

模型地址: [https://www.modelscope.cn/models/iic/nlp_bert_document-segmentation_chinese-base/](https://www.modelscope.cn/models/iic/nlp_bert_document-segmentation_chinese-base/)

被用在: [langchain-chatchat: AliTextSplitter](https://github.com/chatchat-space/Langchain-Chatchat/blob/1fa714ee71940a25818c72b3e663d05ff9b3b19d/text_splitter/ali_text_splitter.py)

关键参数:

- 最大序列长度为 512
- 模型大小为 0.1B
- 语言: 中文/英文

局限性: 因为具体实现上会利用文本中的 `\n` 及标点符号进行预分割 (这一过程决定了最小单元), 因此如果是一段没有标点符号以及换行符的文本, 则必须自己实现预分割逻辑

```python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline(
    task=Tasks.document_segmentation,
    model='iic/nlp_bert_document-segmentation_chinese-base',
    device="cpu"
)

# 注意这个例子中 doc 不包含 \n, 但有标点符号, 最后的分割效果依然尚可
doc = '移动端语音唤醒模型，检测关键词为“小云小云”。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。模型输入为Fbank特征，输出为基于char建模的中文全集token预测，测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。模型训练采用“basetrain + finetune”的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景“小云小云”数据进行微调，得到最终面向业务的模型。后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型，但暂时未开放模型finetune功能。'
result = p.predict(documents=doc)

print(result[OutputKeys.TEXT])
```

**输出**

以 `\n\t` 分割

```python
	移动端语音唤醒模型，检测关键词为“小云小云”。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。模型输入为Fbank特征，输出为基于char建模的中文全集token预测，测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。
	模型训练采用“basetrain + finetune”的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景“小云小云”数据进行微调，得到最终面向业务的模型。后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型，但暂时未开放模型finetune功能。
	
```

**内部实现逻辑**

- 先用正则切分为小段: `p.cut_sentence(doc)`, 基本上就是按标点符号做的处理
    ```python
    # 源码: https://github.com/modelscope/modelscope/blob/master/modelscope/pipelines/nlp/document_segmentation_pipeline.py
    def cut_sentence(self, para):
        para = re.sub(r'([。！.!？\?])([^”’])', r'\1\n\2', para)  # noqa *
        para = re.sub(r'(\.{6})([^”’])', r'\1\n\2', para)  # noqa *
        para = re.sub(r'(\…{2})([^”’])', r'\1\n\2', para)  # noqa *
        para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)  # noqa *
        para = para.rstrip()
        return [_ for _ in para.split('\n') if _]
    ```
- 然后判断每个的标签: 只有两种标签 `['B-EOP', 'O']`
- 最后将标签合并: 实际上就是按 `B-EOP` 分割
    ```python
    docs = [d1, d2, d3, d4, d5, d6]
    labels = ["O", "B-EOP", "O", "O", "O", "B-EOP"]
    # 合并后
    segs = ["".join([d1, d2]), "".join([d3, d4, d5, d6])]
    ```

可以尝试探索这些

```python
p.cut_sentence(doc)  # 正则分割
p.preprocessor.label_list  # ['B-EOP', 'O']
```


## Unstructured (text splitter?)

用处:

- langchain: `langchain_community.document_loaders.unstructured.UnstructuredFileLoader` 使用 unstructured 包来作为无结构化文件的加载器
- langchain-chatchat: [ppt, image, pdf, doc](https://github.com/chatchat-space/Langchain-Chatchat/tree/1fa714ee71940a25818c72b3e663d05ff9b3b19d/document_loaders) 里的无结构文件加载器也是使用 unstructured 包

### 例子 1

```python
# 见前面 rapidocr 的例子
# 注意此处会下载 nltk 的模型: tokenizers/punkt.zip 与 taggers/averaged_perceptron_tagger.zip
from unstructured.partition.text import partition_text
text = "\n".join([r[1] for r in result])
result = partition_text(text=text)
```

**输出**

```
# result: 似乎是切文档然后做分类
[<unstructured.documents.elements.Title at 0x7fb378d0dd90>,
 <unstructured.documents.elements.Title at 0x7fb378d0de80>,
 <unstructured.documents.elements.Title at 0x7fb378d0df40>,
 <unstructured.documents.elements.Title at 0x7fb378d0df70>,
 <unstructured.documents.elements.Text at 0x7fb378d083a0>,
 <unstructured.documents.elements.Title at 0x7fb378d08430>,
 <unstructured.documents.elements.Text at 0x7fb378d0df10>,
 <unstructured.documents.elements.Text at 0x7fb378b6cbe0>]
```

备注: 不确定在中文上上的表现, 另外这个例子中 `text` 中包含了 `\n`, 似乎暗示着需要预分割且分割符为 `\n`

```python
# 可能不恰当的用法
text = "\t".join([r[1] for r in result])
result = partition_text(text=text)
# result: [<unstructured.documents.elements.NarrativeText at 0x7f3fbc816190>]
```

### 例子 2:

```python
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.documents.base import Document

loader = UnstructuredFileLoader(
    file_path="quant_methods_compare.py",  # https://github.com/BuxianChen/snippet/blob/master/quantization/quant_methods_compare.py
    autodetect_encoding=True,
    mode="elements",  # "single", "paged", "elements", 默认是 single, single 模式下一个文件就是一个 document
)

docs: List[Document] = loader.load()  # 会自动把多行合并为一个document
len(docs)  # 53
```
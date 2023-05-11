---
layout: post
title: "(WIP) Tokenizer 详解"
date: 2023-05-11 10:01:04 +0800
labels: [huggingface]
---

<style>
h2:after {
  content: "# ";
  color: gray;
}
h3:after {
  content: "## ";
  color: gray;
}
h4:after {
  content: "### ";
  color: gray;
}
h5:after {
  content: "#### ";
  color: gray;
}
.alert-red {
    padding: 1em;
    border: 1px solid #f44336;
    background-color: #ffebee;
    color: #f44336;
    /* font-weight: bold; */
    margin-top: 1em;
    margin-bottom: 1em
}
</style>

## 动机、参考资料、涉及内容

动机

- 熟悉 🤗 Tokenizers 的相关 API 与源码
- 熟悉 🤗 Transformers slow/fast tokenizer 的相关 API 与源码
- 适当补充相关知识

参考资料

- 🤗 Transformers 4.26.1 源代码
- 🤗 Transformers 官方文档
- 🤗 Tokenizers 官方文档


## 原理解析：Tokenizer

取决于不同的 tokenizer 实现, 🤗 Tokenizers 中的 Tokenizer 在encode阶段通常会进行如下几个步骤，具体实现细节见源码解析部分

```
# 以bert-base-uncased的fast版本为例
How are U today?
# Normalization
how are u today?
# Pre-tokenization
[how, are, u, today, ?]
# tokenize
[how, are, u, to, ##day, ?]
# Postprocess
[CLS, how, are, u, to, ##day, ?, SEP]
```

<div class="alert-red">
注意: 本节剩余部分的算法描述不保证与 🤗 Tokenizers 或 🤗 Transformers 中的 slow/fast 版中的实现完全吻合。原因是：
（1）🤗 Tokenizers 的确实现了以下的几种算法[参考官方文档](https://huggingface.co/docs/tokenizers/api/models)，但由于🤗 Tokenizers采用了 Rust 进行实现，笔者暂时无力理清准确的源码，所以没有深究
（2）🤗 Transformers 中的 slow/fast 版的 tokenizer 是为了对齐相应模型的原始实现，因此对于一个个具体的模型的 Tokenizer，有可能会对标准的 BPE/WordPiece/Unigram算法做些小改动。
</div>



### BPE

一个带有完整实现的教程：[🤗 NLP Course](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt)

BPE (Byte Pairwise Encoding) 算法的训练流程如下：
```
输入：句子列表，词表数量上限
（例子）：[" ".join(["hug"]*10), " ".join(["pug"]*5), " ".join(["pun"]*12)]
前处理：将句子列表转换为词列表，并统计词频。同时记录所有出现的字符作为base_vocab：最终得到的结果为：[(词语1, 词频1), ..., (词语N, 词频N)], base_vocab: [字符1, ..., 字符K]
（例子）：以空格作为分隔符进行切词，得到[("hug": 10), ("pug", 5), ("pun", 12)], base_vocab: ["h", "u", "g", "p", "n"]
训练流程：
  首先初始化所有词语的当前拆解方式：{词语1: ([字符1,...,字符k_1], 词频1), ..., 词语N: ([字符1,...,字符k_N], 词频N)}, 当前merge列表为: []
  （例子）：{hug: ([h, u, g], 10), pug: ([p, u, g], 5), pun: ([p, u, n], 12)}

  While True:
    根据当前词的拆解方式计算候选的merge列表及对应的频数, 候选的merge列表指的是所有词语当前拆解方式
    （例子-第1轮）：候选merge列表为：[(h, u): 10, (u, g): 15, (p, u): 17, (u, n): 12]
    （例子-第2轮）：候选merge列表为：[(h, u): 10, (u, g): 10, (pu, g): 5, (pu, n): 12]
    选出词频最大的merge方式, 加入至merge列表, 并对所有词语的拆解方式做更新
    （例子-第1轮）：将原始的词语拆解方式用 [p, u] -> pu更新，得到：{hug: ([h, u, g], 10), pug: ([pu, g], 5), pun: ([pu, n], 12)}, merge列表为: [(p, u)]
    （例子-第2轮）：将原始的词语拆解方式用 [pu, n] -> pun更新，得到：{hug: ([h, u, g], 10), pug: ([pu, g], 5), pun: ([pun], 12)}, merge列表为: [(p, u),(pu, n)]
    循环直至（merge列表长度+base_vocab长度）达到词表数量上限
```

推理流程如下
```
输入：句子，base_vocab与合并规则
（例子）：base_vocab与合并规则：[h, u, g, p, n, (p, u), (h, u), (hu, g)]
前处理：将句子拆解为词语列表
推理流程：
  tokens = []
  for word in sentence:
    word_split = [字符1, ..., 字符k]
    （例子）：word_split = [h,u,g,i,h,u]
    for merge in merges:
      尝试将merge应用于word上, 并更新word_split
      （例子-第1轮）：尝试使用(p, u)合并，word_split不变
      （例子-第2轮）：尝试使用(h, u)合并，word_split变为[hu, g, g, i, hu]
      （例子-第3轮）：尝试使用(hu, g)合并，word_split变为[hug, g, i, hu]
    tokens.extend(word_split)
```

### WordPiece

一个带有完整实现的教程：[🤗 NLP Course](https://huggingface.co/learn/nlp-course/chapter6/6?fw=pt)

WordPiece 算法是 Bert 所用的 tokenize 算法

<div class="alert-red">
正如[🤗 NLP Course](https://huggingface.co/learn/nlp-course/chapter6/6?fw=pt)指出的那样, Google 并未将 WordPiece 的训练算法进行开源，但推理算法是开源的，推理算法可参考[bert源码](https://github.com/google-research/bert/blob/master/tokenization.py)。因此严格地说，WordPiece 的训练算法只是猜测。
</div>

与 BPE 算法的主要区别在于：
- 中间字符采用 "##" 开头表示
- 训练阶段
  - 选取merge时，判断最大值的标准变为: 合并出现的次数/(piece1的次数*piece2的次数)
  - 不保留merge的二元组, 只保留最终结果
- 推理阶段
  - 贪心算法匹配每个词的剩余字符

WordPiece 算法的训练流程如下：

```
输入：句子列表，词表数量上限
（例子）：[" ".join(["hug"]*10), " ".join(["pug"]*5), " ".join(["pun"]*12)]
前处理：将句子列表转换为词列表，并统计词频。同时记录所有出现的字符作为vocab，包括出现在开头的字符与出现在中间的字符：最终得到的结果为：[(词语1, 词频1), ..., (词语N, 词频N)], vocab: [字符1, ..., 字符K]
（例子）：以空格作为分隔符进行切词，得到[("hug": 10), ("pug", 5), ("pun", 12)], vocab: ["h", "##u", "##g", "p", "##n"]
训练流程：
  首先初始化所有词语的当前拆解方式：{词语1: ([字符1,##字符k_2...,##字符k_1], 词频1), ..., 词语N: ([字符1,##字符2...,##字符k_N], 词频N)}
  （例子）：{hug: ([h, ##u, ##g], 10), pug: ([p, ##u, ##g], 5), pun: ([p, ##u, ##n], 12)}
  While True:
    根据当前词的拆解方式计算候选的merge列表及对应的分数(合并后出现的频数/合并前的频数之积), 候选的merge列表指的是所有词语当前拆解方式
    （例子-第1轮）：候选merge列表为：[(h, ##u): 10/(10*27), (##u, ##g): 15/(27*15), (p, ##u): 17/(17*27), (##u, ##n): 12/(27*12)]，这个例子比较特别，分数全部相同
    （例子-第2轮）：候选merge列表为：[(hu, ##g): 10/(10*15) , (p, ##u): 17/(17*17), (##u, ##g): 5/(17*15), (##u, ##n): 17/(17*12)]，最大分数的合并方式为(##u, ##n)
    选出词频最大的merge方式, vocab列表, 并对所有词语的拆解方式做更新
    （例子-第1轮）：将原始的词语拆解方式用 [h, ##u] -> hu更新，得到：{hug: ([hu, ##g], 10), pug: ([p, ##u, ##g], 5), pun: ([p, ##u, ##n], 12)}。vocab.append("hu")
    （例子-第2轮）：将原始的词语拆解方式用 [##u, ##n] -> ##un更新，得到：{hug: ([hu, ##g], 10), pug: ([p, ##u, ##g], 5), pun: ([p, ##un], 12)}。vocab.append("##un")
    循环直至vocab达到词表数量上限
```

推理流程如下（其实是简单的贪心策略，尽量匹配词表里最长的字串，如果某一步碰到OOV，则这个词的剩余部分被标记为UNK），准确代码可直接参考[Bert原始代码](https://github.com/google-research/bert/blob/master/tokenization.py)
```
输入：句子，vocab
前处理：将句子拆解为词语列表
推理流程：
  tokens = []
  for word in sentence:
    start=0, end=len(word)
    while start < len(word):
      while end > start:
        if word[start:end] in vocab:
          tokens.append(word[start:end])
          end -= 1
          start = end
          break
        if end == start:
          这种情况下把整个后续token都作为[unk]，不再进行进一步的分词
```

### Unigram

这里按照 [🤗 nlp course](https://huggingface.co/learn/nlp-course/chapter6/7?fw=pt) 中的描述对算法进行简要介绍。

训练流程：

- 前处理：将句子分割为词
- 首先将所有出现的单个字符作为base-vocab，然后使用一些方法获取到一个相对比较大的词表vocab（教程的代码里采用的是所有出现的词的子序列，并指出实际使用时可以采用BPE算法），其中vocab包含base-vocab。并且计算vocab中每个token出现的频率，供后续计算损失时使用。
- 对于vocab中的每个非base-vocab中的词，计算这个词从词表中排除后，整体损失的增长量，丢弃增长量最大的前20%的vocab。重复此步骤直至词表大小满足要求

给定一个词表，这个词表在数据集上的损失定义为数据集中所有词的损失按词频加权平均，而每个词的损失为：

$$
L(word)=\max_{\bold{x}\in S(word)}[-\sum_{i}log(p(x_i))]
$$

这里 $S(word)$ 表示的是按照 vocab，所有能拼凑成 $word$ 的 subword 序列。

推理流程：

(1) one-best-decoding: 即计算每个词的损失时找到的最优 subword 序列，这可以用动态规划（维特比算法）来解决，具体过程从略。
(2) k-best-decoding: [🤗 nlp course](https://huggingface.co/learn/nlp-course/chapter6/7?fw=pt) 没有涉及到，但原始论文中指出可以使用 Forward-DP Backward-A* 算法得到最优的 k 种subword 序列, 使用 Forward-Filtering and Backward-Sampling algorithm(FFBS) 可以按概率采样到 k 种 subword 序列


原始论文[Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/abs/1804.10959)中对 Unigram 算法的描述与上述基本一致，稍有不同的是在 $p(x_i)$ 的计算上，论文中描述用 EM 算法得到，而上述描述里直接使用频率得到。

对原始论文的理解以及一些实现细节可以参考这篇[博客](https://towardsdatascience.com/sentencepiece-tokenizer-demystified-d0a3aac19b15)

### SentencePiece

<div class="alert-red">
本小节的描述可能不准确，需进一步分辨
</div>

在 🤗 Transformers 中,  按[官方文档](https://huggingface.co/docs/transformers/tokenizer_summary)描述：SentencePiece 算法总是和 Unigram 配合使用, 因此可以认为在 🤗 Transformers 中, 这两者基本上可以划等号。（🤗 Transformers 中的 SentencePiece = 一些预处理 + Unigram）

在实现细节上，🤗 Transformers 中 fast tokenizer 依赖于 🤗 Tokenizers，而 🤗 Tokenizers 中对 sentencepiece 的处理方式是使用 protobuf 解析 sentencepiece 的词表存储格式, 然后再组合上 🤗 Tokenizers 自身实现的 Unigram, 详细内容可以参考[tokenizers/implementations/sentencepiece_unigram.py](https://github.com/huggingface/tokenizers/blob/main/bindings/python/py_src/tokenizers/implementations/sentencepiece_unigram.py)。即相当于 🤗 Tokenizers 重新实现了 sentencepiece。但需要注意的是： 🤗 Tokenizers 也实现了 SentencePieceBPETokenizer ，但并未在 🤗 Transformers 被使用到。

🤗 Transformers 中 slow tokenizer 则一般依赖于 sentencepiece 包


### T5 使用的 tokenizer

T5 使用 SentencePiece 作为 tokenizer，细节参考实现部分

## 源码解析: 🤗 Tokenizers

本节只介绍 🤗 Tokenizers 本身的使用，不涉及 🤗 Transformers 中 fast tokenizer 对 🤗 Tokenizers 的进一步封装

🤗 Tokenizers 的[官方文档-Getting Started](https://huggingface.co/docs/tokenizers/index)对使用的介绍已经足够充分，此处仅起一个浓缩的作用。

🤗 Tokenizers 代码库的核心类为 `tokenizers.Tokenizer`。

### 组成

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
# (component-3: model): 将词tokenize为token列表: List(str) -> List(Token)
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# (component-1: normalizer): 对原始句子进行预处理: str -> str
from tokenizers.normalizers import NFD, StripAccents
tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])

# (component-2: pre_tokenizer): 将句子拆分为词列表: str -> List(str)
from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()

# (component-4: post-processor): 对token列表进行后处理, 例如增加EOS: List(Token) -> List(Token)
from tokenizers.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",  # 这里的:1指的是将这部分的token_type_id标记为1
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

# (component-5: decoder): 将token列表转换为句子: List(str) -> str
from tokenizers import decoders
tokenizer.decoder = decoders.WordPiece()
tokenizer.decode(output.ids)

# trainer
from tokenizers.trainers import BpeTrainer
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
tokenizer.train(files, trainer)

# tokenizer 的保存格式为一个单一的 json 文件
tokenizer.save("data/tokenizer-wiki.json")
tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")  # 与 🤗 Transformers 中 fast tokenizer 的使用类似

# 这种用法可能不常用? BertWordPieceTokenizer 的基类是BaseTokenizer, 而BaseTokenizer与Tokenizer类无关
from tokenizers import BertWordPieceTokenizer
tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
```

以 gpt2 为例简要看一下各个组成部分怎么单独被调用

<div class="alert-red">
注意: 一般情况下, 不要单独使用各个组成部分
</div>

```python
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_pretrained("gpt2")

# (component-1: normalizer): 对原始句子进行预处理: str -> str
tokenizer.normalizer                              # None
# 不巧的是, gpt2并没有normalizer, 所以这里只好另外造一个例子
from tokenizers.normalizers import StripAccents, NFD, NFC, Sequence
normalizer = Sequence([NFD(), StripAccents()])    # StripAccents 需要与 NFD 配合使用
normalizer.normalize_str("é")                     # 输出: 'e'

text = "中国"

# (component-2: pre_tokenizer): 将句子拆分为词列表: str -> List(str)
tokenizer.pre_tokenizer                           # tokenizers.pre_tokenizers.ByteLevel
word_with_pos = tokenizer.pre_tokenizer.pre_tokenize_str("中国")
# word_with_pos: [('ä¸ŃåĽ½', (0, 2))], 切词的结果, 这个看起来乱码的东西实际上长度为6(在utf-8编码中汉字一般由3个字节构成)
print([ord(x) for x in word_with_pos[0][0]])      # [228, 184, 323, 229, 317, 189]
print(list(text.encode()))                        # [228, 184, 323, 229, 317, 189]

# (component-3: model): 将词tokenize为token列表: List(str) -> List(Token)
tokenizer.model                                   # tokenizers.models.ByteLevel
all_tokens = []
for word, (start, end) in word_with_pos:
    tokens = tokenizer.model.tokenize(word)       # tokens: List[tokenizers.Token]
    all_tokens.append(tokens)

# tokenizers.Token 主要方法为 as_tuple(), 主要属性是 value, id
print([token.as_tuple() for token in all_tokens[0]])
# 输出为: [(40792, 'ä¸Ń', (0, 6)), (32368, 'åĽ', (6, 10)), (121, '½', (10, 12))]
# 为什么是(0, 6), (6, 10), (10, 12)而不是(0, 3), (3, 5), (5, 6)？

# (component-4: post-processor): 对token列表进行后处理, 例如增加EOS: List(Token) -> List(Token)
tokenizer.post_processor                          # tokenizers.processors.ByteLevel

# 不巧的是, gpt2的post_processor没有追加任何token
# tokenizer = Tokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
encoding = tokenizer.encode(text, add_special_tokens=False)
print(encoding.tokens)                            # ['中', '国']
encoding = tokenizer.post_processor.process(encoding)
print(encoding.tokens)                            # ['[CLS]', '中', '国', '[SEP]']

# (component-5: decoder): 将token列表转换为句子: List(str) -> str
tokenizer.decoder                                 # tokenizers.decoders.ByteLevel
token_strs = [token.value for tokens in all_tokens for token in tokens]  # ['ä¸Ń', 'åĽ', '½']
tokenizer.decoder.decode(token_strs)              # "中国"
```

下面的内容本质上是API文档介绍的浓缩

### 实例化与序列化

```python
# 构建方法1
from tokenizers import Tokenizer
from tokenizers.models import BPE
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
# tokenizer.normalizer = ...
# tokenizer.pre_tokenizer = ...
# tokenizer.post_processor = ...
# tokenizer.decoder = ...

# 构建方法2: 在 🤗 hub 中保存的 tokenizer.json 文件
tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")

# 构建方法3: 类似于 🤗 Transformers 的使用
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

# 保存
save_path = "tokenizer.json"
tokenizer.save(save_path)

# 补充: 🤗 Transformers 中使用 tokenizers.Tokenizer 构建 fast tokenizer
# 本质上: (1) PreTrainedTokenizerFast 的行为完全由 tokenizer_object 决定
# (2) PreTrainedTokenizerFast.save_pretrained 实际上调用了 Tokenizer.save
from transformers import PreTrainedTokenizerFast
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=save_path)
```

### encode and decode

```python
tokenizer.token_to_id  # <unk> -> unk_id(int), sdhdhud -> None
tokenizer.id_to_token  # 1273773 -> None
# 这里的add_special_tokens无论True/False，用户自定义增加的token都会被encode
# add_special_tokens=True表示进行post-processing过程(即增加CLS等)
# is_pretokenized 用默认值即可
tokenizer.encode(sequece: str, pair: Optional[str]=None, is_pretokenized=False, add_special_tokens=True)  # -> tokenizers.Encoding
tokenizer.encode_batch(sequeces: List[str], pair: Optional[str]=None, is_pretokenized=False, add_special_tokens=True)  # -> tokenizers.Encoding

# skip_special_tokens的默认值为True, 若设定为True, 则解码时滤掉特殊token
# 特殊token指的：tokenizer本身特殊token例如CLS, BOS，以及通过add_special_token添加的token
tokenizer.decode(ids: List[int], skip_special_tokens=True)
tokenizer.decode_batch(sequences: List[List[int]], skip_special_tokens=True)
```

### padding and truncate

```python
# 仅仅是例子, 而非完整的API
tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", ...)
tokenizer.enable_truncation(max_lenghth=128, strategy="longest_first", direction="right")

tokenizer.no_padding()
tokenizer.no_truncation()
```

### add tokens, vocab

`add_special_tokens` 与 `add_tokens`。见后文对 🤗 Transformers 中 tokenizer 的相关方法【待补充】

```python
# 总是包含特殊token, 但可以控制是否加入add_token时增加的token
tokenizer.get_vocab(with_added_tokens=True)
tokenizer.get_vocab_size(with_added_tokens=True)
```

### train

```python
tokenizer.train(files: List[str], trainer: tokenizers.trainer.Trainer)
# next(iterator) 返回 str 或 List[str], 推荐List[str]
tokenizer.train_from_iterator(iterator, trainer: tokenizers.trainer.Trainer, length=None)
```

### 可视化

```python
from tokenizers.tools.visualizer import EncodingVisualizer, Tokenizer

tokenizer = Tokenizer.from_pretrained("t5-small/tokenizer.json")
viz = EncodingVisualizer(tokenizer) # Change here
text = "I am a boy, using sentencepiece tokenizer 中国"
viz(text=text)
```

## 源码解析: sentencepiece【TODO：源码解析待后续另起一篇博客进行介绍】

🤗 Transformers 中每个具体的 slow tokenizer 的实现里, 如果 tokenizer 的类型为 BPE 或者是 WordPiece, 那么一般是在相应的 `xxx_tokenizer.py` 中使用 python 实现 BPE 和 WordPiece。因此会发现一些重复的代码，例如[tokenization_bert.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py)与[tokenization_distilbert.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/tokenization_distilbert.py)，这符合 🤗 Transformers 代码库的[哲学](https://huggingface.co/docs/transformers/philosophy)。而 tokenizer 的类型为 SentencePiece 时，相应的 slow tokenizer 的实现会借助 [sentencepiece](https://pypi.org/project/sentencepiece/) 包。

sentencepiece包的使用方法请直接参考: [官方示例](https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb)


## 源码解析: 🤗 Transformers 中的 tokenizer

首先说明一下 🤗 Transformers 与 🤗 Tokenizers 之间的关系：🤗 Transformers 4.x 版本中每个模型都会尽量支持两种 tokenizer 的实现, slow版本的实现与fast版本的实现, 后者依赖于 🤗 Tokenzers 包, 而前者不依赖, 且为纯 python 实现，所以 slow 版本的 tokenizer 更方便阅读。

具体来说，🤗 Transformers 中 fast tokenizer 在实例初始化时有如下代码段：
```python
from .convert_slow_tokenizer import convert_slow_tokenizer
from tokenizers import Tokenizer as TokenizerFast

class PreTrainedTokenizerFast(PreTrainedTokenizerBase):
  vocab_files_names = VOCAB_FILES_NAMES
  slow_tokenizer_class: PreTrainedTokenizer = None
  can_save_slow_tokenizer: bool = True
  
  # 节选了一部分
  def __init__(self, *args, **kwargs):
    tokenizer_object = kwargs.pop("tokenizer_object", None)
    slow_tokenizer = kwargs.pop("__slow_tokenizer", None)
    fast_tokenizer_file = kwargs.pop("tokenizer_file", None)
    from_slow = kwargs.pop("from_slow", False)
    if from_slow and slow_tokenizer is None and self.slow_tokenizer_class is None:
        raise ValueError("...")
    if tokenizer_object is not None:
        fast_tokenizer = copy.deepcopy(tokenizer_object)
    elif fast_tokenizer_file is not None and not from_slow:
        # We have a serialization from tokenizers which let us directly build the backend
        fast_tokenizer = TokenizerFast.from_file(fast_tokenizer_file)
    elif slow_tokenizer is not None:
        # We need to convert a slow tokenizer to build the backend
        fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
    elif self.slow_tokenizer_class is not None:
        # We need to create and convert a slow tokenizer to build the backend
        slow_tokenizer = self.slow_tokenizer_class(*args, **kwargs)
        fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
    else:
        raise ValueError("...")
    self._tokenizer = fast_tokenizer
    # ...
```
fast tokenizer 的各类方法例如：`tokenize`、`convert_tokens_to_ids`、`get_vocab`、`decode` 最终都会直接转换为对 `self._tokenizer` 的相应方法的调用。从前面对于 🤗 Tokenizers 的介绍可以知道，`self._tokenizer` 封装了这些组成部分：`normalizer`、`pre_tokenizer`、`tokenizer`、`post_processor`、`decoder`。

🤗 Transformers 中的每个 slow tokenizer 需要逐一用python实现 `normalizer`、`pre_tokenizer`、`tokenizer`、`post_processor`、`decoder` 这些组成部分，其中 `tokenizer` 是 BPE 或是 WordPiece 时，则需手动实现 encode 的过程，如果是 SentencePiece 时，则一般借助 sentencepiece 包来实现主要逻辑。


🤗 Transformers 中, 每个模型都会对应于其特有的 tokenizer, 例如: t5 模型的 tokenizer 为 `T5Tokenizer` 和 `T5TokenizerFast`。继承关系如下：

![](../assets/figures/t5/tokenizer.png)

<div class="alert-red">
注意: slow 版本的 tokenizer 与 fast 版本的 tokenizer 的行为未必能完全一致
</div>

后续章节使用如下术语：

- base tokenizer: transformers.PretrainedTokenizerBase
- slow tokenizer: transformers.PretrainedTokenizer
- fast tokenizer: transformers.PretrainedTokenizerFast
- specific slow tokenizer: transformers.PretrainedTokenizer 的子类, 例如: T5Tokenizer
- specific fast tokenizer: transformers.PretrainedTokenizerFast 的子类, 例如: T5TokenizerFast
- specific tokenizer: specific slow tokenizer 和 specific fast tokenizer
- tokenizers.Tokenizer: 🤗 Tokenizer 中的 tokenizers.Tokenizer

### 使用【TODO: 需调整】

从一个疑惑引入：[issue](https://github.com/huggingface/transformers/issues/5087)

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
pretrained_name_or_path = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(pretrained_name_or_path)

print(tokenizer.special_tokens_map_extended)
# {"eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>", "additional_special_tokens": ['<extra_id_0>', ..., '<extra_id_99>']}
print(tokenizer.eos_token, tokenizer.unk_token, tokenizer.pad_token)
# eos_token: 1, unk_token: 2, pad_token: 0

text = "abc __"
tokens = tokenizer.tokenize(text)  # ["__ab", "c", "__", "_", "_"]
ids = tokenizer.convert_tokens_to_ids(tokens)  # [703, 75, 3, 834, 834]

ids = tokenizer.encode(text)  # [703, 75, 3, 834, 834, 1]
```

由此可见，在 `T5Tokenizer` 的实现里，没有 `bos_token` 这个属性，并且每个 word 起始的 subword 会加上 `__` 的前缀。注意词表中既有以 `__` 开头的 token，例如 `__ab`，而 `__` 本身也在词表中。这种处理方式是因为 `T5Tokenizer` 使用了 SentencePiece Tokenizer。

tokenizer 的常用方法如下参考[笔记](https://buxianchen.gitbook.io/notes/note/dl/huggingface#pretrainedtokenizerbase)【后续考虑怎么合并/删减】


### add tokens

另外，`PretrainedTokenizerBase` 的 `add_tokens` 与 `add_special_tokens` 的这两个方法也让人困惑。因此有必要理清楚。

首先，这里引用 [SentencePiece 教程](https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb) 中的几个术语对 token 进行分类（在🤗 Transformers的文档中，笔者没有见到类似的术语）：

- normal symbols: 普通的 token, sentencepiece tokenizer 可能会将这种 token 切分开
- user defined symbols: 用户增加的特殊 token, 可以出现在原始文本中, sentencepiece tokenizer 保证不会对这种 token 进行切分
- control symbols: 对tokenizer的结果进行后处理时使用的 token, 例如：sentencepiece tokenizer 将句子 tokenize 后, 后处理加上 `"[CLS]"` 和 `"[SEP]"`, 如果在输入的句子中含有 `"[CLS]"`, sentencepiece tokenizer 有可能会将这种 token 切分开

从上面的例子推广开来, 对 tokenizer 增加 token 应该要包含这几种情形:

- 普通token,出现在原始文本, 不保证它不被切分开: 例如: 假设词表中已经有了 `"中国"` 和 `"人"` 这两个token，现在增加 `"中国人"` 到词表里, 目的是希望 tokenizer 有可能会将 `"中国人"` 当作一个整体, 当然也不排除 tokenizer 仍然会被切分为 `"中国"` 和 `"人"`。然而在 BPE、WordPiece、Unigram 这三类算法中，为了增加这种类型的token，
  - BPE 需要增加的是 merge 规则, 即 `("中国", "人")`, 甚至于需要调整这个 merge 的规则到合适的位置(优先级)
  - WordPiece 只需要将 `"中国人"` 加入到词表中即可
  - Unigram 需要将 `"中国人"` 以及相应的概率值加入至词表里, 甚至于需要调整已有词的概率值
  因此 🤗 Transformers 中不支持这种添加方式(slow tokenizer不支持, 不确定 fast tokenizer 的情况)
- 出现在原始文本中的token, 保证它不会被切分开（🤗 Transformers 支持）
- 后处理token, 主要用途用于后处理时追加。并且即使它出现在原始文本中, 也不会切分开（🤗 Transformers 的EOS等都有此性质）
- 后处理token, 主要用途用于后处理时追加。但如果它出现在原始文本中, 有可能会被切分开（🤗 Transformers 不支持）


有了上述认知，下面具体分析源代码

`PrtrainedTokenizerBase` 与 `add_tokens` 和 `add_special_tokens` 中有关的代码片段如下
```python
class SpecialTokensMixin:
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]
    def add_special_tokens(self, special_tokens_dict: Dict[str, Union[str, AddedToken]], replace_additional_special_tokens=True) -> int:
        if not special_tokens_dict:
            return 0
        added_tokens = 0
        for key, value in special_tokens_dict.items():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES, f"Key {key} is not a special token"
            if key == "additional_special_tokens":
                assert isinstance(value, (list, tuple)) and all(isinstance(t, (str, AddedToken)) for t in value)
                if replace_additional_special_tokens:
                    setattr(self, key, value)
                else:
                    # This is a copy of `self._additional_special_tokens`
                    additional_special_tokens = getattr(self, key)
                    additional_special_tokens_set = set(additional_special_tokens)
                    to_add = []
                    for token in value:
                        if str(token) not in additional_special_tokens_set and str(token) not in to_add:
                            to_add.append(token)
                    # update the property
                    additional_special_tokens.extend(to_add)
                    self.additional_special_tokens = additional_special_tokens
                added_tokens += self.add_tokens(value, special_tokens=True)
            else:
                assert isinstance(value, (str, AddedToken))
                setattr(self, key, value)
                added_tokens += self.add_tokens([value], special_tokens=True)
        return added_tokens
    def add_tokens(self, new_tokens: Union[str, AddedToken, List[Union[str, AddedToken]]], special_tokens: bool = False) -> int:
        if not new_tokens:
            return 0
        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]
        return self._add_tokens(new_tokens, special_tokens=special_tokens)
    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        raise NotImplementedError
```

由此可见:
- `add_special_tokens` 实际上只是用来操作 `[bos_token, eos_token, unk_token, sep_token, pad_token, cls_token, mask_token]` 以及 `additional_special_tokens` 这几个属性的, 从【其他】分析可以知道 `additional_special_tokens` 跟前面 6 种 token 并没有本质区别。而 `add_special_tokens` 的行为是给这 7 个实例变量赋新值, 然后再调用 `added_tokens`
- `add_tokens` 的行为完全由 `_add_tokens` 决定, 由子类 `PrtrainedTokenizer` 和 `PrtrainedTokenizerFast` 实现

**slow tokenizer**

对于 slow tokenizer, `_add_tokens` 的实现如下

```python
def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
    new_tokens = [str(tok) for tok in new_tokens]
    tokens_to_add = []
    for token in new_tokens:
        if not isinstance(token, str):
            raise TypeError(f"Token {token} is not a string but a {type(token)}.")
        if not special_tokens and hasattr(self, "do_lower_case") and self.do_lower_case:
            token = token.lower()
        if (
            token != self.unk_token
            and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token)
            and token not in tokens_to_add
        ):
            tokens_to_add.append(token)
    added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(tokens_to_add))
    added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
    self.added_tokens_encoder.update(added_tok_encoder)
    self.added_tokens_decoder.update(added_tok_decoder)

    # Make sure we don't split on any special tokens (even they were already in the vocab before e.g. for Albert)
    if special_tokens:
        if len(new_tokens) == 1:
            _insert_one_token_to_ordered_list(self.unique_no_split_tokens, new_tokens[0])
        else:
            self.unique_no_split_tokens = sorted(set(self.unique_no_split_tokens).union(set(new_tokens)))
    else:
        # Or on the newly added tokens
        if len(tokens_to_add) == 1:
            _insert_one_token_to_ordered_list(self.unique_no_split_tokens, tokens_to_add[0])
        else:
            self.unique_no_split_tokens = sorted(set(self.unique_no_split_tokens).union(set(tokens_to_add)))
    self._create_trie(self.unique_no_split_tokens)
    return len(tokens_to_add)
```

因此 slow tokenizer `add_tokens` 方法的流程如下:

- 通过 `convert_tokens_to_ids(token)==convert_tokens_to_ids(self.unk_token)` 判断是否为新增词, 如果是, 则在 `self.added_tokens_encoder` 以及 `self.added_tokens_decoder` 中记录 token to idx 和 idx to token 的映射关系, 注意这两个实例变量是 slow tokenizer 独有的, fast tokenizer 无此实例变量【还需要在其他地方介绍这两个实例变量】
- 入参 `special_tokens=True`, 那么就不进行前一步筛选, 直接将入参 `new_tokens` 作为不可分割的 token 加入到词表中【还需要在其他地方介绍self.unique_no_split_tokens】。入参 `special_tokens=False`, 那么就需要经过前一步筛选再作为不可分割的 token 加入到词表中
- 调用 `self._create_trie`, 便于tokenize的时候先保证不可分割的词不被切开

因此, `add_tokens(tokens, special_tokens=False)`的行为是:

- 如果被加入的token不在词表内, 则为其增加对应的token_id, 并且将被加入的token不可分割
- 如果被加入的token在词表内, 则什么都不做

`add_tokens(tokens, special_tokens=True)` 的行为是:

- 如果被加入的token不在词表内, 则为其增加对应的token_id, 并且将被加入的token不可分割
- 如果被加入的token在词表内, 则将其作为不可分割的token


**fast tokenizer**

对于 fast tokenizer, `_add_tokens` 的实现如下
```python
def _add_tokens(self, new_tokens: List[Union[str, AddedToken]], special_tokens=False) -> int:
    if special_tokens:
        return self._tokenizer.add_special_tokens(new_tokens)
    return self._tokenizer.add_tokens(new_tokens)
```
所以本质上回到了 🤗 Tokenizers 中 `tokenizers.Tokenizer` 的两个方法: `add_special_tokens`, `add_tokens`

总结如下：

- 🤗 Tokenizers 中 `tokenizers.Tokenizer` 相关的方法总结如下:
  - `Tokenizer.add_tokens(tokens: List[Union[AddedToken, str]])`: 如果加入的token不在原本的词表内, 则为其增加token_id, 并保证新增的token不会被切分开；如果加入的token在原本的词表内, 则什么都不做(即它在tokenize阶段仍有可能被切分开)
  - `Tokenizer.add_special_tokens(tokens: List[Union[AddedToken, str]])`: 如果加入的token不在原本的词表内, 则为其增加token_id, 并保证新增的token不会被切分开；如果加入的token在原本的词表内, 则不为其增加token_id, 但保证它不会被切分开
- 🤗 Transformers 中的 slow/fast tokenizer 的相关方法总结如下:
  - `PretrainedTokenizerBase.add_tokens(tokens, special_tokens=False)`的行为是:
    - 如果被加入的token不在词表内, 则为其增加对应的token_id, 并且将被加入的token不可分割
    - 如果被加入的token在词表内, 则什么都不做
  - `PretrainedTokenizerBase.add_tokens(tokens, special_tokens=True)`的行为是:
    - 如果被加入的token不在词表内, 则为其增加对应的token_id, 并且将被加入的token不可分割
    - 如果被加入的token在词表内, 则将其作为不可分割的token
  - `PretrainedTokenizerBase.add_special_tokens(special_tokens_dict)`: 只能增加 8 种特殊 token, 首先设置相关的属性, 例如: `self.cls_token`, 然后调用 `PretrainedTokenizerBase.add_tokens(tokens, special_tokens=True)`, 这么一来保证加入的token总是不会被切分开


### vocabulary

对于一个特定的 tokenizer, 我们现在已经知道，它词表里的token一般分为几类

- (1) 普通 token
- (2) SpecialTokensMixin.SPECIAL_TOKENS_ATTRIBUTES
- (3) 通过`add_tokens`添加的 token

那么在获取词表时或计算词表长度时，就会出现几种不同的计算方式，有必要理清一下：

**base tokenizer**

在基类 `PretrainedTokenizerBase` 的父类 `SpecialTokensMixin` 中

```python
class SpecialTokensMixin:
    @property
    def special_tokens_map(self) -> Dict[str, Union[str, List[str]]]:
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = (
                    type(attr_value)(str(attr_value_sub) for attr_value_sub in attr_value)
                    if isinstance(attr_value, (list, tuple))
                    else str(attr_value)
                )
        return set_attr

    @property
    def special_tokens_map_extended(self) -> Dict[str, Union[str, AddedToken, List[Union[str, AddedToken]]]]:
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self) -> List[str]:
        all_toks = [str(s) for s in self.all_special_tokens_extended]
        return all_toks

    @property
    def all_special_tokens_extended(self) -> List[Union[str, AddedToken]]:
        all_toks = []
        set_attr = self.special_tokens_map_extended
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
        all_toks = list(OrderedDict.fromkeys(all_toks))
        return all_toks

    @property
    def all_special_ids(self) -> List[int]:
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids
```

这些属性都仅是第(2)类token的集合

- `special_tokens_map_extended`: `bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens` 八者的总和, value可以是🤗 Tokenizers 中 `tokenizers.AddedToken` 类型
- `special_tokens_map`: 与上者相同, 唯一的区别是value一定是str
- `all_special_tokens_extended`: `special_tokens_map_extended` 的value列表
- `all_special_tokens`: 与上者相同, 唯一的区别是value一定是str
- `all_special_ids`: 使用 `convert_tokens_to_ids` 转换第(2)类token的token_id列表

**slow tokenizer**

- `added_tokens_encoder: Dict[str, int]` 属性: 只在调用 `add_tokens` 时被更新, 在将 token 转化为 id 时被使用, 即只包含第(3)部分的token
- `added_tokens_decoder: Dict[int, str]` 属性: 只在调用 `add_tokens` 时被更新, 在将 id 转化为 token 时被使用, 即只包含第(3)部分的token
- `__len__()`: `self.vocab_size+len(self.added_tokens_encoder)`, 即包含(1)(2)(3)全部的token
- `vocab_size`: 由具体的 tokenizer 实现, 包含的是第(1)和第(2)部分的token数量
- `get_vocab`: 由具体的 tokenizer 实现, 返回 `Dict[str, int]`, 即包含(1)(2)(3)全部的token
- `get_added_vocab()`: 返回 `self.added_tokens_encoder`, 即第(3)部分token
- `unique_no_split_tokens`: 一般情况下（使用`from_pretrained`方法初始化slow tokenizer时）包含第(2)和第(3)部分的token
- `tokens_trie`: 由`unique_no_split_tokens`构成的`Trie`数据结构
- `save_vocabulary`: 由具体的 tokenizer 实现, 实现的逻辑是保存第(1)和第(2)部分token。`save_pretrained` 方法会额外处理以下两件事：将第(3)部分token保存在`added_token.json` 文件内, 第(2)部分token还会同时再度被保存在`special_tokens_map.json`文件中

**fast tokenizer**

fast tokenizer 与 slow tokenizer 在相同命名的属性/方法上的含义是相同的 

- `vocab_size`: 包含的是第(1)和第(2)部分的token数量
  ```python
  @property
  def vocab_size(self) -> int:
      # `int`: Size of the base vocabulary (without the added tokens).
      return self._tokenizer.get_vocab_size(with_added_tokens=False)
  ```
- `get_vocab`方法与`vocab`属性一致: 包含(1)(2)(3)三部分token
  ```python
  def get_vocab(self) -> Dict[str, int]:
      return self._tokenizer.get_vocab(with_added_tokens=True)
  ```
- `get_added_vocab`: `Dict[str, int]`, 第(3)部分的token
  ```python
  def get_added_vocab(self) -> Dict[str, int]:
      base_vocab = self._tokenizer.get_vocab(with_added_tokens=False)
      full_vocab = self._tokenizer.get_vocab(with_added_tokens=True)
      added_vocab = dict((tok, index) for tok, index in full_vocab.items() if tok not in base_vocab)
      return added_vocab
  ```
- `__len__()`: 包含(1)(2)(3)三部分token
  ```python
  def __len__(self) -> int:
      return self._tokenizer.get_vocab_size(with_added_tokens=True)
  ```


### 实例化与序列化【TODO：代码已抄完, 但联系没搞清楚】

本节主要涉及如下方法 `__init__`、`from_pretrained`、`save_pretrained`，以及一些 3 个基类定义的一些属性

#### 类属性与`__init__`

```python
class SpecialTokensMixin:
    def __init__(self, verbose=True, **kwargs):
        self._bos_token, self._eos_token, self._unk_token self._sep_token = ...  # kwargs
        self._pad_token, self._cls_token, self._mask_token = ...  # kwargs
        self._pad_token_type_id = 0
        self._additional_special_tokens = ...  # kwargs
        self.verbose = verbose

class PretrainedTokenizerBase(SpecialTokensMixin, PushToHubMixin):
    vocab_files_names: Dict[str, str] = {}                           # specific slow tokenizer 需指定, fast tokenizer 为固定值 {"tokenizer_file": "tokenizer.json"}, 用于 __init__, from_pretrained, save_pretrained
    pretrained_vocab_files_map: Dict[str, Dict[str, str]] = {}       # specific tokenizer 需指定, "官方"模型的vocab_files_names
    pretrained_init_configuration: Dict[str, Dict[str, Any]] = {}    # ??? specific tokenizer 需指定, "官方"模型的init_kwargs
    max_model_input_sizes: Dict[str, Optional[int]] = {}             # specific tokenizer 需指定, "官方"模型的max_model_input_sizes
    _auto_class: Optional[str] = None                                # ???
    model_input_names: List[str] = ["input_ids", "token_type_ids", "attention_mask"]  # ??? specific tokenizer 需指定
    padding_side: str = "right"
    truncation_side: str = "right"
    slow_tokenizer_class = None                                      # specific fast tokenizer 需设定
    def __init__(self, **kwargs):
        # inputs and kwargs for saving and re-loading (see ``from_pretrained`` and ``save_pretrained``)
        self.init_inputs = ()
        self.init_kwargs = copy.deepcopy(kwargs)
        self.name_or_path = ...                   # kwargs
        self._processor_class = ...               # kwargs, ???
        self.model_max_length = ...               # kwargs, default: int(1e30)
        self.padding_side = ...                   # kwargs, default: cls.padding_side
        self.truncation_side = ...                # kwargs, default: cls.padding_side
        self.model_input_names = ...              # kwargs, default: cls.model_input_names
        self.clean_up_tokenization_spaces = ...   # default True
        self.deprecation_warnings = ({})
        self._in_target_context_manager = False   # ???


class PretrainedTokenizer(PreTrainedTokenizerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Added tokens - We store this for both slow and fast tokenizers
        # until the serialization of Fast tokenizers is updated
        self.added_tokens_encoder: Dict[str, int] = {}
        self.added_tokens_decoder: Dict[int, str] = {}
        self.unique_no_split_tokens: List[str] = []
        self.tokens_trie = Trie()

        self._decode_use_source_tokenizer = False  # ???

class PreTrainedTokenizerFast(PreTrainedTokenizerBase):
    vocab_files_names = {"tokenizer_file": "tokenizer.json"} # 所有的 specific fast tokenizer 都是这个
    slow_tokenizer_class: PreTrainedTokenizer = None         # specific fast tokenizer 需指定, 例如: T5TokenizerFast 中设为 T5Tokenizer
    can_save_slow_tokenizer: bool = True                     # 控制 save_pretrained 的行为
    def __init__(self, *args, **kwargs):
        self._tokenizer: tokenizers.Tokenizer                # construct directly / convert slow tokenizer to fast tokenizer
        self._decode_use_source_tokenizer = False
        # We call this after having initialized the backend tokenizer because we update it.
        super().__init__(**kwargs)
```


#### from_pretrained

**base tokenizer**

```python
@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, ...):
    # 一堆下载与缓存必要的文件的操作
    # resolved_vocab_files 一般会包含如下key, 若对应的value的文件存在
    # {
    #    **cls.vocab_files_names,
    #    "added_tokens_file": "added_tokens.json",
    #    "special_tokens_map_file": "special_tokens_map.json",
    #    "tokenizer_config_file": "tokenizer_config.json"
    #  }
    return cls._from_pretrained(resolved_vocab_files, pretrained_model_name_or_path, ...)
@classmethod
def _from_pretrained(cls, pretrained_model_name_or_path, ...):
    ... # 一堆操作
    tokenizer = cls(*args, **kwargs)
    tokenizer.add_tokens(...)
```

#### save_pretrained

**base tokenizer**
```python
def save_pretrained(self, save_directory, legacy_format: Optional[bool] = None, filename_prefix=None, push_to_hub=False, **kwargs):
    ...  # 保存信息至 tokenizer_config.json 中去
    ...  # 将self.special_tokens_map_extended保存至special_tokens_map.json 中去
    file_names = (tokenizer_config_file, special_tokens_map_file)
    save_files = self._save_pretrained(save_directory, file_names, legacy_format, filename_prefix)
    return save_files

def _save_pretrained(save_directory, file_names, legacy_format, filename_prefix):
    ...  # 将self.get_added_vocab()保存至added_tokens.json中去
    vocab_files = self.save_vocabulary(save_directory, filename_prefix)  # 子类实现
    return file_names + vocab_files + (added_tokens_file,)
```

**fast tokenizer**
```python
# 覆盖父类方法
def _save_pretrained(save_directory, file_names, legacy_format, filename_prefix):
    # legacy_format 为 None, 则尽量分别保存 slow tokenizer 的文件以及 fast tokenizer 的文件
    # legacy_format 为 False, 则只保存 fast tokenizer 的文件
    # legacy_format 为 True, 则只保存 slow tokenizer 的文件
    if save_slow:
        ...  # 将self.get_added_vocab()保存至added_tokens.json中去
        vocab_files = self.save_vocabulary(save_directory, filename_prefix)  # 子类实现
        file_names = file_names + vocab_files + (added_tokens_file,)
    if save_fast:
        self.backend_tokenizer.save(tokenizer_file)  # fast tokenizer 只需要保存tokenizer.json
        file_names = file_names + (tokenizer_file,)
    return filenames
```

因此 specific slow tokenizer 必须实现 `save_vocabulary` 方法, 用来保存第(1)(2)类token???【待搞清楚】, 而 specific fast tokenizer 也尽量实现 `save_vocabulary` 方法, 以支持对应的 slow tokenizer 的保存。具体可参考 T5Tokenizer 与 T5TokenizerFast 的实现。


### `BatchEncoding`

在介绍 `__call__` 方法的逻辑之前, 先对它的返回值做简单介绍。

提示: 下面介绍的 fast tokenizer 专属的属性实际来源于 backend_tokenizer 处理后的返回类型 `Tokenizer.Encoding`，`PretrainedTokenizerFast` 中的 `_convert_encoding` 方法用于将 `Tokenizer.Encoding` 转换为字典形式, 最终再转换为 `transformers.BatchEncoding` 作为 `__call__` 方法的返回值


首先看一个使用示例：

```python
from transformers import AutoTokenizer  # AutoTokenizer总是尝试加载fast版本的tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
encodings = tokenizer(["This is a sentence"])
```

此处的 `encodings` 的类型为 `BatchEncoding` 类型, 它继承自 `UserDict`，即继承自字典。除了字典的方法外，它还具备以下方法

```python
inputs = encoding.convert_to_tensors("pt")  # inplace操作, 将内部的value例如input_ids等转换为tensor
inputs = encoding.to("cuda:0")  # inplace操作, 改变设备
# 注: 以下方法仅使用于 fast 版本的 tokenizer 的情形
# 注: 以下方法对于特殊token例如[CLS], 返回结果会处理成None
encoding.tokens()                     # List[str], 所有的token字符串, 假设长度为N
# tokenize一个batch的数据时, 注意需要调整入参, encoding.tokens(i), i为第几个样本, 下同
encoding.word_ids()                   # List[int], 每个token所在的word_idx, 注意这里的word的概念通常取决于pre-tokenizer的定义
encoding.sequence_ids()               # List[int], 每个token所在的sequence_idx, 这里的sequence的概念取决于pre-tokenizer的定义
encoding.token_to_word(token_idx)     # 第token_idx个token所在的word_idx
encoding.token_to_sequence(token_idx) # 第token_idx个token所在的sequence_idx
start, end = encoding.word_to_chars(word_idx)      # 第word_idx个word对应的原始string的起始/结束位置
start, end = encoding.word_to_tokens(word_idx)     # 第word_idx个word对应的起始与结束的token_idx
start, end = encoding.token_to_chars(token_idx)    # 第token_idx个token对应的原始string的起始/结束位置
word_idx = encoding.char_to_word(i)                # 原始string中第i个字符对应的word_idx
token_idx = encoding.char_to_token(i)              # 原始string中第i个字符对应的token_idx
```

简单来说, fast 版本的 tokenizer 的 encode 过程保存了原始字符串中每个字符与token, word, sequence的对应关系, 而 slow 版本不具备


### encode: `PretrainedTokenizerBase.__call__`【TODO】

官方建议不要直接调用 `batch_encode_plus`，`encode_plus` 方法，而是通过 `__call__` 方法来调用。这一过程实际起作用的“组件”函数为：`encode`、`convert_tokens_to_ids`、`prepare_for_model`。

不做说明的情况下，默认指的是`PretrainedTokenizerBase`的方法，首先对 `__call__` 方法的重要的输入参数做介绍

```python
def __call__(
    self,
    text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
    text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
    text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
    text_pair_target: Optional[
        Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
    ] = None,
    add_special_tokens: bool = True,  # 是否需要在tokenize之后添加一些特殊token(例如起始结束token), 需具体的tokenizer实现，具体见后面说明
    padding: Union[bool, str, PaddingStrategy] = False,  # padding, truncation, max_length见后面说明
    truncation: Union[bool, str, TruncationStrategy] = None,
    max_length: Optional[int] = None,
    is_split_into_words: bool = False,  # 见后面说明
    return_overflowing_tokens: bool = False,  # 返回被截断的部分
    return_offsets_mapping: bool = False,  # 返回每个token_id对应于原始文本的起始位置(slow tokenizer不支持此特性)
    ...
) -> BatchEncoding:
    # 以下为大体逻辑, 有删改
    if text is not None:
        if not self._in_target_context_manager:
            self._switch_to_input_mode()
        encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
    if text_target is not None:
        self._switch_to_target_mode()
        target_encodings = self._call_one(text=text_target, text_pair=text_pair_target, **all_kwargs)
    self._switch_to_input_mode()
    if text_target is None:
        return encodings
    elif text is None:
        return target_encodings
    else:  # source和target都给的时候, 只把target encodings结果中的input_ids作为labels添加到source的encoding结果里
        encodings["labels"] = target_encodings["input_ids"]
    return encodings
# 注意 text_pair 指的是第2个句子, 而非句子对

# text/text_pair/text_target/text_pair_target的数据类型为以下4种情况:
TextInput = str  # 即整句话, 转为token序列
List[TextInput] = List[str] # batch版本, 多句话分别转为token序列
PreTokenizedInput = List[str]  # 已经预先切好"词"的序列, 这个时候会对每个小段进行token化, 最后拼接在一起
List[PreTokenizedInput] = List[List[str]]  # 多个已经且为小段的句子
# 一个 PreTokenizedInput 的使用例子是: ["紫禁城", "是xxx,坐落于", "北京"]
# 得到的序列会是 List[int] = tokenize("紫禁城") + tokenize(是xxx,坐落于) + tokenize("北京")
# 保证命名实体本身不会被切分开来
```

对输入参数做简要解释如下

**`text`, `text_pair`, `text_target`, `text_pair_target`**

需要被序列化为整数的“东西”，某些 tokenizer 对 source(输入) 和 target(输出) 的序列化方式可能有所不同, 所以留了

**`add_special_tokens`**

表示是否需要在tokenize之后添加一些特殊token(例如起始结束token), 默认值为True，这个参数在 `prepare_for_model` 中用到，不同的 tokenizer 需要通过重载如下几个方法进行实现：
  ```python
  def prepare_for_model(self, ...):
      # 前序处理省略, 主要包括truncate

      # Add special tokens
      if add_special_tokens:
          sequence = self.build_inputs_with_special_tokens(ids, pair_ids)  # 追加特殊token
          token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)  # 通常第1句话的位置为0，第2句的位置为1
      else:
          sequence = ids + pair_ids if pair else ids
          token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

      # Build output dictionary
      encoded_inputs["input_ids"] = sequence
      if return_token_type_ids:
          encoded_inputs["token_type_ids"] = token_type_ids
      if return_special_tokens_mask:
          if add_special_tokens:
              encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
          else:
              encoded_inputs["special_tokens_mask"] = [0] * len(sequence)
      # 后续处理主要是pad
  
  # 上面几个方法的默认实现如下:
  def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1):
      if token_ids_1 is None:
          return token_ids_0
      return token_ids_0 + token_ids_1
  def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1):
      if token_ids_1 is None:
          return len(token_ids_0) * [0]
      return [0] * len(token_ids_0) + [1] * len(token_ids_1)
  def get_special_tokens_mask(self, token_ids_0, token_ids_1, already_has_special_tokens=False):
      # 1 代表 special token, 0 代表普通的 token
      all_special_ids = self.all_special_ids  # cache the property
      special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]
      return special_tokens_mask

  ```
  例子: `BertTokenizer`，`T5Tokenizer` 均对 `build_inputs_with_special_tokens`、`create_token_type_ids_from_sequences`、`get_special_tokens_mask`做重载。并且也是这两个 tokenizer 除了必须实现的5个方法 `save_vocabulary`、`get_vocab`、`_tokenize`、`_convert_token_to_id`、`convert_id_to_token` 以外的全部重载方法。（对于decode过程，`T5Tokenizer`还重载了`convert_tokens_to_string`）

**`padding`、`truncate`、`max_length`、`is_split_into_words`**

is_split_into_words 与调用 `_batch_encode_plus` 还是调用 `_encode_plus` 是相关的【待补充】

备注：

- `truncation_side` 取值为 `"left"` 表示截断时去掉左边的字符，取值为 `"right"` 表示截断时去掉右边的字符

**`__call__`方法的调用流程**

`__call__` 方法的具体流程如下：首先将需要转换为 token 序列的输入分为两组 `text, text_pair` 和 `text_target, text_pair_target`，分别调用 `_call_one` 方法，然后将两部分进行合并。而 `_call_one` 方法根据 `text` 或 `text_target` 的变量类型以及 `is_split_into_words` 参数的取值确定进一步调用两者之一: `batch_encode_plus` 或是 `encode_plus`，此时注意这两个函数的函数签名如下:

```python
EncodedInput=List[int]
EncodedInputPair=Tuple[List[int], List[int]]

def batch_encode_plus(
    self,
    batch_text_or_text_pairs: Union[
        List[TextInput],  # List[str]
        List[TextInputPair],  # List[Tuple[str, str]]
        List[PreTokenizedInput],  # List[List[str]]
        List[PreTokenizedInputPair],  # List[Tuple[List[str], List[str]]]
        List[EncodedInput],  # 如果只看 __call__ 方法docstring, 在调用__call__方法时, 不可能以这种变量类型触发batch_encode_plus方法
        List[EncodedInputPair],  # 如果只看 __call__ 方法docstring, 在调用__call__方法时, 不可能以这种变量类型触发batch_encode_plus方法
    ],
    ...
) -> BatchEncoding:
    ...

def encode_plus(
    self,
    text: Union[TextInput, PreTokenizedInput, EncodedInput],  # 同理EncodeInput这种类型按理也不会触发
    text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
    ...
) -> BatchEncoding:
    ...
```

而这两个方法首先根据输入参数 `padding`、`truncate`、`max_length` 处理好（转换成相应的枚举类型，用户如果不传 max_length，这一步也会将其转换为一个确确实实的整数），然后继续调用 `_batch_encode_plus` 或 `_encode_plus` (这两个方法在子类 `PretrainedTokenizer` 和 `PretrainedTokenizerFast` 中分别实现) ，以 slow 版本的为例，它们实际做的事情可以参考如下简化版本的源代码实现：

```python
# 简化版本(只考虑text为str类型, 不考虑List[str]类型)
def _encode_plus(self, text, text_pair):
    # tokenize方法内部依次调用: 一次 prepare_for_tokenization 和多次 _tokenize 完成
    # _tokenize方法必须在具体的tokenizer中实现
    # tokenize的大体逻辑为: 把不可拆分的token抽出来, 其余的调用_tokenize来完成
    # 例如: "我在<extra_001>马路边" => ["我在", "<extra_001>", "马路边"]
    # => [_tokenize("我在"), 32001, _tokenize("马路边")] = [34, 567, 32001, 76, 98]
    first_tokens = self.tokenize(text, **kwargs)
    first_ids = self.convert_tokens_to_ids(first_tokens)
    second_tokens = self.tokenize(text_pair, **kwargs)
    second_ids = self.convert_tokens_to_ids(second_tokens)
    # prepare_for_model 需要做后处理: 首尾加特殊token, 获取
    return self.prepare_for_model(first_ids, second_ids, **kwargs)  # 这个方法定义在父类方法中, 见前面关于__call__的入参解释部分

# 简化版本(只考虑batch_text_or_text_pairs为List[tuple[str, str]]的情况)
def _batch_encode_plus(self, batch_text_or_text_pairs):
    input_ids = []
    for text_or_text_pair in batch_text_or_text_pairs:
        text, text_pair = text_or_text_pair
        first_tokens: List[str] = self.tokenize(text, **kwargs)
        first_ids: List[int] = self.convert_tokens_to_ids(first_tokens)
        second_tokens = self.tokenize(text_pair, **kwargs)
        second_ids = self.convert_tokens_to_ids(second_tokens)
        input_ids.append((first_ids, second_ids))
    return self._batch_prepare_for_model(input_ids, **kwargs)

def _batch_prepare_for_model(input_ids):
    batch_out = defaultdict(list)
    for first_ids, second_ids in input_ids:
        outputs: BatchEncoding = self.prepare_for_model(first_ids, second_ids)
        for key, value in outputs.items():
            batch_out[key].append(value)
    return BatchEncoding(self.pad(batch_out))
```

上面的源码中, 涉及到 `PretrainedTokenizer` 的 `tokenize`、`convert_tokens_to_ids`、`prepare_for_model` 方法，此处再做一些展开说明：
```python

```


### decode【TODO】


### 训练一个 Tokenizer

在 🤗 Transformers 库中, fast 版本的 tokenizer 实际上利用了 🤗 Tokenizers 的一些内容, 因此 `PretrainedTokenizerFast` 是可以训练的，而 slow 版本的 tokenizer 不支持训练。使用方法如下：

```python
from transformers import AutoTokenizer
old_tokenizer = AutoTokenizer.from_pretrained("t5-small")
training_corpus: List[str] = ["sentence one", "sentence one"]
training_corpus = ([training_corpus[i*32: (i+1)*32]] for i in range(100))  # 迭代器即可
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
tokenizer.save_pretrained("save-dir")
```

注意：这种做法适用于与现有的一个 tokenizer 一致的设定, 例如：BOS token等, 绝大多数情况下, 已经足够使用。如果确实需要做比较大的调整，则需要借助 🤗 Tokenizers 包（见前文介绍）。

### 自定义 specific slow tokenizer【TODO】

本节以 `T5Tokenizer` 为例, 介绍如何写一个 slow tokenizer

### 自定义 specific fast tokenizer【TODO】

本节以 `T5TokenizerFast` 为例, 介绍如何写一个 fast tokenizer

### Converter【TODO】

### 杂项【TODO: 需调整】

fast 版本的 tokenizer 可以通过如下方式查看其背后的 tokenizer 类型：

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("chinese-roberta-wwm-ext", use_fast=False)
json.loads(tokenizer._tokenizer.to_str())["model"]["type"]
```

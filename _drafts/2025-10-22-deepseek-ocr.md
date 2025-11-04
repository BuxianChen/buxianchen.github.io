---
layout: post
title: "(P0) deepseek-ocr 浅析"
date: 2025-10-22 09:05:04 +0800
labels: [llm,deepseekvllm]
---

## 模型结构


主入口

```python
# modeling_deepseekocr.py:DeepseekOCRForCausalLM.infer

es = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)


# 基本参数
{
    "vocab_size": 129280,
    # lm_head 的输出维度, bias=False
    "hidden_size": 1280,
}
```


### SAM

#### 目录结构

(1) transformers 的 `transformers/models/sam` 目录文件如下:

```
__init__.py
convert_sam_to_hf.py
image_processing_sam_fast.py  # ImageProcessing
image_processing_sam.py       # ImageProcessing
processing_sam.py             # Processing
configuration_sam.py
modeling_sam.py
```

(2) huggingface hub

```
README.md
model.safetensors
pytorch_model.bin
config.json
preprocessor_config.json  # 是 image_processor 的默认配置文件名
tf_model.h5
```

```python
# transformers.utils.__init__.py
WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"  # audio, transformers/feature_extraction_utils.py: FeatureExtractionMixin
IMAGE_PROCESSOR_NAME = "preprocessor_config.json"    # image
VIDEO_PROCESSOR_NAME = "video_preprocessor_config.json"  # video
AUDIO_TOKENIZER_NAME = "audio_tokenizer_config.json"  # audio, audio_tokenizer, 在 Processor 中被用到, 目前用的很少
PROCESSOR_NAME = "processor_config.json"   # 综合 tokenizer, image_processor, video_processor, feature_extractor
GENERATION_CONFIG_NAME = "generation_config.json"
MODEL_CARD_NAME = "modelcard.json"
# 注意 audio 和 video 是默认用同样的配置文件名

# audio_tokenizer 目前用的很少, 仅见于
# https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/dia#transformers.DiaProcessor
```

#### Processor

**SamImageProcessor**

```
PushToHubMixin
- push_to_hub
- (内部方法) _create_repo
- (内部方法) _get_files_timestamps
- (内部方法) _upload_modified_files

# transformers.image_processing_base.py
ImageProcessingMixin(PushToHubMixin)
- __init__
- fetch_images
- from_pretrained
- save_pretrained
- (内部方法) get_image_processor_dict
- (内部方法) from_dict
- (内部方法) to_dict
- (内部方法) from_json_file
- (内部方法) to_json_string
- (内部方法) to_json_file
- (具体在哪用未知) register_for_auto_class
- (不知道在哪用上) _set_processor_class

# transformers.image_processing_utils.py 
BaseImageProcessor(ImageProcessingMixin)
- __init__
- is_fast
- __call__: 调用 preprocess
- (由子类实现) preprocess
- (公共方法) rescale
- (公共方法) normalize
- (公共方法) center_crop
- to_dict

# transformers.models.sam.image_processing_sam.py
SamImageProcessor(BaseImageProcessor)
```


```
# transformers.processing_utils.py
ProcessorMixin(PushToHubMixin)
- __call__: 默认实现是使用 self.tokenizer, self.image_processor, self.video_processor, self.feature_extractor 分别处理各个模态的输入
- from_pretrained
- save_pretrained
- ...

SamProcessor(ProcessorMixin)
- __call__: 重写了父类的方法
```






```python
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
from transformers import AutoImageProcessor, AutoProcessor

# model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)

# processor = SamProcessor.from_pretrained("./sam-vit-huge")
processor = AutoProcessor.from_pretrained("./sam-vit-huge")
# image_processor = AutoImageProcessor.from_pretrained("./sam-vit-huge")

# 这张图片宽为2646, 长为1764, 即为宽图
img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

input_points = [[[450, 600]]]  # 注意这里的 450, 600 是针对原始图片的, 450是宽, 600是高, 原点是左上角
inputs = processor(raw_image, input_points=input_points, return_tensors="np")

# inputs: BatchFeature, key 包含下面的内容:
# (Pdb) inputs['pixel_values'].shape
# (1, 3, 1024, 1024)
# 已经是除以255.0然后减均值除方差以后的数值了
# (Pdb) inputs['original_sizes']
# array([[1764, 2646]])
# 注意这里是 (h, w), 是 numpy 的习惯
# (Pdb) inputs['reshaped_input_sizes']
# array([[ 683, 1024]])
# (Pdb) inputs['input_points']
# array([[[174.14965986, 232.31292517]]])


# 可以传一个 batch 的图片进去, 但是每张图片的 point 也得固定(不能第1张图片传3个point,第2张图片传2个point)
crop_image = raw_image.crop([200, 200, 1000, 1000])
input_points = [[[450, 600], [300, 400], [20, 300]], [[300, 400], [450, 600], [20, 300]]]
inputs = processor([raw_image, crop_image], input_points=input_points, return_tensors="np")

# input_points list 可以传入的形状 与输出的 inputs["input_points"] 的形状
# B: 图片数量, N: 每个图片中的物体数量, P: 每个物体的参考点数量, 2: x与y
# [B, N, P, 2] -> [B, N, P, 2]
# [B, P, 2] -> [B, 1, P, 2]
# [B, 1, 2] -> [B, 1, 1, 2]
# [B, 2] -> error
```

SAM 在上述处理的实际过程是:
(1) 图片: 先将长边对齐到1024,然后normalize(先除以255,然后减均值,除以方差),在右下角做padding,直至对其1024
(2) `input_points` 按比例放缩倍数

关于 `input_points` 的进一步说明: `input_points` 可以传 3 维或者 4 维, 传 3 维的情况就是每张图片只要检测一个目标(但目标可以用多个点来标记)


```python
# 接着前面的代码
# inputs: pixel_values, original_sizes, reshaped_input_sizes, input_points
with torch.no_grad():
    outputs = model(**inputs)

# outputs.iou_scores: (B, N, 3), 这里的这个 3 好像就是固定值, 3 个结果
# outputs.pred_masks: 浮点数: (B, N, 3, 256, 256)
masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
# len(masks) = B, type(masks) == list
# masks[0].shape = (N, 3, H, W)
scores = outputs.iou_scores
```



**一个完整的例子**

下面的例子中:

输入的图片是 1764 x 2646 大小, 输入模型前会将其长边放缩到 1024 并在右/下部分padding, 模型输入的图片维度是 1024x1024, 而模型输出的 mask 是 (B, N, 3, 256, 256). 其中 B=1 是图片数量, N=2 是需要检测的物体数. 也就是说模型输出的 mask 的长/宽是要输入大小的 1/4.

- B: `batch_size`, 图片数量
- N: `point_batch_size`: 物体数量, 每个图片中有 N 个物体
- P: `num_points_per_image`: 每个物体用 P 个点做标识

```python
import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
from transformers import AutoImageProcessor, AutoProcessor
import torch
import os

model_name_or_path = "facebook/sam-vit-huge"
model_name_or_path = "./sam-vit-huge"
model = SamModel.from_pretrained(model_name_or_path)
processor = SamProcessor.from_pretrained(model_name_or_path)

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

input_points = [[
    [
        [450, 600],
        [460, 600],
        [470, 600]
    ],
    [
        [900, 600],
        [910, 600],
        [920, 600]
    ],
]]
# B=1, N=2, P=3. 实际上分别指向图片中左边和右边的窗户
# B: 图片数量, N: 每个图片中的物体数量, P: 每个物体的参考点数量, 2: x与y
# [B, N, P, 2] -> [B, N, P, 2]

inputs = processor(
    raw_image,
    input_points=input_points,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)
# outputs.iou_scores: (1, 2, 3), 也就是: (B, N, 3)
# outputs.pred_masks: (1, 2, 3, 256, 256), 浮点数, 也就是: (B, N, 3, 256, 256)

# post_process_masks 实际做的事情是把模型输出的 outputs.pred_masks 放缩回原始尺寸
# 然后根据该像素位置的值来判断是否为物体: 大于0.0则为物体,小于0.0则不是物体
masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
# masks: list[tensor], masks[0] 是 bool 型 tensor, False 表示不是物体, True 表示物体区域
# len(masks) = 1 = B
# masks[0].shape = (2, 3, 1764, 2646) = (N, 3, H, W)
# 把预测 mask 输出为图片
os.makedirs("output_masks", exist_ok=True)
for mask_idx, mask in enumerate(masks):
    N, three, _, _ = mask.shape
    for i in range(N):
        for j in range(three):
            img = mask[i, j].cpu().numpy().astype("uint8") * 255
            img = Image.fromarray(img)
            img.save(f"output_masks/image_{mask_idx}_object_{i}_result_{j}.png")

# 输出的mask的展示
# image_0_object_0_result_0.png  -> 左侧窗户
# image_0_object_0_result_1.png  -> 两侧窗户
# image_0_object_0_result_2.png  -> 整个车子
# image_0_object_0_result_0.png  -> 右侧窗户
# image_0_object_0_result_1.png  -> 两侧窗户
# image_0_object_0_result_2.png  -> 整个车子
```

Point, Box, Mask

point1: (400, 500), label: 1 (物体)    embedding + w1
point2: (500, 600), label: 0 (非物体)  embedding + w0
point3: (600, 700), label: 1 (非物体)   embedding + w1
(??)point4: label: -10, embedding -> 全0向量

没有 box 时
追加一个 point: v0 (可学习的向量)

有 box 时, 追加啊 2 个 point (box 的左上和右下角)
p1: embedding + w3
p2: embedding + w4

总结下来:

有 point, 无 box 时:
sparse_embeddings = (B, N, P+1, 256)

有 point, 有 box 时:
sparse_embeddings = (B, N, P+2, 256)

无 point, 有 box 时:
sparse_embeddings = (B, N, 2, 256)


输入:(B, 1, 256, 256)
输出:
dense_embeddings = (B, 256, 64, 64)

或者是: (B, 256, 64, 64), 实际上是 256 个可学参数 repeat 出来的



mask-decoder

复习一下 attention 的基础:

```
输入形状:
Q: (B, L, C)
K: (B, M, C)
V: (B, M, D)
输出形状
(B, L, D)
```

在 mask-decoder 中, `C=D`, 而 Q, K, V 在三种 attention layer 中有所不同

```
(1) point-to-point
Q: (B, N, 5+P+1, 256)
K: (B, N, 5+P+1, 256)
V: (B, N, 5+P+1, 256)
(2) point-to-dense
Q: (B, N, 5+P+1, 256)
K: (B*N, 1, 64*64, 256)
V: (B*N, 1, 64*64, 256)
(3) dense-to-point
Q: (B*N, 1, 64*64, 256)
K: (B, N, 5+P+1, 256)
V: (B, N, 5+P+1, 256)
```

TODO: 卷积与反卷积的理解


## 模型整体流程

按 huggingface 中 SamModel 的 forward 函数来看:

```python
# pixel_values 和 image_embeddings 必须指定其一: 其实就是是否预先用 image_encoder 通过 pixel_value 计算了 image_embeddings
pixel_values: Optional[torch.FloatTensor] = None
"""(B, 3, 1024, 1024), B 代表图片数量, 即 batch_size"""
image_embeddings: Optional[torch.FloatTensor] = None
"""(B, 256, 64, 64)"""

# 这两个是匹配出现的
input_points: Optional[torch.FloatTensor] = None
"""(B, N, P, 2), 每张图片里有 N 个物体, 每个物体有 P 个指示点, 2 代表 x,y 坐标, 这里的 input_points 的 x,y 坐标已经是处理到 (0, 1024) 的范围里了"""
input_labels: Optional[torch.LongTensor] = None
"""(B, N, P), 每个点的标识, 推理时调用方传参取值只能是 0/1, 0代表该指示点不属于目标物体, 1代表该指示点属于该目标物体, 内部推理过程还用到了 -1 和 -10 的情况, 其中 -1 代表调用方没有给出 input_boxes 信息, 用特殊的指示点(0,0)和-1作为label来标识, -10的含义未知. 假设调用方只传 input_points 而没有传 input_labels, 则使用全1张量, 即默认调用者给的物体指示点都属于需要分割的物体"""

input_boxes: Optional[torch.FloatTensor] = None
"""(B, N, 4), 目标物体的 bounding box 框, 使用 (x1, y1, x2, y2) 的格式, 坐标同样是已经处理到 (0, 1024) 的范围了, 后续处理时是将 bounding box 作为两个指示点(即左上角,右下角)进行处理的"""
input_masks: Optional[torch.LongTensor] = None
"""(B, 1, 256, 256), TODO: 含义不明, 主要是为什么第二维与 N 无关, 也许只适用于 N=1 的情况"""
# 另外这里 input_masks 的 LongTensor 的 type-hint 估计是弄错了, 原始代码的注释里实际上写对了是 FloatTensor

multimask_output: bool = True
"""模型最终会输出两个最重要的东西: iou_scores: (B, N, K), pred_masks: (B, N, K, 256, 256), K原本是4, 其中 iou_scores 的取值是正负浮点数(很奇怪,本来应该归一化到0-1之间的), 如果传入 multimask_output=True, 则截取后面 3 个mask"""
attention_similarity: Optional[torch.FloatTensor] = None
target_embedding: Optional[torch.FloatTensor] = None
```

```
images: [(H, W, 3)]*B, B张图片, 大小可以任意

# points 和 labels 必须配对
points: (B, N, P, 2), 每张图片有N个物体,每个物体用P个点进行提示
labels: (B, N)  # -1, 0, 1, -10

boxes: (B, N, 4), 每张图片的N个物体用 x1, y1, x2, y2 来指示左上角和右下角
masks: (B, 1, 256, 256)  这个输入很奇怪, 代表什么提示, 并且这个输入数据是浮点数类型, 并且也不能传 N 个进去

attention_similarity
target_embedding

```


import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import (
    T5Block, T5Stack, T5LayerSelfAttention, T5Attention,
    T5LayerFF, T5LayerNorm, T5LayerCrossAttention)
import torch.nn as nn
device = "cpu"

import torch
torch.manual_seed(0)
import random
random.seed(0)
import random
random.seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

tokenizer = T5Tokenizer.from_pretrained("./PromptCLUE-base")
model = T5ForConditionalGeneration(T5Config.from_json_file("./promptclue-base-config.json"))

x = [
    "哪个类别最好的描述了这篇新闻？汶川地震10周年丨航拍新北川 楼房拔地起 旧貌换新颜\n选项：故事，文化，娱乐，体育，财经，房产，汽车，教育，科技，军事，旅游，国际，股票，农业，游戏\n答案：",
    "“现在买不是很好的时机了”我们这样说有道理吗“现在能以历史最低价买到”？是的,不是,或也许？\n答案："
]
y = [
    "国际",
    "不是"
]

source = tokenizer.batch_encode_plus(
    x,
    max_length=512,
    pad_to_max_length=True,
    truncation=True,
    padding="max_length",
    return_tensors="pt",
)
target = tokenizer.batch_encode_plus(
    y,
    max_length=64,
    pad_to_max_length=True,
    truncation=True,
    padding="max_length",
    return_tensors="pt",
)

data = {
    "source_ids": source["input_ids"],
    "source_mask": source["attention_mask"],
    "target_ids": target["input_ids"],
    "target_mask": target["attention_mask"]
}

y = data["target_ids"].to(device, dtype=torch.long)
y_ids = y[:, :-1].contiguous()
lm_labels = y[:, 1:].clone().detach()
lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
ids = data["source_ids"].to(device, dtype=torch.long)
mask = data["source_mask"].to(device, dtype=torch.long)


def model_forward(model, ids, mask, y_ids, lm_labels):
    outputs = model(
        input_ids=ids,
        attention_mask=mask,
        decoder_input_ids=y_ids,
        labels=lm_labels,
    )
    loss = outputs[0]
    return loss

def encoder_block_forward(block, block_idx, hidden, mask, position_bias=None):
    """
    block: T5Block
    block_idx (int):
    hidden: (B, src_len, C)
    mask: (B, src_len), 1/0, 1表示需要attention
    position_bias: (B, n_heads, src_len, src_len)
    """
    ori_hidden = hidden
    B, L, C = hidden.shape
    attention_layer: T5LayerSelfAttention = block.layer[0]
    
    layer_norm_1: T5LayerNorm = attention_layer.layer_norm
    hidden = layer_norm_1(hidden)
    
    # ============= begin T5Attention ===============
    self_attention: T5Attention = attention_layer.SelfAttention
    n_heads = self_attention.n_heads
    head_dim = self_attention.key_value_proj_dim
    if block_idx == 0:
        # TODO
        position_bias = self_attention.compute_bias(L, L)  # (1, n_head, L, L) -inf/其他
        # mask: 1的位置填充0, 0的位置填充-inf
        mask_transformed = torch.zeros((B, 1, 1, L)).masked_fill(1 - mask.view((B, 1, 1, L)), torch.finfo(torch.float32).min)
        # input_ids: [[3, 4, 5, <pad>, <pad>]]
        # =>
        # mask: [1, 1, 1, 0, 0]
        # => mask_transformed[0, 0]:
        # [[0, 0, 0, -inf, -inf],
        # [0, 0, 0, -inf, -inf],
        # [0, 0, 0, -inf, -inf],
        # [0, 0, 0, -inf, -inf],
        # [0, 0, 0, -inf, -inf]]
        position_bias = position_bias + mask_transformed
    q: nn.Linear = self_attention.q
    k: nn.Linear = self_attention.k
    v: nn.Linear = self_attention.v
    o: nn.Linear = self_attention.o

    query = q(hidden).view(B, L, n_heads, head_dim).transpose(1, 2)
    key = k(hidden).view(B, L, n_heads, head_dim).transpose(1, 2)
    value = v(hidden).view(B, L, n_heads, head_dim).transpose(1, 2)
    
    scores = torch.matmul(query, key.transpose(3, 2))  # (B, n_heads, L, L)
    scores += position_bias
    weight = nn.functional.softmax(scores, dim=-1)
    weight = nn.functional.dropout(weight, p=self_attention.dropout, training=self_attention.training)
    # weight 的 q, k 分别为第2, 3维
    output = torch.matmul(weight, value)  # (B, n_head, L, head_dim)
    output = output.transpose(1, 2).contiguous().reshape(B, L, n_heads * head_dim)
    output = o(output)
    # ====== end T5Attention ========

    output = attention_layer.dropout(output) + ori_hidden
    ff_layer: T5LayerFF = block.layer[1]
    output = ff_layer(output)
    return output, position_bias

def decoder_self_attention_forward(
        attention_layer: T5LayerSelfAttention,
        block_idx,
        hidden,
        mask,
        position_bias=None
    ):
    """
    attention_layer: T5LayerSelfAttention (每个decoder的T5Block的第一个层, 共3层)
    block_idx (int): 第几个decoder block
    hidden: (B, trg_len, C), decoder当前层的输入
    mask: (B, 1, trg_len, trg_len): 下三角及对角线部分为0, 其余为-inf
    position_bias: None/(B, n_heads, trg_len, trg_len)

    Returns:
    output: (B, trg_len, C)
    position_bias: (B, n_heads, trg_len, trg_len)
    """

    B, L, C = hidden.shape
    ori_hidden = hidden
    layer_norm_1: T5LayerNorm = attention_layer.layer_norm
    hidden = layer_norm_1(hidden)
    
    # ============= begin T5Attention ===============
    self_attention: T5Attention = attention_layer.SelfAttention
    n_heads = self_attention.n_heads
    head_dim = self_attention.key_value_proj_dim
    if block_idx == 0:
        # TODO
        position_bias = self_attention.compute_bias(L, L)  # (1, n_head, L, L) -inf/其他
        position_bias = position_bias + mask
    q: nn.Linear = self_attention.q
    k: nn.Linear = self_attention.k
    v: nn.Linear = self_attention.v
    o: nn.Linear = self_attention.o

    query = q(hidden).view(B, L, n_heads, head_dim).transpose(1, 2)
    key = k(hidden).view(B, L, n_heads, head_dim).transpose(1, 2)
    value = v(hidden).view(B, L, n_heads, head_dim).transpose(1, 2)
    
    scores = torch.matmul(query, key.transpose(3, 2))  # (B, n_heads, L, L)
    scores += position_bias
    weight = nn.functional.softmax(scores, dim=-1)
    weight = nn.functional.dropout(weight, p=self_attention.dropout, training=self_attention.training)
    # weight 的 q, k 分别为第2, 3维
    output = torch.matmul(weight, value)  # (B, n_head, L, head_dim)
    output = output.transpose(1, 2).contiguous().reshape(B, L, n_heads * head_dim)
    output = o(output)
    # ====== end T5Attention ========

    output = attention_layer.dropout(output) + ori_hidden
    return output, position_bias

def decoder_cross_attention_forward(
        layer: T5LayerCrossAttention,
        decoder_hidden,  # decoder上一层输出
        block_idx,
        encoder_hidden,  # encoder输出
        encoder_attention_mask,
        encoder_decoder_position_bias=None
    ):
    """
    decoder_hidden: (B, trg_len, C)
    block_idx (int):
    encoder_hidden: (B, src_len, C)
    encoder_attention_mask: (B, 1, 1, src_len) 0/-inf mask
    encoder_decoder_position_bias: (1, n_heads, trg_len, src_len)
    """
    ori_decoder_hidden = decoder_hidden
    decoder_hidden = layer.layer_norm(decoder_hidden)

    B, trg_len, C = decoder_hidden.shape
    src_len = encoder_hidden.shape[1]

    cross_attention: T5Attention = layer.EncDecAttention
    n_heads = cross_attention.n_heads
    head_dim = cross_attention.key_value_proj_dim
    if block_idx == 0:
        encoder_decoder_position_bias = torch.zeros((1, n_heads, trg_len, src_len))
        encoder_decoder_position_bias = encoder_decoder_position_bias + encoder_attention_mask
    q: nn.Linear = cross_attention.q
    k: nn.Linear = cross_attention.k
    v: nn.Linear = cross_attention.v
    o: nn.Linear = cross_attention.o

    # (B, n_heads, trg_len, head_dim)
    query = q(decoder_hidden).view(B, trg_len, n_heads, head_dim).transpose(1, 2)
    # (B, n_heads, src_len, head_dim)
    key = k(encoder_hidden).view(B, src_len, n_heads, head_dim).transpose(1, 2)
    # (B, n_heads, src_len, head_dim)
    value = v(encoder_hidden).view(B, src_len, n_heads, head_dim).transpose(1, 2)
    
    scores = torch.matmul(query, key.transpose(3, 2))  # (B, n_heads, trg_len, src_len)
    scores += encoder_decoder_position_bias
    weight = nn.functional.softmax(scores, dim=-1)
    weight = nn.functional.dropout(weight, p=cross_attention.dropout, training=cross_attention.training)
    # weight 的 q, k 分别为第2, 3维
    output = torch.matmul(weight, value)  # (B, n_head, trg_len, head_dim)
    output = output.transpose(1, 2).contiguous().reshape(B, trg_len, n_heads * head_dim)
    output = o(output)
    # ====== end T5Attention ========

    output = layer.dropout(output) + ori_decoder_hidden
    return output, encoder_decoder_position_bias

def manual_forward(model, ids, mask, y_ids, lm_labels):
    """
    model (T5ForConditionalGeneration):
    ids: (B, src_len), [0, vab_size), long
    mask: (B, src_len), 0或1, long, 1表示ids在此处为普通字符, 0表示ids在此处为padding字符
    y_ids: (B, trg_len), [0, vab_size), long
    lm_label: (B, trg_len), [0, vab_size)或-100, long
    """
    # embedding
    hidden = model.shared(ids)  # (B, src_len) -> (B, src_len, C)
    
    # encoder
    position_bias = None
    encoder = model.encoder

    hidden = encoder.dropout(hidden)
    for block_idx, block in enumerate(encoder.block):
        hidden, position_bias = encoder_block_forward(block, block_idx, hidden, mask, position_bias)
    hidden = encoder.final_layer_norm(hidden)
    hidden = encoder.dropout(hidden)

    # decoder
    B, trg_len = y_ids.shape
    decoder = model.decoder
    decoder_hidden = model.shared(y_ids)  # 与encoder采用相同的embedding
    decoder_hidden = decoder.dropout(decoder_hidden)
    
    decoder_attention_mask = torch.ones(B, trg_len)
    seq_ids = torch.arange(trg_len)
    # causal_mask (B, trg_len, trg_len): 下三角含对角线部分全为True, 其余部分为True
    causal_mask = seq_ids[None, None, :].repeat(B, trg_len, 1) <= seq_ids[None, :, None]
    # (B, 1, trg_len, trg_len)
    causal_mask = causal_mask[:, None, :, :] * decoder_attention_mask[:, None, None, :]
    causal_mask = (1.0 - causal_mask) * torch.finfo(torch.float32).min

    encoder_attention_mask = (1 - mask[:, None, None, :].to(torch.float32)) * -1e9
    decoder_position_bias = None
    encoder_decoder_position_bias = None
    for block_idx, block in enumerate(decoder.block):
        decoder_hidden, decoder_position_bias = decoder_self_attention_forward(
            block.layer[0],  # T5LayerSelfAttention
            block_idx, decoder_hidden, causal_mask, decoder_position_bias)
        # from pdb import set_trace; set_trace()
        decoder_hidden, encoder_decoder_position_bias = decoder_cross_attention_forward(
            block.layer[1],  # T5LayerCrossAttention
            decoder_hidden,  # decoder上一层输出
            block_idx,
            hidden,  # encoder输出
            encoder_attention_mask,
            encoder_decoder_position_bias
            )
        # block.layer[2]: T5LayerFF
        decoder_hidden = block.layer[2](decoder_hidden)
    decoder_hidden = decoder.final_layer_norm(decoder_hidden)
    decoder_hidden = decoder.dropout(decoder_hidden)
    
    if model.config.tie_word_embeddings:
        decoder_hidden = decoder_hidden * (model.model_dim ** -0.5)
    # model.lm_head: Linear, may tie with embedding
    lm_logits = model.lm_head(decoder_hidden)

    loss = None
    if lm_labels is not None:
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
    # from pdb import set_trace; set_trace()
    return loss


if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-manual", action="store_true", default=False)
    args = parser.parse_args()
    if args.use_manual:
        t1 = time.time()
        loss = manual_forward(model, ids, mask, y_ids, lm_labels)
        loss.backward()
        t2 = time.time()
        print(f"{t2 - t1}")
        torch.save([p.grad for p in model.parameters()], "xx.pth")
        # from pdb import set_trace; set_trace()
        x = 1
    else:
        t1 = time.time()
        loss = model_forward(model, ids, mask, y_ids, lm_labels)
        loss.backward()
        t2 = time.time()
        print(f"{t2 - t1}")
        torch.save([p.grad for p in model.parameters()], "yy.pth")
    # Then I check xx.pth and yy.pth is allclose
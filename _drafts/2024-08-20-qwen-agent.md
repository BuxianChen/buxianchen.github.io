---
layout: post
title: "(P0) Qwen-Agent"
date: 2024-08-20 10:00:00 +0800
labels: [llm,qwen,agent]
---

## DEMO & 引入

本节 demo 内容完全参考自 README: [https://github.com/QwenLM/Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)

**启动大模型服务**

注意用 vllm 启动的服务不包含原生的 `function_call` 功能

```bash
GIT_LFS_SKIP_SMUDGE=0 git clone https://huggingface.co/Qwen/Qwen2-0.5B-Instruct
python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-0.5B-Instruct --model /root/autodl-tmp/Qwen2-0.5B-Instruct --dtype auto --port 8000 --host 0.0.0.0
```

**Qwen-Agent 的使用**

使用 Qwen-Agent 获取 function call 功能

```python
# pip install qwen-agent[gui,rag,code_interpreter]
import json
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

@register_tool('my_image_gen')
class MyImageGen(BaseTool):
    description = 'AI画图工具,返回图片URL给用户'
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': '期望的图像内容的详细描述,注意描述必须用英文',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        prompt = json.loads(params)['prompt']
        return json.dumps({'image_url': f'image/{prompt}'}, ensure_ascii=False)


llm_cfg = {
    'model': 'Qwen2-0.5B-Instruct',
    'model_server': 'http://localhost:8000/v1',
    'api_key': 'EMPTY',
}


bot = Assistant(llm=llm_cfg,
                function_list=["my_image_gen"],
                files=None)

messages = [{"role": "user", "content": "帮我画张小狗在草地上的图片"}]
for r in bot.run(messages=messages):
    print(r)
```

**输出**

<details>
<summary>
详细输出
</summary>

```
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'm', 'arguments': ''}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_im', 'arguments': ''}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_ima', 'arguments': ''}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_imag', 'arguments': ''}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_ge', 'arguments': ''}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': ''}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': ''}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': ''}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"pr'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prom'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片"'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","im'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"!'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![]'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站在'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站在草地'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站在草地上的'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站在草地上的图片'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站在草地上的图片。!'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站在草地上的图片。![小'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站在草地上的图片。![小狗站'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站在草地上的图片。![小狗站在草'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站在草地上的图片。![小狗站在草地上的图片'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站在草地上的图片。![小狗站在草地上的图片]'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站在草地上的图片。![小狗站在草地上的图片](i'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站在草地上的图片。![小狗站在草地上的图片](ima'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站在草地上的图片。![小狗站在草地上的图片](image'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站在草地上的图片。![小狗站在草地上的图片](image/小'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站在草地上的图片。![小狗站在草地上的图片](image/小狗站'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站在草地上的图片。![小狗站在草地上的图片](image/小狗站在'}]
[{'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}}, {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'}, {'role': 'assistant', 'content': '已为您画出小狗站在草地上的图片。![小狗站在草地上的图片](image/小狗站在草地上的图片)'}]
```

</details>

由此可见, Qwen-Agent 的 Assistant 一共产生了 3 句输出:

```python
[
    {'role': 'assistant', 'content': '', 'function_call': {'name': 'my_image_gen', 'arguments': '{"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}'}},
    {'role': 'function', 'content': '{"image_url": "image/小狗站在草地上的图片"}', 'name': 'my_image_gen'},
    {'role': 'assistant', 'content': '已为您画出小狗站在草地上的图片。![小狗站在草地上的图片](image/小狗站在草地上的图片)'}
]
```

备注: 这里的第一句输出由于大模型理解能力有限, 它多传了个 `image_url` 参数

**vllm后台输出**

<details>
<summary>
关键的后台原始输出
</summary>

```
INFO 08-20 10:32:24 async_llm_engine.py:553] Received request cmpl-2701eb0293494b56818e18b29517c1c9: prompt: '<|im_start|>system\nYou are a helpful assistant.\n\n# 工具\n\n## 你拥有如下工具：\n\n### my_image_gen\n\nmy_image_gen: AI画图工具,返回图片URL给用户 输入参数：[{"name": "prompt", "type": "string", "description": "期望的图像内容的详细描述,注意描述必须用英文", "required": true}] 此工具的输入应为JSON对象。\n\n## 你可以在回复中插入零次、一次或多次以下命令以调用工具：\n\n✿FUNCTION✿: 工具名称，必须是[my_image_gen]之一。\n✿ARGS✿: 工具输入\n✿RESULT✿: 工具结果\n✿RETURN✿: 根据工具结果进行回复，需将图片用![](url)渲染出来<|im_end|>\n<|im_start|>user\n帮我画张小狗在草地上的图片<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.7, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=['✿RESULT✿', '✿RETURN✿'], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=32583, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), prompt_token_ids: [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 382, 2, 83002, 98, 76813, 271, 565, 220, 56568, 103926, 104506, 102011, 48443, 14374, 847, 4954, 16322, 271, 2408, 4954, 16322, 25, 15235, 54623, 28029, 102011, 11, 31526, 45930, 3144, 89012, 20002, 69058, 32665, 5122, 58, 4913, 606, 788, 330, 40581, 497, 330, 1313, 788, 330, 917, 497, 330, 4684, 788, 330, 106076, 9370, 107553, 43815, 9370, 100700, 53481, 11, 60533, 53481, 100645, 11622, 105205, 497, 330, 6279, 788, 830, 25439, 71928, 97, 102011, 9370, 31196, 50511, 17714, 5370, 64429, 3407, 565, 220, 56568, 104964, 104787, 15946, 114731, 99822, 32571, 5373, 99796, 57191, 104183, 87752, 106167, 23031, 47872, 11622, 102011, 48443, 144575, 18149, 144575, 25, 83002, 98, 76813, 29991, 3837, 100645, 20412, 58, 2408, 4954, 16322, 60, 100653, 8997, 144575, 47483, 144575, 25, 83002, 98, 76813, 31196, 198, 144575, 14098, 144575, 25, 83002, 98, 76813, 59151, 198, 144575, 51533, 144575, 25, 51461, 117, 16038, 102011, 59151, 71817, 104787, 3837, 58362, 44063, 45930, 11622, 0, 50994, 1085, 8, 115876, 99898, 151645, 198, 151644, 872, 198, 108965, 54623, 86341, 115441, 18493, 114654, 101913, 45930, 151645, 198, 151644, 77091, 198], lora_request: None

INFO 08-20 10:32:25 async_llm_engine.py:553] Received request cmpl-513739efcbc046d2a8a553a6dec74ad3: prompt: '<|im_start|>system\nYou are a helpful assistant.\n\n# 工具\n\n## 你拥有如下工具：\n\n### my_image_gen\n\nmy_image_gen: AI画图工具,返回图片URL给用户 输入参数：[{"name": "prompt", "type": "string", "description": "期望的图像内容的详细描述,注意描述必须用英文", "required": true}] 此工具的输入应为JSON对象。\n\n## 你可以在回复中插入零次、一次或多次以下命令以调用工具：\n\n✿FUNCTION✿: 工具名称，必须是[my_image_gen]之一。\n✿ARGS✿: 工具输入\n✿RESULT✿: 工具结果\n✿RETURN✿: 根据工具结果进行回复，需将图片用![](url)渲染出来<|im_end|>\n<|im_start|>user\n帮我画张小狗在草地上的图片\n\n✿FUNCTION✿: my_image_gen\n✿ARGS✿: {"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}\n✿RESULT✿: {"image_url": "image/小狗站在草地上的图片"}\n✿RETURN✿<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.7, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=['✿RESULT✿', '✿RETURN✿'], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=32529, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), prompt_token_ids: [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 382, 2, 83002, 98, 76813, 271, 565, 220, 56568, 103926, 104506, 102011, 48443, 14374, 847, 4954, 16322, 271, 2408, 4954, 16322, 25, 15235, 54623, 28029, 102011, 11, 31526, 45930, 3144, 89012, 20002, 69058, 32665, 5122, 58, 4913, 606, 788, 330, 40581, 497, 330, 1313, 788, 330, 917, 497, 330, 4684, 788, 330, 106076, 9370, 107553, 43815, 9370, 100700, 53481, 11, 60533, 53481, 100645, 11622, 105205, 497, 330, 6279, 788, 830, 25439, 71928, 97, 102011, 9370, 31196, 50511, 17714, 5370, 64429, 3407, 565, 220, 56568, 104964, 104787, 15946, 114731, 99822, 32571, 5373, 99796, 57191, 104183, 87752, 106167, 23031, 47872, 11622, 102011, 48443, 144575, 18149, 144575, 25, 83002, 98, 76813, 29991, 3837, 100645, 20412, 58, 2408, 4954, 16322, 60, 100653, 8997, 144575, 47483, 144575, 25, 83002, 98, 76813, 31196, 198, 144575, 14098, 144575, 25, 83002, 98, 76813, 59151, 198, 144575, 51533, 144575, 25, 51461, 117, 16038, 102011, 59151, 71817, 104787, 3837, 58362, 44063, 45930, 11622, 0, 50994, 1085, 8, 115876, 99898, 151645, 198, 151644, 872, 198, 108965, 54623, 86341, 115441, 18493, 114654, 101913, 45930, 271, 144575, 18149, 144575, 25, 847, 4954, 16322, 198, 144575, 47483, 144575, 25, 5212, 40581, 3252, 115441, 104224, 114654, 101913, 45930, 2198, 1805, 2903, 3252, 0, 50994, 115441, 104224, 114654, 101913, 45930, 9940, 532, 144575, 14098, 144575, 25, 5212, 1805, 2903, 788, 330, 1805, 14, 115441, 104224, 114654, 101913, 45930, 16707, 144575, 51533, 144575, 151645, 198, 151644, 77091, 198], lora_request: None.
```
</details>

从后台输出可以看出, 总共触发了 2 次对大模型的调用, 下面将 2 次调用的 prompt 进行格式化(仅忠实地转义换行符):

第 1 次大模型调用的 prompt:

```
<|im_start|>system
You are a helpful assistant.

# 工具

## 你拥有如下工具：

### my_image_gen

my_image_gen: AI画图工具,返回图片URL给用户 输入参数：[{"name": "prompt", "type": "string", "description": "期望的图像内容的详细描述,注意描述必须用英文", "required": true}] 此工具的输入应为JSON对象。

## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

✿FUNCTION✿: 工具名称，必须是[my_image_gen]之一。
✿ARGS✿: 工具输入
✿RESULT✿: 工具结果
✿RETURN✿: 根据工具结果进行回复，需将图片用![](url)渲染出来<|im_end|>
<|im_start|>user
帮我画张小狗在草地上的图片<|im_end|>
<|im_start|>assistant\n
```

第 2 次大模型调用的 prompt:

```
<|im_start|>system
You are a helpful assistant.

# 工具

## 你拥有如下工具：

### my_image_gen

my_image_gen: AI画图工具,返回图片URL给用户 输入参数：[{"name": "prompt", "type": "string", "description": "期望的图像内容的详细描述,注意描述必须用英文", "required": true}] 此工具的输入应为JSON对象。

## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

✿FUNCTION✿: 工具名称，必须是[my_image_gen]之一。
✿ARGS✿: 工具输入
✿RESULT✿: 工具结果
✿RETURN✿: 根据工具结果进行回复，需将图片用![](url)渲染出来<|im_end|>
<|im_start|>user
帮我画张小狗在草地上的图片

✿FUNCTION✿: my_image_gen
✿ARGS✿: {"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}
✿RESULT✿: {"image_url": "image/小狗站在草地上的图片"}
✿RETURN✿<|im_end|>
<|im_start|>assistant\n
```

**疑惑**

- function call 的 chatml 格式是什么
- 上面的例子中流式输出是怎么实现的(vllm提供了什么,qwen-agent怎么解析流式的json格式)
- qwen-agent 提供了哪些其他功能: 上面的只是个 demo


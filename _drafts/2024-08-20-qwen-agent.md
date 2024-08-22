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

{% raw %}
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
{% end raw %}
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

{% raw %}
```
INFO 08-20 10:32:24 async_llm_engine.py:553] Received request cmpl-2701eb0293494b56818e18b29517c1c9: prompt: '<|im_start|>system\nYou are a helpful assistant.\n\n# 工具\n\n## 你拥有如下工具：\n\n### my_image_gen\n\nmy_image_gen: AI画图工具,返回图片URL给用户 输入参数：[{"name": "prompt", "type": "string", "description": "期望的图像内容的详细描述,注意描述必须用英文", "required": true}] 此工具的输入应为JSON对象。\n\n## 你可以在回复中插入零次、一次或多次以下命令以调用工具：\n\n✿FUNCTION✿: 工具名称，必须是[my_image_gen]之一。\n✿ARGS✿: 工具输入\n✿RESULT✿: 工具结果\n✿RETURN✿: 根据工具结果进行回复，需将图片用![](url)渲染出来<|im_end|>\n<|im_start|>user\n帮我画张小狗在草地上的图片<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.7, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=['✿RESULT✿', '✿RETURN✿'], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=32583, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), prompt_token_ids: [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 382, 2, 83002, 98, 76813, 271, 565, 220, 56568, 103926, 104506, 102011, 48443, 14374, 847, 4954, 16322, 271, 2408, 4954, 16322, 25, 15235, 54623, 28029, 102011, 11, 31526, 45930, 3144, 89012, 20002, 69058, 32665, 5122, 58, 4913, 606, 788, 330, 40581, 497, 330, 1313, 788, 330, 917, 497, 330, 4684, 788, 330, 106076, 9370, 107553, 43815, 9370, 100700, 53481, 11, 60533, 53481, 100645, 11622, 105205, 497, 330, 6279, 788, 830, 25439, 71928, 97, 102011, 9370, 31196, 50511, 17714, 5370, 64429, 3407, 565, 220, 56568, 104964, 104787, 15946, 114731, 99822, 32571, 5373, 99796, 57191, 104183, 87752, 106167, 23031, 47872, 11622, 102011, 48443, 144575, 18149, 144575, 25, 83002, 98, 76813, 29991, 3837, 100645, 20412, 58, 2408, 4954, 16322, 60, 100653, 8997, 144575, 47483, 144575, 25, 83002, 98, 76813, 31196, 198, 144575, 14098, 144575, 25, 83002, 98, 76813, 59151, 198, 144575, 51533, 144575, 25, 51461, 117, 16038, 102011, 59151, 71817, 104787, 3837, 58362, 44063, 45930, 11622, 0, 50994, 1085, 8, 115876, 99898, 151645, 198, 151644, 872, 198, 108965, 54623, 86341, 115441, 18493, 114654, 101913, 45930, 151645, 198, 151644, 77091, 198], lora_request: None

INFO 08-20 10:32:25 async_llm_engine.py:553] Received request cmpl-513739efcbc046d2a8a553a6dec74ad3: prompt: '<|im_start|>system\nYou are a helpful assistant.\n\n# 工具\n\n## 你拥有如下工具：\n\n### my_image_gen\n\nmy_image_gen: AI画图工具,返回图片URL给用户 输入参数：[{"name": "prompt", "type": "string", "description": "期望的图像内容的详细描述,注意描述必须用英文", "required": true}] 此工具的输入应为JSON对象。\n\n## 你可以在回复中插入零次、一次或多次以下命令以调用工具：\n\n✿FUNCTION✿: 工具名称，必须是[my_image_gen]之一。\n✿ARGS✿: 工具输入\n✿RESULT✿: 工具结果\n✿RETURN✿: 根据工具结果进行回复，需将图片用![](url)渲染出来<|im_end|>\n<|im_start|>user\n帮我画张小狗在草地上的图片\n\n✿FUNCTION✿: my_image_gen\n✿ARGS✿: {"prompt":"小狗站在草地上的图片","image_url":"![](小狗站在草地上的图片)"}\n✿RESULT✿: {"image_url": "image/小狗站在草地上的图片"}\n✿RETURN✿<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.7, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=['✿RESULT✿', '✿RETURN✿'], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=32529, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), prompt_token_ids: [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 382, 2, 83002, 98, 76813, 271, 565, 220, 56568, 103926, 104506, 102011, 48443, 14374, 847, 4954, 16322, 271, 2408, 4954, 16322, 25, 15235, 54623, 28029, 102011, 11, 31526, 45930, 3144, 89012, 20002, 69058, 32665, 5122, 58, 4913, 606, 788, 330, 40581, 497, 330, 1313, 788, 330, 917, 497, 330, 4684, 788, 330, 106076, 9370, 107553, 43815, 9370, 100700, 53481, 11, 60533, 53481, 100645, 11622, 105205, 497, 330, 6279, 788, 830, 25439, 71928, 97, 102011, 9370, 31196, 50511, 17714, 5370, 64429, 3407, 565, 220, 56568, 104964, 104787, 15946, 114731, 99822, 32571, 5373, 99796, 57191, 104183, 87752, 106167, 23031, 47872, 11622, 102011, 48443, 144575, 18149, 144575, 25, 83002, 98, 76813, 29991, 3837, 100645, 20412, 58, 2408, 4954, 16322, 60, 100653, 8997, 144575, 47483, 144575, 25, 83002, 98, 76813, 31196, 198, 144575, 14098, 144575, 25, 83002, 98, 76813, 59151, 198, 144575, 51533, 144575, 25, 51461, 117, 16038, 102011, 59151, 71817, 104787, 3837, 58362, 44063, 45930, 11622, 0, 50994, 1085, 8, 115876, 99898, 151645, 198, 151644, 872, 198, 108965, 54623, 86341, 115441, 18493, 114654, 101913, 45930, 271, 144575, 18149, 144575, 25, 847, 4954, 16322, 198, 144575, 47483, 144575, 25, 5212, 40581, 3252, 115441, 104224, 114654, 101913, 45930, 2198, 1805, 2903, 3252, 0, 50994, 115441, 104224, 114654, 101913, 45930, 9940, 532, 144575, 14098, 144575, 25, 5212, 1805, 2903, 788, 330, 1805, 14, 115441, 104224, 114654, 101913, 45930, 16707, 144575, 51533, 144575, 151645, 198, 151644, 77091, 198], lora_request: None.
```
{% end raw %}
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

## DEMO 解谜

### `register_tool`

register 模式: 极简的 register 就像这样, `register_tool` 会略微复杂些, 但本质相同

```python
CLS_REGISTRY = dict()

def register_cls(name):
    def decorator(cls):
        cls.name = name
        CLS_REGISTRY[name] = cls
        return cls
    return decorator

@register_cls(name="name_a")
class A:
    pass

print(CLS_REGISTRY)
```

### `BaseTool`

`BaseTool` 的子类需要有 `name`, `parameters`, `description` 类属性, 并且包含一个 `call` 方法, 其中 `name` 属性可以由装饰器 `register_tool` 来赋予. 由于 `qwen_agent` 的代码组织形式, 继承自 `BaseTool` 的子类大概必须由 `register_tool` 进行装饰.

### function call

先暂时不看 Assistant 类, 单独研究下 [qwen_agent/llm/function_calling.py](https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/function_calling.py).

以下是自己用 vllm 部署的 Qwen 接口怎么实现与 OpenAI function call 类似的 API

```python
# OAI 大概是 OpenAI 的缩写
from qwen_agent.llm.oai import TextChatAtOAI

llm_cfg = {
    'model': 'Qwen2-0.5B-Instruct',
    'model_server': 'http://localhost:8000/v1',
    'api_key': 'EMPTY',
}
model = TextChatAtOAI(llm_cfg)

res = model.chat(
    messages=[{"role": "user", "content": "帮我画张小狗在草地上的图片"}],
    functions = [
        {
            "name": "my_image_gen",
            "description": "AI画图工具,返回图片URL给用户",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "期望的图像内容的详细描述,描述必须用英文描述"
                    }
                },
                "required": ["prompt"],
            }
        }
    ],
    # 也可以使用与 langchain_openai 的 ChatOpenAI 稍有区别的传递方式
    # functions = [
    #     {
    #         "name": "my_image_gen",
    #         "description": "AI画图工具,返回图片URL给用户",
    #         "parameters": [
    #             {
    #                 "name": "prompt",
    #                 "type": "string",
    #                 "description": "期望的图像内容的详细描述,描述必须用英文描述",
    #                 "required": True
    #             }
    #         ]
    #     }
    # ],
    stream = False
)

# res: 注意返回的结果也与 ChatOpenAI 几乎一致, 注意 arguments 是一个字符串 (json 格式转化为字典, 官方推荐用 json5 来处理)
# [
#     {
#         'role': 'assistant',
#         'content': '',
#         'function_call': {
#             'name': 'my_image_gen',
#             'arguments': '{"prompt": "a cute little dog playing on the green grass"}'
#         }
#     }
# ]
```

作为对比, 回顾一下 openai-python 的调用方式

```python
from openai import OpenAI
model = OpenAI()
res = model.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "帮我画张小狗在草地上的图片"}],
    functions = [
        {
            "name": "my_image_gen",
            "description": "AI画图工具,返回图片URL给用户",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "期望的图像内容的详细描述,描述必须用英文描述"
                    }
                },
                "required": ["prompt"],
            }
        }
    ],
    stream = False
)

print(res.choices[0].message.dict())
# {
#     'content': None,
#     'refusal': None,
#     'role': 'assistant',
#     'function_call': {
#         'arguments': '{\n  "prompt": "A small dog on a grassy field"\n}',
#         'name': 'my_image_gen'
#     },
#     'tool_calls': None
# }
```

## 其他探索

### 代码解释器

目前理解: QWen-Agent 中的代码解释器功能([qwen_agent/tools/code_interpreter.py](https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/tools/code_interpreter.py)) 的实现逻辑是启动一个 jupyter, 并且预先 import 好 matplotlib, numpy, pandas 等包, 然后每次由大模型写好代码后往 jupyter 里添加代码并执行, 并将运行结果 (执行结果里如果有图片的话会被捕获到并特殊处理) 作为角色为 `function` 的 message 添加到对话里. 这与 langchain 中的代码解释器的主要区别是, QWen-Agent 里的代码解释器可以使用前序步骤中定义的变量, 而 langchain 中的代码解释器每次都只能写全新的代码, 独立运行, 不能使用之前代码里的变量.

#### jupyter, ipykernel, jupyter-client

ipykernel 和 jupyter-client 是 jupyter 的核心组件, 安装 jupyter 时会自动装上 ipykernel 和 jupyter-client, 除此之外还会自动安装 jupyter notebook, jupyter lab 等. 但对于下面的 demo 来说, ipykernel 和 jupyter-client 已经足够. ipykernel 相当于 server 端, jupyter-client 相当于 client 端.

##### demo

**第一步**: 执行下面的脚本, 创建连接文件

备注: 很奇怪的是目前在 QWen-Agent 的实现中似乎没有这一步, 不太确定这个连接文件从何而来

```python
from jupyter_client import KernelManager
km = KernelManager()

# 生成连接文件，并获取路径
km.connection_file = "connection.json"
km.write_connection_file()
```

`connection.json` 的文件内容大致如下:

```json
{
  "shell_port": 55791,
  "iopub_port": 43629,
  "stdin_port": 47423,
  "control_port": 60551,
  "hb_port": 33619,
  "ip": "127.0.0.1",
  "key": "58007905-e9114700e4df9d6e74f7a027",
  "transport": "tcp",
  "signature_scheme": "hmac-sha256",
  "kernel_name": "python3"
}
```

**第二步**: 启动 kernel

```python
# start_kernel.py
from ipykernel import kernelapp as app
app.launch_new_instance()
```

启动方式如下, 默认在前台运行, 会占据一个终端

```bash
python start_kernel.py --IPKernelApp.connection_file=connection.json --matplotlib=inline
```

**第三步**: 连接 kernel 并执行一些命令以获取结果

```python
from jupyter_client import BlockingKernelClient

# 创建客户端实例，并加载连接文件
kc = BlockingKernelClient(connection_file="connection.json")
kc.load_connection_file()
kc.start_channels()
kc.wait_for_ready()

# 发送代码到内核并执行
kc.execute("print('Hello, world!')")

# 获取执行结果
reply = kc.get_shell_msg(timeout=5)

# 获取 stdout 的输出, 这个例子中的处理不完善, 如果代码执行没有输出会报错
while True:
    io_msg = kc.get_iopub_msg(timeout=5)
    if io_msg['msg_type'] == 'stream' and io_msg['content']['name'] == 'stdout':
        print(io_msg['content']['text'])
        break

# 可以继续 kc.execute(...)
```

<details>
<summary>
上面代码中 `reply` 的内容如下:
</summary>

{% raw %}
```python
{
    'header': {
        'msg_id': 'dc7fc97a-e66bc6bc20a4c2965b25b6b4_71700_7',
        'msg_type': 'execute_reply',
        'username': 'buxian',
        'session': 'dc7fc97a-e66bc6bc20a4c2965b25b6b4',
        'date': datetime.datetime(2024, 8, 21, 6, 25, 56, 558959, tzinfo=tzutc()),
        'version': '5.3'
    },
    'msg_id': 'dc7fc97a-e66bc6bc20a4c2965b25b6b4_71700_7',
    'msg_type': 'execute_reply',
    'parent_header': {
        'msg_id': '851154db-2fe3cb640e719346e8f546f2_71765_1',
        'msg_type': 'execute_request',
        'username': 'buxian',
        'session': '851154db-2fe3cb640e719346e8f546f2',
        'date': datetime.datetime(2024, 8, 21, 6, 25, 56, 542836, tzinfo=tzutc()),
        'version': '5.3'
    },
    'metadata': {
        'started': '2024-08-21T06:25:56.545541Z',
        'dependencies_met': True,
        'engine': '319d4b63-fce8-4473-8490-b3051b6c1edf',
        'status': 'ok'
    },
    'content': {
        'status': 'ok',
        'execution_count': 1,
        'user_expressions': {},
        'payload': []
    },
    'buffers': []
}
```
{% end raw %}
</details>

##### 官方文档

ipykernel 和 jupyter-client 的官方介绍

#### Qwen-Agent 中的使用方式

```python
# TODO
@register_tool('code_interpreter')
class CodeInterpreter(BaseToolWithFileAccess):
    def call(...):
        ...
```

TODO: 这在做啥?

```python
from jupyter_client import BlockingKernelClient
import subprocess

kernel_process = subprocess.Popen("python xx.py --IPKernelApp.connection_file xx.json --matplotlib=inline --quiet")

kc = BlockingKernelClient(connection_file=connection_file)
asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
kc.load_connection_file()
kc.start_channels()
kc.wait_for_ready()
# return kc, kernel_process
```

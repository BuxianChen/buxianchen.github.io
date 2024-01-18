---
layout: post
title: "(READY) HTTP with Streaming"
date: 2024-01-18 11:10:04 +0800
labels: [llm]
---

## 例子

客户端的用法应该与服务端的实现相关, 注意 OpenAI 与 Qwen 的小区别, 应该是 OpenAI 的实现会同时考虑请求参数里的 stream 参数, 以及请求的 stream 参数, 而 Qwen 因为采用 `sse_starlette.sse.EventSourceResponse` 所以只认请求的 stream 参数.

### OpenAI

#### Raw HTTP with Streaming (2024/01/18)

相关参考:

- [https://www.codingthesmartway.com/stream-responses-from-openai-api-with-python/](https://www.codingthesmartway.com/stream-responses-from-openai-api-with-python/)

```python
data = dict(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ],
    stream=True
)
url = "	https://api.openai.com/v1/chat/completions"
headers = {
    'Authorization': 'Bearer sk-xxx',
    'Content-Type': 'application/json',
    # 'Accept': 'text/event-stream',
  }

# 无需额外再设置 stream=True !!!
resp = requests.post(
    url,
    headers=headers,
    json=data
)
for line in resp.iter_lines():
    print(line)
```

#### openai-python v1.x with Streaming

```python
import openai

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
]

client = openai.OpenAI(api_key="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", base_url="https://api.openai.com/v1")

resp = client.chat.completions.create(
    messages=messages,
    model="gpt-3.5-turbo-1106",
    stream=True
)
for chunk in resp:
    print(chunk)
```


### Qwen

#### Raw HTTP with Streaming (2024/01/18)

服务端使用 [openai_api.py](https://github.com/QwenLM/Qwen/blob/204c2c59f49cfa7461e8e02d5ad2f6b3d082f08c/openai_api.py) 启动

客户端代码

```python
# 需要在请求数据以及request参数都设置 stream=True !!
data = dict(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ],
    stream=True
)
url = "	https://30.79.18.22/v1/chat/completions"  # ip 自行设置
headers = {
    'Content-Type': 'application/json',
  }
resp = requests.post(
    url,
    headers=headers,
    json=data,
    stream=True   # 如果这里不设置为 True, 会出现伪流式的情况, 即实际上是拿到全部流式结果后再触发打印的
)
for line in resp.iter_lines():
    print(line)
```

#### openai-python v1.x with Streaming

与上面 OpenAI 的例子中完全一致


### 最简例子

流式接口及客户端接收(参考 Qwen)

**服务端实现**

```python
# server.py
from fastapi import FastAPI
from pydantic import BaseModel
import json
import uvicorn
from sse_starlette.sse import EventSourceResponse
import time

class Simple(BaseModel):
    query: str

def character_stream_generator(query: str):
    for s in query:
        yield json.dumps({"chunk": s})
        time.sleep(0.3)

app = FastAPI()

@app.post("/stream_echo")
def echo(request: Simple):
    generator = character_stream_generator(request.query)
    return EventSourceResponse(generator, media_type='text/event-stream')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6543)
```

**客户端实现**

```python
import requests
from datetime import datetime

data = {"query": "echo this"}
url = "http://localhost:6543/stream_echo"
headers = {
    'Content-Type': 'application/json'
}
resp = requests.post(
    url,
    headers=headers,
    json=data,
    stream=True  # 必须设置, 否则会是伪流式
)

for line in resp.iter_lines():
    print(str(datetime.now()), line)
```

**客户端输出**

```
2024-01-18 14:31:45.604362 b'data: {"chunk": "e"}'
2024-01-18 14:31:45.604434 b''
2024-01-18 14:31:45.906128 b'data: {"chunk": "c"}'
2024-01-18 14:31:45.906387 b''
2024-01-18 14:31:46.209324 b'data: {"chunk": "h"}'
2024-01-18 14:31:46.209475 b''
2024-01-18 14:31:46.510782 b'data: {"chunk": "o"}'
2024-01-18 14:31:46.510901 b''
2024-01-18 14:31:46.814433 b'data: {"chunk": " "}'
2024-01-18 14:31:46.814671 b''
2024-01-18 14:31:47.116672 b'data: {"chunk": "t"}'
2024-01-18 14:31:47.116844 b''
2024-01-18 14:31:47.418330 b'data: {"chunk": "h"}'
2024-01-18 14:31:47.418405 b''
2024-01-18 14:31:47.720657 b'data: {"chunk": "i"}'
2024-01-18 14:31:47.720737 b''
2024-01-18 14:31:48.022229 b'data: {"chunk": "s"}'
2024-01-18 14:31:48.022355 b''
```
---
layout: post
title: "(READY) Socket and HTTP"
date: 2024-01-18 11:10:04 +0800
labels: [socket, http]
---

## 动机、参考资料、涉及内容

- 怎么使用 socket
- HTTP 协议的具体内容
- 怎么实现 HTTP 的 streaming 与 上传文件

## socket

以下是一个用于观察请求具体内容的 server 程序, 这份代码主要用来观察 requests 库发请求时的 HTTP 头与内容究竟是什么

```python
# socket_server.py
import socket

def start_server(host="127.0.0.1", port=8000):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(f"Server started at {host}:{port}")
    while True:
        conn, addr = s.accept()
        conn.settimeout(0.1)
        print(f"=====Connected by {addr} =======")
        file_data = b''
        while True:
            try:
                data = conn.recv(1024)
            except socket.timeout as e:
                break
            if not data:
                break
            print("--------------------")
            print(data.decode())
            file_data += data
        content = file_data.decode()
        response = f"HTTP/1.1 200 OK\nContent-Type: text/html\n\n{content}"
        conn.sendall(response.encode())
        conn.close()

start_server()
```

注意如果不设置 `conn.settimeout(0.1)`, 当客户端使用 `requests_client.py` 进行请求时, 在数据发送完毕时, 服务端的 `s.accept()` 会一直阻塞, 导致无法 `break`. flask/fastapi 里应该会去从请求里解析 `Content-Length` 等信息来处理 (具体机制不确定)

## HTTP 协议

### body 类型: text, json, form

json, text, form 本质上只是 body 的内容不一样, 且请求头里的 `Content-Type` 字段有所不同, 使用 requests 库发起三种请求的方式为:

```python
import requests
import json
data = {"a": ["abc", "def", {"bec": 1}]}
response = requests.post(url, data=json.dumps(data))  # text, data 是字符串类型时, 不会设置 header
response = requests.post(url, json=data)              # json, 会自动设置 header, {"Content-Type": "application/json"}
response = requests.post(url, data=data)              # form, data 是字典类型时, 会自动设置 header, {"Content-Type": "application/x-www-form-urlencoded"}, 这种情况下 data 不能是过于复杂的结构, 最好就是 Dict[str, str]
```

请求内容如下:

**text**

```
POST /upload/ HTTP/1.1
Host: 127.0.0.1:8000
User-Agent: python-requests/2.31.0
Accept-Encoding: gzip, deflate, br
Accept: */*
Connection: keep-alive
Content-Length: 33

{"a": ["abc", "def", {"bec": 1}]}
```

**json**

```
POST /upload/ HTTP/1.1
Host: 127.0.0.1:8000
User-Agent: python-requests/2.31.0
Accept-Encoding: gzip, deflate, br
Accept: */*
Connection: keep-alive
Content-Length: 33
Content-Type: application/json

{"a": ["abc", "def", {"bec": 1}]}
```

**form**

```
POST /upload/ HTTP/1.1
Host: 127.0.0.1:8000
User-Agent: python-requests/2.31.0
Accept-Encoding: gzip, deflate, br
Accept: */*
Connection: keep-alive
Content-Length: 17
Content-Type: application/x-www-form-urlencoded

a=abc&a=def&a=bec
```

### files

注意: files 只能与表单类型数据共存

```python
files = [('file', open("a.txt", 'rb'))]
data = {"a": "1", "b": 2}
response = requests.post(
    url,
    files=files,  # header 会自动设置 {"Content-Type": "multipart/form-data ..."}
    data=data  # 使用了 files 时, data 只能传字典 (表单类型), 而不能是字符串
)
```

## requests

requests 封装了客户端 socket 的代码, `requests.get`, `requests.post` 等方法实际上都会最终回落到对 `requests.request` 的调用. 注意本质上, `requests.request` 方法只是将数据转化为符合 HTTP 协议的文本通过 socket 发送给服务端.

### requests.request

`requests.request` 的全部参数如下:

```python
import requests
resp = requests.request(
    method="GET",
    url="http://localhost:8000/a/b",
    params={"q": "1", "r": 23},    # 像这样传字典且不加 {"Content-Type": "application/json"} 的 header, 就会被处理成表单
    data={"body-param": "value"},  # 一般不会用于 GET 请求中
    # json={"a": 1, "b": 2},       # 如果同时给 data 和 json, 则会忽略 json 字段.
    headers={"User-Agent": "custom"},
    cookies={"message": "This is cookies", "session_id": "name"},
    files=[],
    auth=("api_key", "xxx"),
    timeout=2,
    allow_redirects=True,
    proxies={"HTTP_PROXY": "http://127.0.0.1", "HTTPS_PROXY": "http://127.0.0.1"},
    verify=True,
    # hooks=,
    stream=False,
    # cert= ,
)
```

注意:

- `params` 参数会被组合到 HTTP 请求的第一行里
- `headers`, `cookies`, `auth` 被整合至 HTTP 请求头中, 注意在本例中 `auth` 被使用了 base64 进行了编码: `import base64; base64.b64decode("YXBpX2tleTp4eHg=")==b'api_key:xxx'`
- `data` 参数可以是字典,字符串类型(也许还可以是其他类型), 在本例中是字典, 为此 requests 会自动加上 `Content-Type: application/x-www-form-urlencoded` 请求头
- `json` 参数必须是可以用 json 序列化的数据类型, 如果传入 json 参数, requests 会自动加上 `Content-Type: application/json` 参数. 如果即传入 json 参数又传入 data 参数, requests 会忽略 json 参数
- `files` 参数见后续章节专门介绍
- `timeout`, `allow_redirects`, `proxies` 是一些连接参数
- `verify`, `cert`, `hooks`: 未知
- `stream`: 此参数并不直接体现在 HTTP 请求内容里, 而是包装在 requests 这一层的实现, 因此能否做到流式返回主要取决于服务端是否是以流式返回的 (requests 客户端没法通过 stream 参数告知服务器, 但可以通过 `data` 或 `json` 等告知服务器)
    ```python
    # requests/sessions.py:Session
    # r: requests.models.Response
    if not stream:  # stream 参数来源于 requests.post(..., stream=stream)
        r.content   # 如果 stream=False, 即使服务端按流式返回, 客户端也会等所有数据发送完毕, 再返回给 requests.request 的调用者
    # 如果 stream=True, 则直接返回, 但是否为真流式则需要由服务端和请求参数确定
    # 在真流式的情形下, 客户端以怎样的方式流式读取数据, 则是由服务器和客户端之间的约定来决定的
    # 一种常见的约定形式是, 客户端使用 for line in resp.iter_lines() 的方式来流式获取数据
    return r
    ```

```
GET /a/b?q=1&r=23 HTTP/1.1
Host: localhost:8000
User-Agent: custom
Accept-Encoding: gzip, deflate, br
Accept: */*
Connection: keep-alive
Cookie: message=This is cookies; session_id=name
Content-Length: 16
Content-Type: application/x-www-form-urlencoded
Authorization: Basic YXBpX2tleTp4eHg=

body-param=value
```

### requests.get

一个 GET 请求:

```python
import requests
requests.get("http://localhost:8000/a/b?q=1&r=23")
# 或者也以把 params 直接写在 url 里
# requests.get("http://localhost:8000/a/b?q=1&r=23")
```

`socket_server.py` 的后台打印

```
GET /a/b?q=1&r=23 HTTP/1.1
Host: 127.0.0.1:8000
User-Agent: python-requests/2.31.0
Accept-Encoding: gzip, deflate, br
Accept: */*
Connection: keep-alive

```

其中 `/a/b` 是 path, `q=1&r=23` 是 query paramter, 按 HTTP/1.1 协议, GET 请求的 body 通常是空的, 但在使用 `requests.get` 时, 可以传递 `data` 或 `json` 参数, 这会导致请求体不为空

### requests.post

```python
import requests
import json
requests.post("http://localhost:8000/a/b?q=1&r=23", data=json.dumps({"a": "abc"}))
requests.post("http://localhost:8000/a/b?q=1&r=23", json={"a": "abc"})
```

使用 `data` 传递参数

```
POST /a/b?q=1&r=23 HTTP/1.1
Host: localhost:8000
User-Agent: python-requests/2.31.0
Accept-Encoding: gzip, deflate, br
Accept: */*
Connection: keep-alive
Content-Length: 12

{"a": "abc"}

```

使用 `json` 传递参数, 区别在于增加了一个 HTTP 请求头字段: `Content-Type: application/json`

```
POST /a/b?q=1&r=23 HTTP/1.1
Host: localhost:8000
User-Agent: python-requests/2.31.0
Accept-Encoding: gzip, deflate, br
Accept: */*
Connection: keep-alive
Content-Length: 12
Content-Type: application/json

{"a": "abc"}

```


## File

遵循 HTTP 协议的文件传输, 服务端用 socket, flask, fastapi 实现, 客户端用 socket, requests 实现. 另外计划附 streamlit 怎么接收上传的文件

### 服务端

- socket: TODO
- flask: TODO
- fastapi: 见另一篇关于 pydantic 和 fastapi 的博客

### 客户端

**例子**

```python

import requests
url = 'http://127.0.0.1:8000/upload/'

with open("a.txt", 'rb') as f:
    files = [('file', f)]  # "file" 不是固定的字段名
    response = requests.post(url, files=files, data={"a": "123"})
```

注意在这个情形下, `Content-Length=1450`, `content[-1450:] = "--ef0...."`

```
Server started at 127.0.0.1:8000
=====Connected by ('127.0.0.1', 48518)=======
--------------------
POST /upload/ HTTP/1.1
Host: 127.0.0.1:8000
User-Agent: python-requests/2.31.0
Accept-Encoding: gzip, deflate, br
Accept: */*
Connection: keep-alive
Content-Length: 1450
Content-Type: multipart/form-data; boundary=ef0479c6edb2827623ed87f943bafa08

--ef0479c6edb2827623ed87f943bafa08
Content-Disposition: form-data; name="a"

123
--ef0479c6edb2827623ed87f943bafa08
Content-Disposition: form-data; name="file"; filename="a.txt"

hello world hello world hello world hello worldhello worldhello worldhello worldhello worldhello worldhello worldhello world
hello half
hello world hello world hello world hello worldhello worldhello worldhello worldhello worldhello worldhello worldhello world
hello half
hello world hello world hello world hello worldhello worldhello worldhello worldhello worldhello worldhello worldhello world
hello half
hello world hello world hello world hello worldhello worldhello worldhello worldhello worldhello worldhello worldhello world
hello half
hello world hello world hello world h
--------------------
ello worldhello worldhello worldhello worldhello worldhello worldhello worldhello world
hello half
hello world hello world hello world hello worldhello worldhello worldhello worldhello worldhello worldhello worldhello world
hello half
hello world hello world hello world hello worldhello worldhello worldhello worldhello worldhello worldhello worldhello world
hello half
hello world hello world hello world hello worldhello worldhello worldhello worldhello worldhello worldhello worldhello world
hello half
hello world hello world hello world hello worldhello worldhello worldhello worldhello worldhello worldhello worldhello world
hello half

--ef0479c6edb2827623ed87f943bafa08--

```

`files` 参数的输入格式以下两种之一:

```python
files = [
    ("file1", open("a.txt")),
    ("file2", ("a.txt", open("a.txt"))),
    ("file3", ("a.txt", open("a.txt"), "docx")),
    ("file4", ("a.txt", open("a.txt"), "text", {"a": "b"})),
]

files = {
    "file1": open("a.txt"),
    "file2": ("a.txt", open("a.txt")),
    "file3": ("a.txt", open("a.txt"), "docx"),
    "file4": ("a.txt", open("a.txt"), "text", {"a": "b"}),
}
```


**原理**

**例1**

构造 body 内容的主要逻辑位于 `requests.models.RequestEncodingMixin._encode_files` 函数内

```python
import requests

# The tuples may be 2-tuples (filename, fileobj), 3-tuples (filename, fileobj, contentype) or 4-tuples (filename, fileobj, contentype, custom_headers).
body, content_type = requests.models.RequestEncodingMixin._encode_files(
    files = [
        # 以下四种方式均可
        ("filename", open("a.txt")),
        # ("filename", ("a.txt", open("a.txt"))),
        # ("filename", ("a.txt", open("a.txt"), "docx")),
        # ("filename", ("a.txt", open("a.txt"), "text", {"a": "b"})),
    ],
    # data=None,
    data={"c": 1, "d": 2},
)
# type(body): bytes
# type(content_type): str
# content_type: 'multipart/form-data; boundary=5ceb5926d95d0f9da655826454e770b7'
print(body.decode())
```

输出:

```
--5ceb5926d95d0f9da655826454e770b7
Content-Disposition: form-data; name="c"

1
--5ceb5926d95d0f9da655826454e770b7
Content-Disposition: form-data; name="d"

2
--5ceb5926d95d0f9da655826454e770b7
Content-Disposition: form-data; name="filename"; filename="a.txt"

文件内容

--5ceb5926d95d0f9da655826454e770b7--
```

**例2**

```python
import requests

# The tuples may be 2-tuples (filename, fileobj), 3-tuples (filename, fileobj, contentype) or 4-tuples (filename, fileobj, contentype, custom_headers).
body, content_type = requests.models.RequestEncodingMixin._encode_files(
    files = [
        ("file1", open("a.txt")),
        ("file2", ("a.txt", open("a.txt"))),
        ("file3", ("a.txt", open("a.txt"), "docx")),
        ("file4", ("a.txt", open("a.txt"), "text", {"a": "b"}))],
    data={"c": 1, "d": 2},
)
print(body.decode())
```

输出

```
--a87a3680eef38c2b584032cd35ac6c19
Content-Disposition: form-data; name="c"

1
--a87a3680eef38c2b584032cd35ac6c19
Content-Disposition: form-data; name="d"

2
--a87a3680eef38c2b584032cd35ac6c19
Content-Disposition: form-data; name="file1"; filename="a.txt"

文件内容

--a87a3680eef38c2b584032cd35ac6c19
Content-Disposition: form-data; name="file2"; filename="a.txt"

文件内容

--a87a3680eef38c2b584032cd35ac6c19
Content-Disposition: form-data; name="file3"; filename="a.txt"
Content-Type: docx

文件内容

--a87a3680eef38c2b584032cd35ac6c19
Content-Disposition: form-data; name="file4"; filename="a.txt"
Content-Type: text
a: b

文件内容

--a87a3680eef38c2b584032cd35ac6c19--

```

## Streaming

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
    # 'Accept': 'text/event-stream',  # 无需设置
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
    model="Qwen-72B-Chat-Int4",
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
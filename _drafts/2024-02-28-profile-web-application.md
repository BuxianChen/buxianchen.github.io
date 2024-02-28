---
layout: post
title: "(Alpha) profile web application"
date: 2024-02-28 10:05:04 +0800
labels: [web, profile]
---

**cProfile**

```python
import cProfile
from flask import Flask

app = Flask(__name__)
cProfile("app.run()", "profile_status")

# 分析
import pstats

p = pstats.Stats("profile_stats")
p.sort_stats("cumulative").print_stats(10)
```

**ProfilerMiddleware + snakeviz**

参考 [https://srinaveendesu.medium.com/profiling-your-web-application-120d1e2602de](https://srinaveendesu.medium.com/profiling-your-web-application-120d1e2602de)

**疑问**

如果使用 gunicorn 启动多进程/多线程怎么监控呢?
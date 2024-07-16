---
layout: post
title: "(P1) llm-compiler"
date: 2024-07-17 00:10:04 +0800
labels: [llm]
---

参考: [https://github.com/langchain-ai/langgraph/blob/main/examples/llm-compiler/LLMCompiler.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/llm-compiler/LLMCompiler.ipynb)

主要包括:

- planner
- task fetching unit
- joiner

```
planner(replaner) + task fetching unit  -> joiner        # 无条件边
joiner -> [END, planner(replaner) + task fetching unit]  # 有条件边
```

## Task Fetching Unit

备注: 这里用的是线程, 所以 `observations` 变量是共享的.

```python
from concurrent.futures import ThreadPoolExecutor, wait
import time
import re

tasks = [
    {
        "idx": 1,
        "tool_name": "sleep_next",
        "args": {"t": 1},
        "dependencies": []
    },
    {
        "idx": 2,
        "tool_name": "sleep_next",
        "args": {"t": 2},
        "dependencies": []
    },
    {
        "idx": 3,
        "tool_name": "sleep_total",
        "args": {"time1": "$1", "time2": "$2"},
        "dependencies": [1, 2]
    },
]

observations = {}

def sleep_next(t):
    time.sleep(t)
    return t

def sleep_total(time1=0, time2=0):
    return time1 + time2

functions = {
    "sleep_next": sleep_next,
    "sleep_total": sleep_total
}

def _resolve_arg(args, observations):
    new_args = {}
    for arg_name, arg_value in args.items():
        ID_PATTERN = r"\$\{?(\d+)\}?"
        if isinstance(arg_value, str):
            match = re.match(ID_PATTERN, arg_value)
            if match:
                idx = int(match.group(1))
                observation = observations.get(idx)
                new_args[arg_name] = observation
            else:   
                new_args[arg_name] = arg_value
        else:   
            new_args[arg_name] = arg_value
    return new_args

def schedule_task(task, observations):
    args = task["args"]
    resolved_args = _resolve_arg(args, observations)
    observation = functions[task["tool_name"]](**resolved_args)
    observations[task["idx"]] = observation
    print(f"Finish:\n{task}, {observations}")

def schedule_pending_task(task, observations, retry_after):
    while True:
        deps = task["dependencies"]
        if deps and (any([dep not in observations for dep in deps])):
            time.sleep(retry_after)
            continue
        schedule_task(task, observations)
        break

retry_after = 0.2
futures = []
with ThreadPoolExecutor() as executor:
    for task in tasks:
        
        if any([dep not in observations for dep in task["dependencies"]]):
            print(f"pending {task['idx']}")
            futures.append(executor.submit(schedule_pending_task, task, observations, retry_after))
        else:
            print(f"execute {task['idx']}")
            executor.submit(schedule_task, task, observations)
    wait(futures)
```
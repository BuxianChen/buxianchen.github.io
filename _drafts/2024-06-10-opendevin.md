---
layout: post
title: "(P1) Opendevin 项目探索"
date: 2024-06-10 15:10:04 +0800
labels: [project]
---

## 动机、参考资料、涉及内容

探索 opendevin 项目的各个方面, 包括但不限于: 前后端项目结构, Makefile, react 前端框架, 后端 agent llm 的逻辑

项目地址: [https://github.com/OpenDevin/OpenDevin](https://github.com/OpenDevin/OpenDevin)

由于此项目还远非稳定, 本文限定在 2024.06.05 的 0.6.2 版本

## 依赖安装与运行

项目的一级目录如下:

```
.
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Development.md  # 运行方式记录
├── LICENSE
├── Makefile     # 安装与运行
├── README.md
├── agenthub/    # 各个不同的 agent, 与 opendevin 目录的关系
├── cache/       # 运行时生成的目录
├── conf.toml
├── containers/  # Dockerfile
├── dev_config/  # precommit 相关
├── docs/        # 文档
├── evaluation/  # 未知
├── frontend/    # 前端代码, React 项目
├── logs/        # 运行时生成的目录
├── opendevin/   # 后端代码, 用 python + fastapi
├── poetry.lock
├── pydoc-markdown.yml
├── pyproject.toml
├── pytest.ini
├── tests/       # 单元测试?
└── workspace/   # 运行时生成的目录, agent 的工作区, 也就是它只能在这个目录新建文件
```

依赖安装与运行

参考 Development.md, 下面是拆开各个步骤的做法

这里仅记录一下笔者的操作. 此项目依赖是 Docker, nodejs, python3.11, poetry. 笔者使用的是 Windows11 的 WSL2:

(1) 在 WSL2 内已经预先安装好了 anaconda, nodejs=20.13.1
(2) 在 Windows 本机安装了 Docker-Desktop, docker 版本为 docker==26.1.1.

```bash
# (1) 安装 poetry
conda create --name pipx-py39 python=3.9
conda activate pipx-py39
pip install pipx
pipx install poetry

# (2) 创建环境, 克隆代码
conda create --name devin python=3.11
conda activate devin
git clone https://github.com/OpenDevin/OpenDevin && git checkout 0.6.2

# (3) 使用 poetry 安装后端服务的依赖项以及安装 opendevin
poetry install  # 很奇怪的是需要安装 torch, 初看可以去掉这个依赖 (因为笔者的电脑硬盘空间比较吃紧), 但还不知道怎么去除
pip install playwright
playwright install --with-deps chromium

# (4) 拉取 sandbox 镜像, 其实就只是一个普通的带 python 以及一些 python 包的镜像, 目的应该只是在项目跑起来是用挂载目录的方式来保证只修改 sandbox 镜像内和 workspace 目录的文件. 这样可以方式 agent 删除本地文件
make pull-docker-image  # 其实就是 docker pull ghcr.io/opendevin/sandbox

# (5) 安装前端依赖项
# i18n 其实就是前端界面上可以选择多种语言, detect-node-version.js 就是检查 nodejs >= 18.17.1
# frontend/package.json 中包含这一项配置:
# {"scripts": {"make-i18n": "node scripts/make-i18n-translations.cjs"}}
# cd frontend && node ./scripts/detect-node-version.js && npm install && npm run make-i18n
install-frontend-dependencies

# (6) 将前端代码使用 tsc 编译, 并进行 build
# frontend/package.json 中包含这一项配置:
# {"scripts": {"build": "tsc && vite build"}}
make build-frontend     # 其实就是 frontend && npm run build

# 配置后端服务
make setup-config    # 命令行交互填写配置文件 config.toml
# config.toml 类似如下:
# [core]
# workspace_base="./workspace"

# [llm]
# model="gpt-4o"
# api_key="sk-xxx"
# embedding_model="openai"

# (7) 启动 python + fastapi 后端, 占用 3000 端口
# 其实也就是
# poetry run uvicorn opendevin.server.listen:app --port 3000
make start-backend

# (8) 启动前端, 占用 3001 端口
# frontend/package.json 中包含这一项配置:
# {"scripts": {"start": "npm run make-i18n && vite"}}
# 其实也就是
# cd frontend && VITE_BACKEND_HOST=3000 VITE_FRONTEND_PORT=3001 npm run start
make start-frontend
```

## 前端探索

## 后端探索

llm 方面使用 litellm 做接入

```python
from opendevin.runtime.browser.browser_env import BrowserEnv
env = BrowserEnv(is_async=False)
env.step(action_str="goto('https://zh.wikipedia.org/wiki/%E9%97%AD%E9%9B%86')")
```

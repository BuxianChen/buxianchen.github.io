---
layout: post
title: "(LST) Python 环境管理"
date: 2025-03-18 10:05:04 +0800
labels: [python]
---

## 动机、参考资料、涉及内容

本篇博客主要涉及 python 环境管理相关的内容

对于 Linux 系统, 默认的 `PATH` 环境变量一般包括如下优先级顺序的 bin 目录:

- `/usr/local/sbin:/usr/local/bin`: 其他系统软件的默认安装路径
- `/usr/sbin:/usr/bin`: 系统核心工具, 一般由 apt 等系统包管理工具来处理, 因此需要避免在此目录下自行安装内容
- `/sbin:/bin`: 系统基础工具, 例如 ls, cp 等

python 的安装方式主要有这几类:

- 系统自带: 一般在 `/usr/bin` 目录下
- uv: 安装在 `.local/share/uv/python` 下
- pyenv: 安装在`~/.pyenv/shims`
- conda: 

其中: uv 号称代替 `pip`, `pip-tools`, `pipx`, `poetry`, `pyenv`, `twine`, `virtualenv`

## 注意事项

pipx 会共用 ~/.local/bin 目录, 可能会导致版本冲突, pipx 本身也需要基于一个已有的 python 环境才能安装, 因此应该使用一个系统级的 python 环境来安装 pipx, 以及在系统级 python 的环境下使用 pipx install, 不要在虚拟或conda环境中安装或使用 pipx

conda 环境应该与 uv, pyenv 完全隔离开, 不要混用

## uv

uv 的使用方式是(TODO)

```bash
# uv 的安装, 默认安装的二进制文件是 ~/.local/bin/uv 和 ~/.local/bin/uvx
curl -LsSf https://astral.sh/uv/install.sh | sh

# 首先安装 python
uv python install 3.10
# 可以显示所有被 uv 安装和管理的 python 的路径, 也能发现系统级安装的 python 的路径
# uv python list

# 然后创建虚拟环境
cd /path/to/project_dir
# 创建一个 .venv 目录, 其中是一个 python 3.10 的虚拟环境
uv venv -p 3.10
```

uv 管理的 python 在如下目录

```
~/.local/share/uv/
  - python/  # 各个版本的 python
    - cpython-3.11.11-linux-x86_64-gnu/
      - bin/
      - include/
      - lib/
      - share/
```

## pyenv

pyenv 与 conda 共存的方案如下:

```bash
# ~/.bashrc
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"  # 先加载 pyenv 命令行工具
eval "$(pyenv init --path)"

# conda 初始化
CONDA_PATH="$HOME/anaconda3"  # 根据实际安装路径修改
__conda_setup="$('$CONDA_PATH/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
        . "$CONDA_PATH/etc/profile.d/conda.sh"
    fi
fi
unset __conda_setup
# 建议配置: 不要自动激活 base 环境
# 另外也建议不将 ~/anaconda3/bin 加入PATH: ~anaconda3/bin 目录下会包含 conda, 但它也包含 python, 会造成污染
conda config --set auto_activate_base false

export PATH="$HOME/.local/bin:$PYENV_ROOT/shims:$PATH"
# export PATH="$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
```
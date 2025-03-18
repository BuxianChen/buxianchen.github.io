---
layout: post
title: "(LST) Python 环境管理"
date: 2025-03-18 10:05:04 +0800
labels: [python]
---

## 动机、参考资料、涉及内容

python 环境管理相关的内容

对于 Linux 系统, 默认的 `PATH` 环境变量一般包括如下优先级顺序的 bin 目录:

- `/usr/local/sbin:/usr/local/bin`: 其他系统软件的默认安装路径
- `/usr/sbin:/usr/bin`: 系统核心工具, 一般由 apt 等系统包管理工具来处理, 因此需要避免在此目录下自行安装内容
- `/sbin:/bin`: 系统基础工具, 例如 ls, cp 等


python 的安装方式主要有这几类:

- 系统自带: 一般在 `/usr/bin` 目录下
- pyenv: `~/.pyenv/shims`
- conda

其中系统自带和 pyenv 均可视为系统级安装的 python, conda 则是另外独立的内容. 建议按这种方式设置 PATH 环境变量

```bash
# TODO: 需确认
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"  # 先加载 pyenv 命令行工具

# 初始化 pyenv 版本管理
eval "$(pyenv init --path)"

# conda 初始化（不直接修改 PATH
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

# 禁用自动激活 base 环境
conda config --set auto_activate_base false

export PATH="$HOME/.local/bin:$PYENV_ROOT/shims:$PATH"
export PATH="$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
```
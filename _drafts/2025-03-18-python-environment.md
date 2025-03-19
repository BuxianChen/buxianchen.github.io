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

## pyenv (OK)

pyenv 只负责安装和切换 python 版本, 其本身不支持创建虚拟环境的命令(可以切换 python 版本后通过 venv 命令来创建, 或者借助 `pyenv-virtualenv` 插件更方便)

pyenv 与 conda 共存的方案如下:

```bash
# ~/.bashrc
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"  # 先加载 pyenv 命令行工具
eval "$(pyenv init --path)"

# conda 初始化, 此步骤会将 ~/anaconda3/condabin 目录加入环境变量, 此目录仅包含 conda 可执行文件
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
# 另外也建议不要将 ~/anaconda3/bin 加入PATH: ~/anaconda3/bin 目录下会包含 conda, 但它也包含 python, 会造成污染
conda config --set auto_activate_base false

export PATH="$HOME/.local/bin:$PYENV_ROOT/shims:$PATH"
# export PATH="$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
```

**安装**

```bash
curl -fsSL https://pyenv.run | bash
```

**配置**

```bash
# ~/.bashrc 中添加以下内容:
export PYENV_ROOT="$HOME/.pyenv"

# 将 ~/.pyenv/bin 目录添加至 PATH, 此目录底下仅包含 pyenv 这个命令本身
# 备注: ~/.pyenv/bin/pyenv 是对 ~/.pyenv/libexec/pyenv 的软链接
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"

# 这一行会将 ~/.pyenv/shims 加入至 PATH 环境变量
# 在后续执行了 pyenv install 3.10; pyenv install 3.11 之后
# 此目录底下会包含 python, python3.10, python3.11, pip, pip3.10, pip3.11 等 shell 脚本
eval "$(pyenv init - bash)"

# 最终的 PATH 变量会是
# ~/.pyenv/shims:~/.pyenv/bin
```

注意: 像 `~/.pyenv/shims/python`, `~/.pyenv/shims/pip3.10` 这种文件实际上都是 shell 脚本, 而不是对 pyenv 所安装的 `python` 二进制可执行文件的软链接, 并且这些 shell 脚本文件内容都相同, 如下:

```
[buxian@~/.pyenv/shims] (master) $ ls
2to3       idle3.10  pip3.11    python          python3.10-config
2to3-3.10  idle3.11  pydoc      python-config   python3.10-gdb.py
2to3-3.11  pip       pydoc3     python3         python3.11
idle       pip3      pydoc3.10  python3-config  python3.11-config
idle3      pip3.10   pydoc3.11  python3.10      python3.11-gdb.py
```

文件内容如下:

```bash
#!/usr/bin/env bash
set -e
[ -n "$PYENV_DEBUG" ] && set -x

program="${0##*/}"

export PYENV_ROOT="/home/buxian/.pyenv"
exec "/home/buxian/.pyenv/libexec/pyenv" exec "$program" "$@"
```

因此, 本质上是通过 PATH 搜索路径的优先级, 对 `python ...` 命令进行拦截, 最终执行的是 `~/.pyenv/libexec/pyenv python ...` (与 `~/.pyenv/bin/pyenv` 是软链接关系), 备注: `shim` 英文原意是薄垫片, 在这里相当于是用户命令 `python` 与实际的 `/path/to/python` 之间的这一层.

**使用**

```bash
# 下载 python 安装包, 并安装在 ~/.pyenv/versions/3.10.16 目录下
pyenv install 3.10
pyenv install 3.11
```

切换 python 版本

```bash
# 将 ~/.pyenv/version 文件内容修改为 3.10
pyenv global 3.10
# 查看所有由 pyenv 管理的 python 以及当前 python
pyenv versions
#   system
# * 3.10.16 (set by /home/buxian/.pyenv/version)
#   3.11.8

# 在当前目录下创建一个 .python_version 文件, 文件内容为 3.11, 之后在此目录及子目录下, 输入的 python 命令将会 pyenv 拦截并解释为 python 3.11.8
pyenv local 3.11
pyenv versions
#   system
#   3.10.16
# * 3.11.8 (set by /home/buxian/wsl2_test/xx/.python-version)

# 修改环境变量 PYENV_VERSION 的值为 3.10, 此设置仅对本 shell 生效 
pyenv shell 3.10
pyenv versions
#   system
# * 3.10.16 (set by PYENV_VERSION environment variable)
#   3.11.8

# 优先级: shell > local > global
```

因此, 复原方式如下:

```bash
pyenv global system
rm .python_verison
unset PYENV_VERSION
```

管理和切换 python 版本是 pyenv 的核心功能, pyenv 没有创建虚拟环境的命令. 但可以辅助创建虚拟环境

方案一:

```bash
pyenv local 3.10.6         # 指定 Python 版本
python -m venv .venv       # 创建虚拟环境
source .venv/bin/activate  # 手动激活
```

方案二 (使用 `pyenv-virtualenv` 插件):

```bash
# 安装的具体细节不做展开

# 使用
pyenv virtualenv 3.10.6 myenv  # 一步创建版本+环境, 虚拟环境也在 ~/.pyenv/versions 目录下, 体验类似于 virtualenvwrapper
pyenv activate myenv           # 统一命令管理
```

## uv

**安装**

```bash
# uv 的安装, 默认安装的二进制文件是 ~/.local/bin/uv 和 ~/.local/bin/uvx
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**使用**

安装 python

```bash
# 首先安装 python
uv python install 3.12
```

显示可以安装/已安装的 python

```bash
# 可以显示所有被 uv 安装和管理的 python 的路径, 也能发现系统级安装的 python 的路径
uv python list
```

输出如下:

```
cpython-3.14.0a5+freethreaded-linux-x86_64-gnu    <download available>
cpython-3.14.0a5-linux-x86_64-gnu                 <download available>
cpython-3.13.2+freethreaded-linux-x86_64-gnu      <download available>
cpython-3.13.2-linux-x86_64-gnu                   <download available>
cpython-3.12.9-linux-x86_64-gnu                   /home/buxian/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/bin/python3.12
cpython-3.11.11-linux-x86_64-gnu                  <download available>
cpython-3.10.16-linux-x86_64-gnu                  /home/buxian/.pyenv/versions/3.10.16/bin/python3.10
cpython-3.10.16-linux-x86_64-gnu                  /home/buxian/.pyenv/versions/3.10.16/bin/python3 -> python3.10
cpython-3.10.16-linux-x86_64-gnu                  /home/buxian/.pyenv/versions/3.10.16/bin/python -> python3.10
cpython-3.10.16-linux-x86_64-gnu                  <download available>
cpython-3.9.21-linux-x86_64-gnu                   <download available>
cpython-3.8.20-linux-x86_64-gnu                   <download available>
cpython-3.8.10-linux-x86_64-gnu                   /usr/bin/python3.8
cpython-3.8.10-linux-x86_64-gnu                   /usr/bin/python3 -> python3.8
cpython-3.8.10-linux-x86_64-gnu                   /bin/python3.8
cpython-3.8.10-linux-x86_64-gnu                   /bin/python3 -> python3.8
cpython-3.7.9-linux-x86_64-gnu                    <download available>
pypy-3.11.11-linux-x86_64-gnu                     <download available>
pypy-3.10.16-linux-x86_64-gnu                     <download available>
pypy-3.9.19-linux-x86_64-gnu                      <download available>
pypy-3.8.16-linux-x86_64-gnu                      <download available>
pypy-3.7.13-linux-x86_64-gnu                      <download available>
```

备注: 在我的机器上, 我安装了如下 python

- 系统级别的 python, 由 apt 管理, 也就是上面显示的 `/usr/bin/python3.8`
- pyenv 管理的多个 python, 我按照上一节 pyenv 与 conda 共存的 bashrc 配置进行了配置, 并且使用了 `pyenv global 3.10` 进行了全局设置, uv 也能够找到: `/home/buxian/.pyenv/versions/3.10.16/bin/python3.10`
- uv 管理的 python, 使用 `uv python install 3.12` 安装的: `/home/buxian/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/bin/python3.12`
- conda 管理的 python, 没有体现(可能因为按我的配置, conda 安装的 python 不在 PATH 里)

创建项目目录和虚拟环境

```bash
# 然后创建虚拟环境
mkdir project_dir
cd project_dir
uv init

# 创建一个 .venv 目录, 使用 python 3.12 的虚拟环境 (这里存疑, 如果像我上面那种比较混乱的情形, uv, pyenv, 系统python都有的时候, 怎么确定用哪个 python)
uv venv -p 3.12
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
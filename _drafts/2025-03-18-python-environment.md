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

**概述**

uv 号称一个工具支持这些能力: pip, pip-tools, pipx, poetry, pyenv, twine, virtualenv, 具体的对应关系:

- `pip` 和 `poetry`: 在项目层面主要通过 `uv add`, `uv remove` 来实现同样的功能(严格地说,算是取代 poetry 的功能: 安装或删除相关的依赖, 并更新依赖配置文件), 在项目的虚拟环境层面主要通过 `uv pip` 来实现同样的功能.
- `pip-tools`: `uv pip compile` 和 `uv pip sync` 用于替代 `pip-tools` 用于实现锁定依赖版本至依赖配置文件(requirements.txt等), 以及使用依赖配置文件同步虚拟环境 (严格根据requirements.txt来安装或删除虚拟环境中的依赖包)
- `pipx`: 主要是通过 `uv tool ...` 来实现同样的功能
- `pyenv`: 主要是通过 `uv python ...` 来实现同样的功能
- `wheel`, `twine`: `wheel` 用于打包, `twine` 用于发布至 PyPI, 主要通过 `uv build` 和 `uv publish` 来实现同样的功能
- `virtualenv`: 主要是通过 `uv venv` 或者 `uv run` 自动触发来实现同样的功能.

备注: 实际上 `poetry` 其实也包含了以上除了 `pyenv` 外几乎所有的功能.

uv 主要包含这几块功能与相应的命令

- **python versions (安装,管理,切换 python)**: `uv python install/list/find/pin/uninstall`
- **scripts**: `uv run` (运行脚本), `uv add --script` (为脚本增加依赖), `uv remove --script` (为脚本减少依赖)
  - 注意: 这里的为脚本增加/减少依赖指的是增加 [inline metadata](https://packaging.python.org/en/latest/specifications/inline-script-metadata/) 而非在项目的虚拟环境里安装相应的 python 依赖包, 也不会对 `pyproject.toml` 做修改
- **projects (项目管理)**: 这个是最主要的功能, 主要可以替代 poetry: `uv init`, `uv add`, `uv remove`, `uv sync`, `uv lock`, `uv run`, `uv tree`, `uv build`, `uv publish`
  - 注意: 在同一个项目内不可与 poetry 混用, 但
- **tools (工具)**: 主要用于替代 pipx, `uvx`/`uv tool run`, `uv tool install`, `uv tool uninstall`, `uv tool list`, `uv tool update-shell`
  - 注意: `uv tool` 不要与 pipx 混用(同一个用户不可混用), pipx 对工具 (例如 black, ruff 等) 都安装在 ~/.local/bin 目录下
- **pip interface**: `uv venv`, `uv pip install/uninstall/list/show/tree/check/freeze/compile/sync`
  - 切记不要使用激活虚拟环境 `source .venv/bin/activate` + `pip install`, 而要使用 `uv pip install`, 因为使用 `uv venv` 得到的虚拟环境的 bin 目录 `.venv/bin` 底下不包含 pip 可执行脚本, 因此激活虚拟环境不能使用任何与 pip 有关的内容
  - `uv add xxx` 与 `uv pip install xxx` 的区别是前者会将依赖写入 pyproject.toml, 而后者不会修改. 因此 uv 的 pip interface 适合临时试验.
- **utility**: `uv cache clean/prune`, `uv cache dir`, `uv tool dir`, `uv python dir`, `uv self update`


**使用**

**recipe 1**

使用 uv 安装合适的 python, 并且为项目创建独立的虚拟环境, 并安装项目的依赖以及依赖版本锁定

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

创建项目目录和虚拟环境

```bash
# 然后创建虚拟环境
mkdir project_dir
cd project_dir
uv init

# 创建一个 .venv 目录, 使用 python 3.12 的虚拟环境 (这里存疑, 如果像我上面那种比较混乱的情形, uv, pyenv, 系统 python 都有的时候, 怎么确定用哪个 python)
uv venv -p 3.12
```

添加项目依赖

```bash
uv add requests
```

以上命令会将 requests 包安装在项目的虚拟环境中


**recipe 2**

uv 还可以用来对单个可执行 python 脚本实现环境隔离. **但不知道有什么实用场景**(为每个单独的python脚本都构建一个独立的虚拟环境)

首先编写如下代码

```python
# example.py
import requests
from rich.pretty import pprint

resp = requests.get("https://peps.python.org/api/peps.json")
data = resp.json()
pprint([(k, v["title"]) for k, v in data.items()][:10])
```

然后执行:

```bash
# 也可以最开始就用 uv init 初始化
# uv init --script example.py --python 3.12

# 添加 inline metadata
uv add --script example.py 'requests<3' 'rich' --python 3.12

# --no-project 在使用了 inline metadata 时是隐含参数, 可以不加
# uv run 命令默认是在项目中使用, 而项目一般是会有独立的虚拟环境, 而这里是希望使用脚本的虚拟环境来跑脚本, 因此应该
uv run --no-project example.py

# 也可以不使用上面的 uv add --script 命令, 此处 --no-project 似乎也可以不加
# 但这样的话就必须运行时指定: uv run --no-project --with 'requests<3,rich' --python 3.12 example.py

# 疑问: --no-project 什么时候是必须加的?
```

其本质是为 `example.py` 建立了一个独立的虚拟环境, 目录如下:

```bash
uv cache dir
# 结果通常是 ~/.cache/uv
```

此目录的结构为

```
.
├── CACHEDIR.TAG
├── archive-v0
│   ├── 3GcNENOQ2bwNqiLqhYRMv  # 此目录内部为一个特定的python包
│   ├── 4dFqn_orFfpLxsa5FL0Y2
│   ├── 89KxbYhXVjvZqQvH6ijeK
│   ├── B2-Hl9K-JqHkI-EF2wtnY
│   ├── J2Tk1VZYfrGygoUmnjNkB
│   ├── KjYBbEwa51XaoMPaLKQp0
│   ├── TgfDzS-bKnD1TQuO2hFwN
│   ├── V3DHmT0kTIO1nSHNLWK5M
│   ├── W2WPPNbgxG_Het1ZY1Grj
│   ├── WmsssfbxPd7ezeQcS01nk
│   ├── Z_dchrNtp87GvZQvDuVko
│   ├── fNHbvnMjGurNQ4hF5-QPI
│   ├── foy4DxzhRuCq4zz2_7v-R
│   ├── gl8KZA_-Orf5KfX7nBLcN
│   ├── gnwqyFXCC9Rb7Gw5o6MN8
│   ├── hmqP_cSR2FCsNXJVqqMwE
│   ├── qcdzz9RyHD1AKB5AMilG5
│   └── wfMOCRMkHwTD9_4cdHIA9
├── builds-v0
├── environments-v2
│   ├── 18e5760217f69a9e
│   ├── example-f2c1da13bd0d8822  # example.py 的虚拟环境
│   └── x-680771d4393799f2
├── interpreter-v4
│   └── 746acfdb0dba2ea6
├── sdists-v8
├── simple-v15
│   └── pypi
└── wheels-v5
    └── pypi
```

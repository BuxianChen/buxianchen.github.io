---
layout: post
title: "(Alpha) Python 包管理与CI/CD开发工具"
date: 2024-04-16 10:05:04 +0800
labels: [python, package manager]
---

## 动机、参考资料、涉及内容

**动机**

怎么写一个包, Github 开源项目贡献, CI/CD 等

**涉及内容**

- setup.py, pip, pipx, poetry
- requirements.txt, project.toml
- mypy, pre-commit, isort, black, ruff, flake
- python -m, sys.path, importlib
- CI, CD

**参考资料**

- [https://www.jumpingrivers.com/blog/python-package-managers-pip-conda-poetry/](https://www.jumpingrivers.com/blog/python-package-managers-pip-conda-poetry/)
- [https://www.anaconda.com/blog/using-pip-in-a-conda-environment](https://www.anaconda.com/blog/using-pip-in-a-conda-environment): 在 conda 环境中使用 pip 的注意事项

## pip & pipx

pipx 到底怎么装? 官方推荐系统级别安装

```
sudo apt update
sudo apt install pipx  # /usr/bin/pipx
pipx ensurepath  # 就是把 ~/.local/bin 加入 ~/.bashrc 的 PATH 变量设置中
sudo pipx ensurepath --global  # Optional
```

pipx 用于安装有 entrypoint 的 pypi 包, 使用 pipx 安装 pypi 包时, 它总会为 pypi 包新建一个虚拟环境.


目录结构

```
~/.local/bin/poetry
~/.local/share/
  - vitrualenv/
    - py_info/
      - 1/5f1c06f3099f26449885087d7735eafe4f7373fbfdcd0799979c26ddac9bdb6b.json
  - pipx/  # 见下面
```

使用 `cd ~/.local/share/pipx && tree -L 5` 打印目录 `~/.local/share/pipx` 的目录结构:

```
├── py
│   └── CACHEDIR.TAG
├── shared
│   ├── bin
│   │   ├── Activate.ps1
│   │   ├── activate
│   │   ├── activate.csh
│   │   ├── activate.fish
│   │   ├── pip                  #!/home/buxian/.local/share/pipx/shared/bin/python
│   │   ├── pip3                 #!/home/buxian/.local/share/pipx/shared/bin/python
│   │   ├── pip3.10              #!/home/buxian/.local/share/pipx/shared/bin/python
│   │   ├── pip3.9               #!/home/buxian/.local/share/pipx/shared/bin/python
│   │   ├── python -> /home/buxian/anaconda3/envs/exp/bin/python
│   │   ├── python3 -> python
│   │   └── python3.9 -> python
│   ├── include
│   ├── lib
│   │   └── python3.9
│   │       └── site-packages
│   │           ├── _distutils_hack
│   │           ├── distutils-precedence.pth
│   │           ├── pip
│   │           ├── pip-24.0.dist-info
│   │           ├── pkg_resources
│   │           ├── setuptools
│   │           └── setuptools-58.1.0.dist-info
│   ├── lib64 -> lib
│   └── pyvenv.cfg
└── venvs
    └── poetry
        ├── bin
        │   ├── Activate.ps1
        │   ├── activate
        │   ├── activate.csh
        │   ├── activate.fish
        │   ├── doesitcache
        │   ├── dul-receive-pack
        │   ├── dul-upload-pack
        │   ├── dulwich
        │   ├── keyring
        │   ├── normalizer
        │   ├── pkginfo
        │   ├── poetry
        │   ├── pyproject-build
        │   ├── python -> /home/buxian/anaconda3/envs/exp/bin/python
        │   ├── python3 -> python
        │   ├── python3.10 -> python
        │   ├── python3.9 -> python
        │   └── virtualenv
        ├── include
        ├── lib
        │   ├── python3.10
        │   │   └── site-packages
        │   └── python3.9
        │       └── site-packages
        ├── lib64 -> lib
        ├── pipx_metadata.json
        └── pyvenv.cfg
```

## 附录

### `__main__.py` 和 `__init__.py`

### `ensurepip`

如果遇到特殊情况, 可以按照如下命令安装 pip. ([参考](https://www.jumpingrivers.com/blog/python-package-managers-pip-conda-poetry/))

```bash
python -m ensurepip --upgrade
```

具体的执行逻辑(仅做示意,不同Python版本的实现可能不同)大致是: 首先在 `/path/to/ensurepip` 底下有目录结构 (可以通过 `import ensurepip; print(ensurepip.__file__)` 找到这个路径):

```
├── __init__.py
├── __main__.py
├── _bundled
│   ├── __init__.py
│   ├── pip-23.0.1-py3-none-any.whl
│   └── setuptools-65.5.0-py3-none-any.whl
└── _uninstall.py
```

而 `python -m ensurepip` 的逻辑是将这里的两个 `pip*.whl` 和 `setuptools*.whl` 文件保存到一个临时目录, 然后按以下方式安装:

```python
# 将 *.whl 复制进 tmpdir 内 
additional_paths = 
args = ["install", "--no-cache-dir", "--no-index", "--find-links", tmpdir] + 
    # ["--root", root]  # 安装目录 一般不会指定
    # ["--upgrade"]  # 如果指定 --upgrade 的话
    # ["--user"]  # 如果指定 --user 的话
    # ["-vvv"]   # 可以指定 -v 1, -v 2, -v 3, 最多到 3
code = f"""
import runpy
import sys
sys.path = {additional_paths or []} + sys.path
sys.argv[1:] = {args}
runpy.run_module("pip", run_name="__main__", alter_sys=True)
"""

# runpy.run_module 等价于 python -m
# 有趣的是可以在没有安装pip的时候, 但pip*.whl位于--find-links目录时, 就可以 python -m pip install ...

cmd = [
    sys.executable,  # 这个实际上就是当前 python 的绝对路径
    '-W',
    'ignore::DeprecationWarning',
    '-c',
    code,
]
if sys.flags.isolated:
    # run code in isolated mode if currently running isolated
    cmd.insert(1, '-I')
subprocess.run(cmd, check=True)
```
---
layout: post
title: "(P0) Python 包管理与CI/CD开发工具"
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

- [https://packaging.python.org/en/latest/](https://packaging.python.org/en/latest/): 官方资料, pypa 项目底下包含 pip, setuptools, wheel, twine, virtualenv, pipx 等的源码
- [https://www.jumpingrivers.com/blog/python-package-managers-pip-conda-poetry/](https://www.jumpingrivers.com/blog/python-package-managers-pip-conda-poetry/)
- [https://www.anaconda.com/blog/using-pip-in-a-conda-environment](https://www.anaconda.com/blog/using-pip-in-a-conda-environment): 在 conda 环境中使用 pip 的注意事项

## 打包: 历史与最佳实践

备注: 尽量使用 `python -m pip install xxx` 而非 `pip install xxx`

术语释义 (python 3.8 文档, 术语解释并不过时):
- [https://docs.python.org/3.8/distributing/index.html](https://docs.python.org/3.8/distributing/index.html)
- [https://docs.python.org/3.8/installing/index.html#key-terms](https://docs.python.org/3.8/installing/index.html#key-terms)

要点如下:

- pip: 装包的最底层工具之一, pip 本身不依赖于 setuptools, 但很多包在安装时会需要依赖 setuptools
- setuptools: 最原始的工具是 distutils, setuptools 目前的实现也依赖于 distutils, setuptools 是非官方的事实标准, 在未来可能会成为标准库的一部分
- egg: 已经弃用, 现在都是使用 whl 格式
- `easy_install`: `easy_install` 是作为 setuptools 的一部分在 2004 年发布的, 现在已经完全弃用
- hatch, pdm, poetry: 基本上逻辑都是配置文件只写一个 `pyproject.toml`, 切换起来不困难, 只需要修改 toml 文件即可, 功能类似

Overview:

发布格式大体分为两类: 源码发布(Source Distribution, 简称 sdist, 也就是 `.tar.gz` 格式)与二进制格式发布 (binary distributions, 也称为 Wheels), 例如 [pip==23.3.1](https://pypi.org/project/pip/23.3.1/#files) 就包含两种发布格式: `pip-23.3.1.tar.gz` 和 `pip-23.3.1-py3-none-any.whl`. 最佳实践发布源码格式以及一个或多个 whl 格式.

源码格式发布通常是checkout某个提交, 然后加上元信息文件 `PKG-INFO`, 以 `pip-23.3.1.tar.gz` 为例, 解压后文件目录与原始代码库的提交差距不大, 如下:

```
src/
  - pip/
  - pip.egg-info/  # 这个文件夹应该不是标准的做法, egg 已弃用
    - PKG-INFO
    - dependency_links.txt
    - not-zip-safe
    - SOURCES.txt
    - entry_points.txt
    - top_level.txt
PKG-INFO   # 这个文件的内容和 src/pip.egg-info/PKG-INFO 完全一致
...        # 其他文件都出现在原始代码库的相应提交里, 但原代码库里的一些文件例如 .pre-commit-config.yaml 文件不包含在 .tar.gz 文件内, 为什么会这样待研究, 猜测是和 pip 代码库本身的 CI/CD 工具设置有关
```

而二进制发布基本上等价于用户在安装时需要复制的所有文件, 对于一个包的一个特定版本, PyPI 规定只能发布一个源码包, 但可以包含多个二进制包 (可以参考 [opencv-python==4.8.1.78](https://pypi.org/project/opencv-python/4.8.1.78/#files)). 对于像这种包含 C 代码的项目, whl 文件里通常不包含 C 代码, 而只包含预编译好的 `.so` 文件. 而对于像 `pip` 这类纯 python 包, 其 whl 文件内只包含这种目录结构 (whl 文件实际上可以用 unzip 解压):

```
- pip/
- pip-23.3.1.dist-info/  # 注意这个和 sdist 里的文件夹名不一样, 内容也不太一样, 但都是文本文件
  - AUTHORS.txt
  - entry_points.txt
  - LICENSE.txt
  - METADATA # 这个文件等价于 sdist 中的 PKG-INFO 文件
  - RECORD   # 记录了 pip/ 文件夹中所有文件的哈希值
  - top_level.txt
  - WHEEL
```

这里先暂且不深入这两种格式的发布过程 Github -> .tar.gz/.whl -> PyPI. 我们先看使用者的视角, 使用者安装的过程通常是由 `pip install` 发起的, 这个过程大体上是:

1. 先去尝试下载匹配的 `.whl` 文件, 然后基本上就是直接将 `.whl` 文件解压然后丢到 `site-packages` 文件夹下, 以上面的 `pip==23.3.1` 为例, 就是直接在 `site-package` 文件夹下增加了 `pip` 和 `pip.egg-info` 文件夹.
2. 如果找不到匹配的 `.whl` 文件, 则下载源码格式发布的文件, 然后在本地将其打包为 `.whl` 格式, 然后将 `.whl` 格式文件进行安装

而本文的重点在于发布过程: Github -> CI/CD -> .tar.gz/.whl -> PyPI 或 Local Source Code -> .tar.gz/.whl -> PyPI

[Is `setup.py` deprecated?](https://packaging.python.org/en/latest/discussions/setup-py-deprecated/), setuptools (包含 easy_install) 以及 setup.py 没有被弃用, 只是不要使用命令行用法, 例如 `python setup.py install`. setuptools 搭配 `setup.py` 仍然可以用于 build backend.

## pipx

### TL;DR

pipx 主要用于安装有 entrypoint 的 pypi 包, 使用 pipx 安装 pypi 包时, 它总会为 pypi 包新建一个虚拟环境. pipx 的作用类似于 apt, npm, 换句话说, pipx 将带有 entrypoint 的 pypi 包转换为了一个命令行命令, 由于 pipx 是为每个命令安装了独立的虚拟环境, 因此不会出现命令 A 需要依赖包 C==1.2, 而命令 B 需要依赖包 C==1.3, 造成安装时的意外升级与使用时的不一致.

> In a way, it turns Python Package Index (PyPI) into a big app store for Python applications.

[引用自](https://pipx.pypa.io/stable/#where-does-pipx-install-apps-from)

对比 pip install, 假设带有 entrypoint 的包 A, B 分别需要安装 C==1.2 和 C==1.3, 这样安装可能会造成不易察觉的问题

```bash
pip install A  # 假设 A 的 entrypoint 里包含 Aapp, 实际的例子: pip install torch 时会带有 torchrun 的 entrypoint
pip install B  # 假设 B 的 entrypoint 里包含 Bapp
pip list
# A 1.0.0
# B 1.0.0
# C 1.3
Aapp run ...  # 可能会出问题
Bapp run ...
```

pipx 到底怎么装? 官方推荐系统级别安装

```
sudo apt update
sudo apt install pipx  # /usr/bin/pipx
pipx ensurepath  # 修改 ~/.bashrc, ~/.bash_profile 等配置文件内容, 把 ~/.local/bin 添加进 PATH 变量
sudo pipx ensurepath --global  # Optional
```

用法

```
pipx install ipython
pipx reinstall ipython
pipx uninstall ipython
pipx ensurepath
pipx upgrade ipython
pipx inject ipython numpy  # 在 ipython 这个命令所在的虚拟环境中用 pip 安装 numpy
pipx run ipython
```

### pipx 探幽

pipx 源码并不复杂, 主要是利用了 python 自带的 `venv` 和 `pip` 包, 执行逻辑也可以参考 [https://pipx.pypa.io/stable/how-pipx-works/](https://pipx.pypa.io/stable/how-pipx-works/), 也可以在执行 `pipx` 命令时加上 `--verbose` 选项.

#### 目录结构

备注: pipx 使用的目录结构需要参考[官方文档](https://pipx.pypa.io/stable/installation/#global-installation), 这里只是一个 pipx 版本的实现


```python
import platformdirs
platformdirs.user_data_dir()/pipx/trash   # ~/.local/share/pipx/trash
platformdirs.user_data_dir()/pipx/shared  # ~/.local/share/pipx/shared
platformdirs.user_data_dir()/pipx/venvs   # ~/.local/share/pipx/venv
platformdirs.user_cache_dir()/pipx        # ~/.cache/pipx
platformdirs.user_log_dir()/pipx/log      # ~/.local/state/log/pipx/log
```

具体目录结构

```
~/.local/
  - bin/
    - poetry   # pipx 安装的可执行脚本
  - share/
    - pipx/  # ... 见下面
    - vitualenv/   # 这个似乎与 pipx 无关
```

使用 `cd ~/.local/share/pipx && tree -L 5` 打印目录 `~/.local/share/pipx` 的目录结构:

```bash
conda create --name langchain python=3.10
conda activate langchain
pip install pipx
pipx install poetry
pip list | grep peotry  # 找不到 peotry, 因为 peotry 是在独立的虚拟环境装的
pipx ensurepath
# poetry config virtualenvs.prefer-active-python true  # 这个与 pipx 无关, 是 poetry 的配置
cd ~/.local/share/pipx && tree -L 5
```

`~/.local/share/pipx` 的目录结构如下

```
├── py
│   └── CACHEDIR.TAG
├── shared                  # 看起来似乎是用这里的 python 和 pip 为每个命令建的虚拟环境
│   ├── bin
│   │   ├── Activate.ps1
│   │   ├── activate
│   │   ├── activate.csh
│   │   ├── activate.fish
│   │   ├── pip             #!/home/buxian/.local/share/pipx/shared/bin/python
│   │   ├── pip3            #!/home/buxian/.local/share/pipx/shared/bin/python
│   │   ├── pip3.10         #!/home/buxian/.local/share/pipx/shared/bin/python
│   │   ├── python -> /home/buxian/anaconda3/envs/langchain/bin/python
│   │   ├── python3 -> python
│   │   └── python3.10 -> python
│   ├── include
│   ├── lib
│   │   └── python3.10
│   │       └── site-packages
│   │           ├── _distutils_hack
│   │           ├── distutils-precedence.pth
│   │           ├── pip
│   │           ├── pip-24.0.dist-info
│   │           ├── pkg_resources
│   │           ├── setuptools
│   │           └── setuptools-65.5.0.dist-info
│   ├── lib64 -> lib
│   └── pyvenv.cfg
└── venvs                  # 每个命令都是一个虚拟环境
    └── poetry
        ├── bin
        │   ├── Activate.ps1
        │   ├── activate
        │   ├── activate.csh
        │   ├── activate.fish
        │   ├── doesitcache
        │   ├── dul-receive-pack
        │   ├── dul-upload-pack
        │   ├── dulwich
        │   ├── keyring
        │   ├── normalizer
        │   ├── pkginfo
        │   ├── poetry
        │   ├── pyproject-build
        │   ├── python -> /home/buxian/anaconda3/envs/langchain/bin/python
        │   ├── python3 -> python
        │   ├── python3.10 -> python
        │   └── virtualenv
        ├── include
        ├── lib
        │   └── python3.10
        │       └── site-packages
        ├── lib64 -> lib
        ├── pipx_metadata.json
        └── pyvenv.cfg
```

#### `pipx ensurepath`

`pipx ensurepath` 的本质基本上就是 (`pipx/commands/ensure_path.py:ensure_path`):

```python
import userpath
location_str = "~/.local/bin"
path_added = userpath.append(location_str, "pipx")
```

`userpath` 是一个 python 内置包, 会修改 shell configuration file, 例如: `~/.bashrc`, `~/.bash_profile` 等, 执行完后, 会添加类似这种内容:

```
# Created by `pipx` on 2024-04-12 07:34:33
export PATH="$PATH:/home/buxian/.local/bin"
```

其具体执行逻辑可以参考这个:

```python
# userpath/interface.py:UnixInterface.put
for shell in self.shells:
    for file, contents in shell.config(location, front=front).items():
        try:
            if os.path.exists(file):
                with open(file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            else:
                lines = []

            if any(contents in line for line in lines):
                continue

            lines.append(
                u'\n{} Created by `{}` on {}\n'.format(
                    shell.comment_starter, app_name, datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                )
            )
            lines.append(u'{}\n'.format(contents))

            with open(file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        except:
            continue
```

如果想知道到底修改了哪些 shell configuration file, 可以用这个办法验证:

```python
from userpath.interface import Interface
interface = Interface(shells=None, all_shells=None, home=None)
for shell in interface.shells:
    for file, contents in shell.config("/home/buxian/.local/bin", front=True).items():
        print(file)
        print(contents)
```

输出:

```
/home/buxian/.bashrc
export PATH="/home/buxian/.local/bin:$PATH"
/home/buxian/.bash_profile
export PATH="/home/buxian/.local/bin:$PATH"
```

#### `pipx list`

`pipx list` 的输出也可以作为探索目录结构的参考

```bash
pipx list
```

输出

```
venvs are in /home/buxian/.local/share/pipx/venvs
apps are exposed on your $PATH at /home/buxian/.local/bin
manual pages are exposed at /home/buxian/.local/share/man
   package poetry 1.8.2, installed using Python 3.10.14
    - poetry
```

具体实现逻辑也很简单, 本质上就是 `os.listdir("/home/buxian/.local/share/pipx/venvs")`

```python
# pipx/commands/list_packages:list_text
def list_text(venv_dirs: Collection[Path], include_injected: bool, venv_root_dir: str) -> VenvProblems:
    print(f"venvs are in {bold(venv_root_dir)}")
    print(f"apps are exposed on your $PATH at {bold(str(paths.ctx.bin_dir))}")
    print(f"manual pages are exposed at {bold(str(paths.ctx.man_dir))}")

    all_venv_problems = VenvProblems()
    for venv_dir in venv_dirs:
        # venv_dir = pathlib.PosixPath("/home/buxian/.local/share/pipx/venvs/poetry")
        # 疑问: include_injected 为 True 和 False 是什么意思
        package_summary, venv_problems = get_venv_summary(venv_dir, include_injected=include_injected)
        if venv_problems.any_():
            logger.warning(package_summary)
        else:
            print(package_summary)
        all_venv_problems.or_(venv_problems)

    return all_venv_problems
```

#### `pipx install`

`pipx install` 主要使用了:

- 虚拟环境创建: `/path/to/python -m venv ...`
- pip 安装必要的依赖包: `/path/to/python -m pip install ...`

具体可以参考这些源码

```python
# pipx/venv.py
# Venv.create_venv
# Venv.install_package

def create_venv(self, venv_args: List[str], pip_args: List[str], override_shared: bool = False) -> None:
    """
    override_shared -- Override installing shared libraries to the pipx shared directory (default False)
    """
    with animate("creating virtual environment", self.do_animation):
        cmd = [self.python, "-m", "venv"]
        if not override_shared:
            cmd.append("--without-pip")
        venv_process = run_subprocess(cmd + venv_args + [str(self.root)], run_dir=str(self.root))
    subprocess_post_check(venv_process)

    shared_libs.create(verbose=self.verbose, pip_args=pip_args)
    if not override_shared:
        pipx_pth = get_site_packages(self.python_path) / PIPX_SHARED_PTH
        # write path pointing to the shared libs site-packages directory
        # example pipx_pth location:
        #   ~/.local/share/pipx/venvs/black/lib/python3.8/site-packages/pipx_shared.pth
        # example shared_libs.site_packages location:
        #   ~/.local/share/pipx/shared/lib/python3.6/site-packages
        #
        # https://docs.python.org/3/library/site.html
        # A path configuration file is a file whose name has the form 'name.pth'.
        # its contents are additional items (one per line) to be added to sys.path
        pipx_pth.write_text(f"{shared_libs.site_packages}\n", encoding="utf-8")

    self.pipx_metadata.venv_args = venv_args
    self.pipx_metadata.python_version = self.get_python_version()
    source_interpreter = shutil.which(self.python)
    if source_interpreter:
        self.pipx_metadata.source_interpreter = Path(source_interpreter)

def install_package(
    self,
    package_name: str,
    package_or_url: str,
    pip_args: List[str],
    include_dependencies: bool,
    include_apps: bool,
    is_main_package: bool,
    suffix: str = "",
) -> None:
    # package_name in package specifier can mismatch URL due to user error
    package_or_url = fix_package_name(package_or_url, package_name)

    # check syntax and clean up spec and pip_args
    (package_or_url, pip_args) = parse_specifier_for_install(package_or_url, pip_args)

    with animate(
        f"installing {full_package_description(package_name, package_or_url)}",
        self.do_animation,
    ):
        # do not use -q with `pip install` so subprocess_post_check_pip_errors
        #   has more information to analyze in case of failure.
        cmd = [
            str(self.python_path),
            "-m",
            "pip",
            "--no-input",
            "install",
            *pip_args,
            package_or_url,
        ]
        # no logging because any errors will be specially logged by
        #   subprocess_post_check_handle_pip_error()
        pip_process = run_subprocess(cmd, log_stdout=False, log_stderr=False, run_dir=str(self.root))
    subprocess_post_check_handle_pip_error(pip_process)
    if pip_process.returncode:
        raise PipxError(f"Error installing {full_package_description(package_name, package_or_url)}.")

    self._update_package_metadata(
        package_name=package_name,
        package_or_url=package_or_url,
        pip_args=pip_args,
        include_dependencies=include_dependencies,
        include_apps=include_apps,
        is_main_package=is_main_package,
        suffix=suffix,
    )

    # Verify package installed ok
    if self.package_metadata[package_name].package_version is None:
        raise PipxError(
            f"Unable to install "
            f"{full_package_description(package_name, package_or_url)}.\n"
            f"Check the name or spec for errors, and verify that it can "
            f"be installed with pip.",
            wrap_message=False,
        )
```

#### `pipx run`

`pipx run` 是创建一个虚拟环境在 cache 目录, 并运行这个虚拟环境里的 entrypoint, 例如: `pipx run ipython`, 会在 `~/.cache/pipx/0e4f05d9aae40dd` 目录下安装虚拟环境, 但不会将可执行脚本 `ipython` 的软链接放在 `~/.local/bin` 下, 而是位于原始的 `~/.cache/pipx/0e4f05d9aae40dd/bin/` 目录下.

```python
from pipx import paths
print(paths.ctx.venv_cache)  # ~/.cache/pipx
```

## pre-commit

### TL;DR

安装

```bash
pip install pre-commit
```

用法

```bash
conda create --name precommit python=3.10
pip install pre-commit
git init
pre-commit install
git add .pre-commit-config.yaml  # 此例参考 https://github.com/open-mmlab/mmdeploy/blob/4bb9bc738c9008055fbc9347f46da70ee60fdad3/.pre-commit-config.yaml
git commit -m "add pre-commit config"
git add a.py
git commit -m "add a.py"   # 初次提交时会缓存 repo, 注意这些 repo 是直接 git clone 到缓存目录的, 而不是 pip install 到当前环境
```

执行逻辑

pre-commit 依赖于 virtualenv, git, 在 `git commit` 时, 对于类型为 python 的 hook (代码仓库的 `.pre-commit-hooks.yaml` 里会写明), 如果 `repo` 写的是一个 github 地址而不是 `local` 的话, 那么会为每个 `repo` 用 `virtualenv` 建立虚拟环境, 执行 hook 时会使用虚拟环境进行

### pre-commit 探幽

本节以这个 `.pre-commit-config.yaml` 为例进行探索

```yaml
repos:
  - repo: https://github.com/PyCQA/flake8  # 注意如果将 github 仓库地址改为 local, 那么就不会缓存至目录并创建独立的虚拟环境
    rev: 4.0.1
    hooks:
      - id: flake8
        args: ["--exclude=*/client/inference_pb2.py, \
                */client/inference_pb2_grpc.py, \
                tools/package_tools/packaging/setup.py"]
  - repo: https://github.com/PyCQA/isort
    rev: 5.11.5
    hooks:
      - id: isort
  - repo: https://github.com/pre-commit/mirrors-yapf
    rev: v0.32.0
    hooks:
      - id: yapf
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.2.0
    hooks:
      - id: trailing-whitespace
      - id: check-yaml
      - id: end-of-file-fixer
      - id: requirements-txt-fixer
      - id: double-quote-string-fixer
      - id: check-merge-conflict
      - id: fix-encoding-pragma
        args: ["--remove"]
      - id: mixed-line-ending
        args: ["--fix=lf"]
  - repo: https://github.com/executablebooks/mdformat
    rev: 0.7.9
    hooks:
      - id: mdformat
        args: ["--number", "--disable-escape", "link-enclosure"]
        additional_dependencies:
          - mdformat-openmmlab
          - mdformat_frontmatter
          - linkify-it-py
  - repo: https://github.com/codespell-project/codespell
    rev: v2.1.0
    hooks:
      - id: codespell
        args: ["--skip=third_party/*,*.ipynb,*.proto"]

  - repo: https://github.com/myint/docformatter
    rev: v1.4
    hooks:
      - id: docformatter
        args: ["--in-place", "--wrap-descriptions", "79"]
```

#### `pre-commit install`

执行 `pre-commit install` 的本质是写入 `.git/hooks/pre-commit` 文件, 文件内容类似如下:

```bash
#!/usr/bin/env bash
# File generated by pre-commit: https://pre-commit.com
# ID: 138fd403232d2ddd5efb44317e38bf03

# start templated
INSTALL_PYTHON=/home/buxian/anaconda3/envs/precommit/bin/python
ARGS=(hook-impl --config=.pre-commit-config.yaml --hook-type=pre-commit)
# end templated

HERE="$(cd "$(dirname "$0")" && pwd)"
ARGS+=(--hook-dir "$HERE" -- "$@")

if [ -x "$INSTALL_PYTHON" ]; then
    exec "$INSTALL_PYTHON" -mpre_commit "${ARGS[@]}"
elif command -v pre-commit > /dev/null; then
    exec pre-commit "${ARGS[@]}"
else
    echo '`pre-commit` not found.  Did you forget to activate your virtualenv?' 1>&2
    exit 1
fi
```

具体的关键逻辑如下:

```python
# pre_commit/commands/install_uninstall.py
def resource_text(filename: str) -> str:
    files = importlib.resources.files('pre_commit.resources')
    return files.joinpath(filename).read_text()

# pre_commit/resources/hook-impl 文件中包含上面 .git/hooks/pre-commit 中的内容模板
# 实际写入时会替换掉 `# start templated` 和 `# end templated` 之间的内容
# 替换逻辑详见: _install_hook_script
def _install_hook_script(...):
    ...
```

#### `git commit` & pre-commit 缓存文件夹

第一次执行 `git commit -m "add a.py"` 时的输出为:

```
[INFO] Initializing environment for https://github.com/PyCQA/flake8.
[INFO] Initializing environment for https://github.com/PyCQA/isort.
[INFO] Initializing environment for https://github.com/pre-commit/mirrors-yapf.
[INFO] Initializing environment for https://github.com/pre-commit/pre-commit-hooks.
[INFO] Initializing environment for https://github.com/executablebooks/mdformat.
[INFO] Initializing environment for https://github.com/executablebooks/mdformat:mdformat-openmmlab,mdformat_frontmatter,linkify-it-py.
[INFO] Initializing environment for https://github.com/codespell-project/codespell.
[INFO] Initializing environment for https://github.com/myint/docformatter.
[INFO] Installing environment for https://github.com/PyCQA/flake8.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
[INFO] Installing environment for https://github.com/PyCQA/isort.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
[INFO] Installing environment for https://github.com/pre-commit/mirrors-yapf.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
[INFO] Installing environment for https://github.com/pre-commit/pre-commit-hooks.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
[INFO] Installing environment for https://github.com/executablebooks/mdformat.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
[INFO] Installing environment for https://github.com/codespell-project/codespell.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
[INFO] Installing environment for https://github.com/myint/docformatter.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
flake8...................................................................Failed
- hook id: flake8
- exit code: 1

t.py:3:1: E302 expected 2 blank lines, found 1
t.py:6:1: E305 expected 2 blank lines after class or function definition, found 1
t.py:9:1: E402 module level import not at top of file
t.py:18:6: W292 no newline at end of file

isort....................................................................Failed
- hook id: isort
- files were modified by this hook

Fixing /home/buxian/wsl2_test/test_code/test_precommit/t.py

yapf.....................................................................Failed
- hook id: yapf
- files were modified by this hook
trim trailing whitespace.................................................Passed
check yaml...........................................(no files to check)Skipped
fix end of files.........................................................Passed
fix requirements.txt.................................(no files to check)Skipped
fix double quoted strings................................................Failed
- hook id: double-quote-string-fixer
- exit code: 1
- files were modified by this hook

Fixing strings in t.py

check for merge conflicts................................................Passed
fix python encoding pragma...............................................Passed
mixed line ending........................................................Passed
mdformat.............................................(no files to check)Skipped
codespell................................................................Passed
docformatter.............................................................Passed
```

首先注意观察类似这种输出:

```
[INFO] Installing environment for https://github.com/PyCQA/flake8.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
```

这个过程做的是将依赖的 repo 使用 git clone 至缓存文件夹 (默认位于 `~/.cache/pre-commit`), 这里文件夹名是通过 `tempfile.mkdtemp` 生成的

```python
# pre_commit/store.py:_get_default_directory
import os
def _get_default_directory() -> str:
    ret = os.environ.get('PRE_COMMIT_HOME') or os.path.join(
        os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'),
        'pre-commit',
    )
    return os.path.realpath(ret)

# pre_commit/store.py:Store._new_repo
# 默认情况下: self.directory = _get_default_directory()
directory = tempfile.mkdtemp(prefix='repo', dir=self.directory)
```

缓存文件夹目录如下:

```
.
├── README
├── .lock            # 估计只是一个 filelock 文件
├── db.db            # sqlite3 数据库文件
├── repo3ryn4y_0/    # 对应 https://github.com/codespell-project/codespell
├── repoch_61y63/    # 对应 https://github.com/PyCQA/isort
├── repohkjm0j33/    # 对应 https://github.com/pre-commit/mirrors-yapf
├── repomk6u13hi/    # 对应 https://github.com/pre-commit/pre-commit-hooks
├── reporpoqbcsp/    # 对应 https://github.com/executablebooks/mdformat     # commit-id: 0c86e45
├── repots3lczbh/    # 对应 https://github.com/PyCQA/flake8
├── repoxea00f_0/    # 对应 https://github.com/executablebooks/mdformat     # commit-id: 99099d5
└── repoylmsb8fs/    # 对应 https://github.com/myint/docformatter
```

可以打印出 `db.db` 的 Schema 及数据信息, 如下:

```
==================================================
Schema for table: repos
repo TEXT, ref TEXT, path TEXT, 

Data for table: repos
('https://github.com/PyCQA/flake8', '4.0.1', '/home/buxian/.cache/pre-commit/repots3lczbh')
('https://github.com/PyCQA/isort', '5.11.5', '/home/buxian/.cache/pre-commit/repoch_61y63')
('https://github.com/pre-commit/mirrors-yapf', 'v0.32.0', '/home/buxian/.cache/pre-commit/repohkjm0j33')
('https://github.com/pre-commit/pre-commit-hooks', 'v4.2.0', '/home/buxian/.cache/pre-commit/repomk6u13hi')
('https://github.com/executablebooks/mdformat', '0.7.9', '/home/buxian/.cache/pre-commit/reporpoqbcsp')
('https://github.com/executablebooks/mdformat:mdformat-openmmlab,mdformat_frontmatter,linkify-it-py', '0.7.9', '/home/buxian/.cache/pre-commit/repoxea00f_0')
('https://github.com/codespell-project/codespell', 'v2.1.0', '/home/buxian/.cache/pre-commit/repo3ryn4y_0')
('https://github.com/myint/docformatter', 'v1.4', '/home/buxian/.cache/pre-commit/repoylmsb8fs')


==================================================
Schema for table: configs
path TEXT, 

Data for table: configs
('/home/buxian/wsl2_test/test_code/test_precommit/.pre-commit-config.yaml',)
```

接下来再看 git 的 pre-commit hook 实际执行的内容 (`.git/hooks/pre-commit`):

```bash
python -mpre-commit hook-impl --config=.pre-commit-config.yaml --hook-type=pre-commit --hook-dir /home/buxian/wsl2_test/test_code/test_precommit/.git/hooks --
```

此处跳过一些细节, 本质上执行的是 `pre_commit/languages/python.py` 的相关内容 (此例中所有的 hook 的 language 都是 python)

```python
# pre_commit/commands/run.py:_run_single_hook
language = languages[hook.language]
with language.in_env(hook.prefix, hook.language_version):
    retcode, out = language.run_hook(
        hook.prefix,
        hook.entry,
        hook.args,
        filenames,
        is_local=hook.src == 'local',
        require_serial=hook.require_serial,
        color=use_color,
    )
```

`hook.language` 是由 repo 的 `.pre-commit-hooks.yaml` 决定的, 例如: `https://github.com/PyCQA/flake8/.pre-commit-hooks.yaml` 文件内容是

```yaml
-   id: flake8
    name: flake8
    description: '`flake8` is a command-line utility for enforcing style consistency across Python projects.'
    entry: flake8
    language: python
    types: [python]
    require_serial: true
```

执行的实际方式是先创建虚拟环境(如果没有创建的话), 然后使用这个虚拟环境运行 hook

```python
# STEP 1: pre_commit/languages/python.py:install_environment
# 对于 language=python 类型的 hook, 首先在 /home/buxian/.cache/pre-commit/repots3lczbh 底下用 virtualenv 安装虚拟环境, 例如安装在
# /home/buxian/.cache/pre-commit/repots3lczbh/py_env-python3.10

envdir = lang_base.environment_dir(prefix, ENVIRONMENT_DIR, version)
venv_cmd = [sys.executable, '-mvirtualenv', envdir]
python = norm_version(version)
if python is not None:
    venv_cmd.extend(('-p', python))
install_cmd = ('python', '-mpip', 'install', '.', *additional_dependencies)
proc = subprocess.Popen(install_cmd, **kwargs)

# STEP 2: pre_commit/languages/python.py:in_env, run_hook
# 添加 PATH 环境变量 (通过 contextmanager 来实现), 然后执行
os.environ["PATH"] = "/home/buxian/.cache/pre-commit/repoylmsb8fs/py_env-python3.10/bin" + ":" + os.environ["PATH"]
cmd = ['flake8', '--exclude=*/client/inference_pb2.py, */client/inference_pb2_grpc.py, tools/package_tools/packaging/setup.py']
subprocess.Popen(cmd, **kwargs)
```

## poetry

### TL;DR

- `poetry.lock` 文件不应该被 ignore, 而应该交由 git 管理.
- poetry 现在的 installer 貌似已经不依赖于 pip 了. [blog](https://python-poetry.org/blog/announcing-poetry-1.4.0/).
- poetry 可以用于包含 C++ 代码的项目, 但官方文档似乎没有过多介绍
- 可以在 poetry 命令里加上 `-vvv` 选项, 观察其行为, 例如: `poetry update -vvv`, `poetry config --list -vvv`

```
poetry new --src 
```


## 附录

### `__main__.py` 和 `__init__.py`

### `ensurepip`

TL;DR: ensurepip 是 python 自带的包, 包里面附带了一个 setuptools 和 pip 的 whl 文件, 如果不小心把 pip 包损坏了 (例如升级 pip 时, 把原始的 pip 卸载了, 但是安装新 pip 时又出现权限问题; 或者 debug 时手动乱改了 pip 包的源文件), 可以通过 ensurepip 利用它自带的 whl 文件恢复一个较低版本的 pip, 然后再进行 pip 升级即可.

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

### PyPI JSON API

pip, poetry 等工具在运行时会使用到 PyPI JSON API, 用于解析依赖 (但是这个 API 里似乎没有依赖包的信息, API 返回的信息很大程度上是代码打包时, 如果是二进制打包, 是 METADATA 文件内容, 或者是源码打包, PKG-INFO 文件内容)? 然而并不是所有 PyPI 的包写的 metadata 信息都完善, 因此有些时候对于 pip 或 poetry 来说确认依赖关系只能先下载下来再做验证

- pip 相关的代码似乎在 `pip/_vendor/locations.py:PyPIJSONLocator`
- poetry 的一个 FAQ: [https://python-poetry.org/docs/faq/](https://python-poetry.org/docs/faq/), poetry 会缓存尝试过的包的 metadata 信息, 位于 `~/.cache/pypoetry` 目录下, 可以自行探索

API 参考文档: [https://warehouse.pypa.io/api-reference/json.html](https://warehouse.pypa.io/api-reference/json.html)

```
GET /pypi/<project_name>/json
GET /pypi/<project_name>/<version>/json

https://pypi.org/pypi/pip/json
https://pypi.org/pypi/pip/23.3.1/json
```
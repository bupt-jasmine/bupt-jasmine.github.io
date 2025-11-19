# 开发环境指南

## 开发环境指南

目标：在 Docker 容器中进行开发，使用 VS Code 连接容器，构建系统采用 Bazel，通过 Bazel 生成 compile_commands.json，使用 clangd 提供代码补全并。

---

## 1. Docker 环境（示例）
Dockerfile 示例：
```Dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8

RUN apt-get update &&\
    apt-get install -y lsb-release wget software-properties-common gnupg git git-lfs zip vim

# 安装 clang
ARG CLANG_VERSION=20
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh ${CLANG_VERSION} all && rm llvm.sh

# Bazel
ARG BAZELLIST_VERSION=1.26.0
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v${BAZELLIST_VERSION}/bazelisk-linux-amd64 && \
    mv bazelisk-linux-amd64 /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel
ARG BUILDIFIER_VERSION=8.2.0
RUN wget https://github.com/bazelbuild/buildtools/releases/download/v${BUILDIFIER_VERSION}/buildifier-linux-amd64 &&\
    mv buildifier-linux-amd64 /usr/local/bin/buildifier &&\
    chmod +x /usr/local/bin/buildifier
```

构建并运行容器（示例）：
- docker build -t my-dev-env .
- docker run --rm -it -v /path/to/project:/workspace -p 8000:8000 my-dev-env

---

## 2. VS Code 连接到 Docker
推荐使用 Remote - Containers 扩展。


---

## 3. VS Code 插件（推荐）
- clangd: llvm-vs-code-extensions.vscode-clangd
- Python: ms-python.python
- Bazel: bazelbuild.vscode-bazel

---

## 4. 使用 Bazel 生成 compile_commands.json
根目录下 MODULE.bazel 加上
```
# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
bazel_dep(name = "hedron_compile_commands", dev_dependency = True)
git_override(
    module_name = "hedron_compile_commands",
    remote = "https://github.com/hedronvision/bazel-compile-commands-extractor.git",
    commit = "abb61a688167623088f8768cc9264798df6a9d10",
    # Replace the commit hash (above) with the latest (https://github.com/hedronvision/bazel-compile-commands-extractor/commits/main).
    # Even better, set up Renovate and let it do the work for you (see "Suggestion: Updates" in the README).
)
```

执行: `bazel run @hedron_compile_commands//:refresh_all`

---

## 5. 常用 Bazel 命令
- 构建：bazel build //... 或 bazel build //path/to:target
- 测试：bazel test //...
- 运行：bazel run //path/to:target
- 清理：bazel clean
- 增量构建（保持缓存）：bazel build --config=opt //...
- 查看依赖：bazel query 'deps(//path/to:target)' --noimplicit_deps

---

## 6. 代码格式化与相关命令
- clang-format（手动格式化所有源码）：
  - clang-format -i $(git ls-files '*.c' '*.cpp' '*.cc' '*.h' '*.hpp')

---

## 7. 小贴士
- 在容器里把用户 UID/GID 与宿主对齐以避免权限问题。
- 把 compile_commands.json 加入 .gitignore（通常不提交）。
- 使用 devcontainer.json 可以把开发依赖固化，团队一致性更好。

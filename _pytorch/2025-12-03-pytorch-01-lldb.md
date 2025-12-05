---
title: "PyTorch源码解析系列 - 第01章：使用 LLDB 调试 PyTorch"
collection: pytorch
permalink: /pytorch/2025-12-03-pytorch-01-lldb
excerpt: '介绍如何使用 LLDB 调试 PyTorch 源码'
date: 2025-12-03
---

在前面的笔记中，已经介绍了如何从源码编译和安装 PyTorch。同时也简单介绍了一些 CPython 的基础知识。本笔记将介绍如何使用 LLDB 调试 PyTorch 源码，以便更好地理解其内部实现。

## python 脚本

```python
import torch

a = torch.Tensor([1.0, 2.0, 3.0], requires_grad=True)
b = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

c = a + b

print(c)
print(c.grad_fn)
```

将上述代码保存为 `torch_demo.py`。

## 使用 LLDB 调试

在终端中，使用 LLDB 运行 Python 脚本：

```bash
lldb -- $(which python) torch_demo.py
```

在 LLDB 中，设置断点 `PyInit__C`，这是 PyTorch C++ 扩展模块的初始化函数：

```lldb
(torch) jasmine@mac codes % lldb -- $(which python) torch_demo.py
(lldb) target create "/Users/jasmine/miniconda3/envs/torch/bin/python"
Current executable set to '/Users/jasmine/miniconda3/envs/torch/bin/python' (arm64).
(lldb) settings set -- target.run-args  "torch_demo.py"
(lldb) breakpoint set -n "PyInit__C"
Breakpoint 1: no locations (pending).
WARNING:  Unable to resolve breakpoint to any actual locations.
(lldb) run
Process 65819 launched: '/Users/jasmine/miniconda3/envs/torch/bin/python' (arm64)
1 location added to breakpoint 1
Process 65819 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
    frame #0: 0x0000000100d6ff6c _C.cpython-311-darwin.so`PyInit__C at stub.c:14:10
   11
   12   PyMODINIT_FUNC PyInit__C(void)
   13   {
-> 14     return initModule();
   15   }
Target 0: (python) stopped.
(lldb) step
Process 65819 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = step in
    frame #0: 0x0000000106955064 libtorch_python.dylib`::initModule() at Module.cpp:1798:3
   1795 extern "C" TORCH_PYTHON_API PyObject* initModule();
   1796 // separate decl and defn for msvc error C2491
   1797 PyObject* initModule() {
-> 1798   HANDLE_TH_ERRORS
   1799
   1800   c10::initLogging();
   1801   c10::set_terminate_handler();
Target 0: (python) stopped.
(lldb)
```

这样就可以在 LLDB 中调试 PyTorch 了。可以使用 LLDB 的各种命令来单步执行代码、查看变量值等。

后续可以根据需要设置更多断点，深入理解 PyTorch 的实现细节，也可以一步步通过 `step` 和  `next` 等调试命令梳理清楚 `PyTorch` 加载 C++ 实现部分是如何实现的。


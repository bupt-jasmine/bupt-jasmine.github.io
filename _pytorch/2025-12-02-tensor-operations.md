---
title: "PyTorch 基础：张量操作"
collection: pytorch
permalink: /pytorch/tensor-operations
excerpt: 'PyTorch 张量的基本操作和使用方法'
date: 2025-12-02
---

## 创建张量

```python
import torch

# 创建张量
x = torch.tensor([1, 2, 3])
y = torch.zeros(3, 3)
z = torch.randn(2, 3)
```

## 张量运算

```python
# 加法
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a + b

# 矩阵乘法
x = torch.randn(2, 3)
y = torch.randn(3, 4)
z = torch.mm(x, y)
```

## GPU 加速

```python
# 检查 CUDA 是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = x.to(device)
```

---

**笔记日期**: 2025-12-02

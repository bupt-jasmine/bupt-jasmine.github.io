---
title: "C++ 基础：指针和引用"
collection: cpp
permalink: /cpp/pointers-and-references
excerpt: '这是一篇关于 C++ 指针和引用的学习笔记'
date: 2025-12-02
---

## 指针

指针是一个变量，其值为另一个变量的地址。

```cpp
int x = 10;
int* ptr = &x;  // ptr 指向 x 的地址
```

## 引用

引用是已存在变量的别名。

```cpp
int x = 10;
int& ref = x;  // ref 是 x 的引用
```

## 指针 vs 引用

- 引用必须在声明时初始化，指针可以先声明后初始化
- 引用一旦绑定就不能改变，指针可以指向不同的变量
- 引用不能为空，指针可以为 nullptr

---

**笔记日期**: 2025-12-02

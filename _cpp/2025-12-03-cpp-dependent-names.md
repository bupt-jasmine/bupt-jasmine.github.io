---
title: "浅谈 C++ 模版中的待决名"
collection: cpp
permalink: /cpp/cpp-dependent-names
excerpt: '介绍 C++ 模版编程中待决名的概念及其用法'
date: 2025-12-03
---

最近在阅读 pytorch 的 C++ 源码时，遇到了一些模版代码中使用了“待决名”（dependent name）的概念。为了更好地理解这些代码，特地查阅了一些资料，学习并记录一下相关内容。

---

## 什么是待决名？

在 C++ 模版编程中，待决名是指那些依赖于模版参数的名称。由于模版参数在编译时才会被具体化，因此编译器在第一次解析模版定义时无法确定这些名称的具体含义。

---

## 待决名的 typename 使用

在模版的声明和定义中，不是当前实例化的成员且依赖于模版参数的名称不会被编译器视为类型，除非使用 `typename` 关键字进行显式声明。

在 C++ 中，当一个名称依赖于模版参数时，编译器无法确定它是类型还是非类型。为了告诉编译器该名称是一个类型，需要使用关键字 `typename`。

例如：
```cpp
template <typename T>
T::value_type getValue(const T& container) {
    return container.value();
}

struct Value {
    using value_type = int;
    int value() const { return 42; }
}
```

这里编译器不知道 `T::value_type` 是一个类型。为了明确告诉编译器它是一个类型，我们需要使用 `typename`：
```cpp
template <typename T>
typename T::value_type getValue(const T& container) {
    return container.value();
}
```

`T::value_type` 不是当前实例化的成员，当前实例化的成员是 `getValue` 函数本身。同时 `T::value_type` 依赖于模版参数 `T`，因此不会被编译器视为类型，必须使用 `typename` 进行显式声明。

这个规则在 《C++ Primer》 书中有提到过，这个还是比较常见的用法。

---

## 待决名的 template 使用

这个规则在日常工作中不太常见，在阅读 pytorch 代码时遇到过类似的用法，所以学习并记录一下。

模版定义中不是当前实例化的成员且依赖于模版参数的模版名称不会被编译器视为模版，除非使用 `template` 关键字进行显式声明。

在 C++ 中，当一个模版名称依赖于模版参数时，编译器无法确定它是一个模版还是非模版。为了告诉编译器该名称是一个模版，需要使用关键字 `template`。

例如：
```cpp
template <typename T>
struct Wrapper {
    template <typename U>
    void func();
};

template <typename T>
void test() {
    Wrapper<T> w;
    w.func<T>(); // 编译器无法确定 func 是不是一个模版

    w.template func<T>(); // 使用 template 关键字明确告诉编译器它是一个模版
}
```

`w.func` 不是当前实例化的成员，当前实例化的成员是 `test` 函数本身。同时 `w.func` 依赖于模版参数 `T`，因此不会被编译器视为模版，必须使用 `template` 进行显式声明。

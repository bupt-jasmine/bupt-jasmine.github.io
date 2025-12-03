---
title: "浅谈 C++ 17 折叠表达式"
collection: cpp
permalink: /cpp/cpp-fold-expressions
excerpt: '介绍 C++ 17 中引入的折叠表达式及其用法'
date: 2025-12-03
---

折叠表达式（Fold Expressions）是 C++17 引入的一种简化变参模版函数编写的语法。它允许你对参数包中的所有参数应用一个二元运算符，从而避免了手动展开参数包的繁琐过程。

---

## 基本语法

折叠表达式的基本语法如下：
```cpp
(... op args)          // 一元左折叠
(args op ...)          // 一元右折叠
(args1 op ... op args2) // 二元折叠
```
其中，`op` 是一个二元运算符，`args` 是一个参数包。

左折叠和右折叠的区别在于运算的顺序。左折叠从左到右应用运算符，而右折叠从右到左应用运算符。从 `...` 和模版参数 `args` 的位置可以看出这一点， `...` 在 `args` 的左侧表示左折叠，反之则表示右折叠。

---

## 示例

下面是一些使用折叠表达式的示例：

### 一元折叠

```cpp
template<typename... Args>
auto sum(Args... args) {
    return (... + args); // 使用左折叠计算和
}
```

这里是一元左折叠，`(... + args)` 会将所有参数相加。

展开后相当于：
```cpp
return (((args1 + args2) + args3) + ...);
```


```cpp
template<typename... Args>
auto sum(Args... args) {
    return (args + ...); // 使用右折叠计算和
}
```

这里是一元右折叠，`(args + ...)` 会将所有参数相加。

展开后相当于：
```cpp
return (args1 + (args2 + (args3 + ...)));
```

### 二元折叠

```cpp
template<typename... Args>
auto multiplyAndAdd(Args... args) {
    return (1 * ... * args) + (0 + ... + args); // 计算乘积和
}
```

这里是二元折叠，`(1 * ... * args)` 计算所有参数的乘积，`(0 + ... + args)` 计算所有参数的和。(左折叠)

以三个参数为例，展开后相当于：
```cpp
return (((1 * args1) * args2) * args3) + (((0 + args1) + args2) + args3);
```

---
title: "浅谈 C++ 模版编程中的 SFINAE、约束和概念"
collection: cpp
permalink: /cpp/cpp-template-feature
excerpt: '介绍 C++ 模版编程中的 SFINAE 技术、约束和概念的用法及区别'
date: 2025-12-03
---

在 C++ 20 之前，SFINAE（Substitution Failure Is Not An Error，替换失败不是错误）是一种常用的模版编程技巧，用于在模版实例化过程中根据类型特性选择合适的模版版本。SFINAE 允许编译器在遇到替换失败时忽略该模版实例化，而不是报错，从而实现条件编译。

然而，C++ 20 引入了概念（Concepts）和约束（Constraints），为模版编程提供了更直观和强大的工具。概念允许程序员定义类型的要求，而约束则可以直接应用于模版参数，使得代码更加清晰和易于维护。

最近也是在继续学习模版相关的知识，顺便记录一下 SFINAE、约束和概念的相关内容。

---

## SFINAE

SFINAE 是一种在模版实例化过程中根据类型特性选择合适模版版本的技术。它利用了编译器在替换模版参数时遇到错误不会立即报错的特性，从而允许程序员编写多个模版版本，并根据类型特性选择合适的版本。

举一个模版替换失败的例子：
```cpp
template <typename T
    , typename U = typename T::value_type> // 如果 T 没有 value_type 成员，这里会替换失败
void func(double t) {
    // ...
}

template <typename T>
void func(int t) {
    // ...
}

int main() {
    func<std::vector<int>>(3.14); // 调用第一个版本
    func<int>(42);                 // 调用第二个版本
    func<void>(3.14);            // 第一个版本替换失败，调用第二个版本
    return 0;
}
```
在上面的例子中，`func` 函数有两个版本。第一个版本依赖于类型 `T` 的成员类型 `value_type`。当传入的类型 `T` 没有 `value_type` 成员时，模版参数替换会失败，但由于 SFINAE 的特性，编译器不会报错，而是忽略该版本，选择第二个版本进行实例化。

如果需要一个支持 operator+ 的范型函数，可以使用 SFINAE 来实现：
```cpp
template <typename T>
auto add(const T& a, const T& b) -> decltype(a + b) {
    return a + b;
}
```
在这个例子中，`add` 函数使用了 `decltype` 来推断返回类型。如果传入的类型 `T` 不支持 `operator+`，则替换失败，编译器会忽略该版本。

### 标准库

C++ 标准库提供了一些工具来简化 SFINAE 的使用，例如 `std::enable_if` 和类型特征（type traits）。这些工具可以帮助程序员更方便地实现条件编译。

```cpp
#include <type_traits>
template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
increment(T value) {
    return value + 1;
}
```
在这个例子中，`increment` 函数只有在类型 `T` 是整数类型时才会被实例化，否则替换失败，编译器会忽略该版本。

还有 std::declval、std::void_t 等工具也常用于 SFINAE 编程中。
```cpp
template <typename T, typename = void>
struct has_to_string : std::false_type {};

template <typename T>
struct has_to_string<T, std::void_t<decltype(std::declval<T>().to_string())>> : std::true_type {};
```
在这个例子中，`has_to_string` 结构体用于检测类型 `T` 是否具有 `to_string` 成员函数。通过使用 `std::void_t` 和 `std::declval`，我们可以利用 SFINAE 来实现这一检测。

---

## 概念和约束

C++ 20 引入了概念和约束，为模版编程提供了更强大和直观的工具。概念允许程序员定义类型的要求，而约束则可以直接应用于模版参数。

如果想要定义一个支持 operator+ 的模版函数除了使用 SFINAE，还可以怎么实现呢？

在 C++ 20，概念和约束就可以做到这一点。

### 概念
```cpp
template <typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::same_as<T>;
};
```
这里定义了一个名为 `Addable` 的概念，要求类型 `T` 必须支持 `operator+`，并且返回类型与 `T` 相同。

这个概念会返回 true 或 false，可以用于约束模版参数。

### 约束
```cpp
template <Addable T>
T add(const T& a, const T& b) {
    return a + b;
}
```
在这个例子中，`add` 函数使用了 `Addable` 概念来约束模版参数 `T`。只有满足 `Addable` 概念的类型才能实例化该函数。

也可以使用 requires 子句来添加约束：
```cpp
template <typename T>
requires Addable<T>
T add(const T& a, const T& b) {
    return a + b;
}
```
这两种方式在功能上是等价的，选择哪种方式取决于个人偏好。

---

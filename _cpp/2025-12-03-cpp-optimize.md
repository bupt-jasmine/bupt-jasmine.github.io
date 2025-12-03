---
title: "关于 C++ 优化的几点思考"
collection: cpp
permalink: /cpp/cpp-optimize
excerpt: '介绍一些常见的 C++ 代码优化技巧和思考'
date: 2025-12-03
---

# 关于 C++ 优化的几点思考

在 C++ 编程中，代码优化是一个重要的方面，它不仅可以提升程序的性能，还能改善资源的使用效率。以下是一些常见的 C++ 代码优化技巧和思考。

## 使用 constexpr
C++11 引入了 constexpr 关键字，它允许在编译时计算常量表达式。通过使用 constexpr，可以减少运行时的计算开销，从而提升性能。
```cpp
constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}
```

## 内联 inline 函数
内联函数通过将函数体直接插入到调用点，减少了函数调用的开销。对于小型函数，使用 inline 可以显著提升性能。
```cpp
inline int add(int a, int b) {
    return a + b;
}
```

## 避免不必要的拷贝
使用引用（&）和指针（*）可以避免不必要的对象拷贝，尤其是在传递大型对象时。使用 const 引用可以确保对象不会被修改。
```cpp
void process(const std::vector<int>& data) {
    // 处理数据
}
```
对于 std::string，使用 std::string_view 可以避免拷贝，提高效率。
```cpp
void printString(std::string_view str) {
    std::cout << str << std::endl;
}
```

## 使用移动语义
C++11 引入了移动语义，通过使用 std::move，可以避免不必要的拷贝，提高性能。
```cpp
std::vector<int> createVector() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    return vec; // 利用移动语义
}
```

## 使用 [[likely]] 和 [[unlikely]]
C++20 引入了 [[likely]] 和 [[unlikely]] 属性，可以帮助编译器优化分支预测，从而提升性能。
```cpp
if (condition) [[likely]] {
    // 可能性较大的代码路径
} else [[unlikely]] {
    // 可能性较小的代码路径
}
```

## 避免使用虚函数
虚函数引入了运行时开销，尤其是在频繁调用的情况下。尽量使用模板和 CRTP（Curiously Recurring Template Pattern）来实现静态多态。
```cpp
template <typename Derived>
class Base {    
public:
    void interface() {
        static_cast<Derived*>(this)->implementation();
    }
};
``` 

## 使用 __restrict
在支持的编译器中，使用 __restrict 可以告诉编译器指针不会别名，从而允许更 aggressive 的优化。
```cpp
void add(int* __restrict a, const int* __restrict b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        a[i] += b[i];
    }
}
```

## CPU 代码使用 SIMD 指令集
利用 SIMD（Single Instruction, Multiple Data）指令集可以显著提升数据并行处理的性能。可以使用编译器内置函数或第三方库（如 Intel Intrinsics）来实现 SIMD 优化。
```cpp
#include <immintrin.h>
void addSIMD(float* a, float* b, float* result, size_t size) {
    for (size_t i = 0; i < size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }
}
```

## 并行编程
利用多线程和并行算法可以提升程序的性能。C++17 引入了并行 STL 算法，可以轻松实现数据并行处理。
```cpp
#include <algorithm>
#include <execution>
std::vector<int> data = { /* ... */ };
std::sort(std::execution::par, data.begin(), data.end());
```

## 善用编译器优化选项

编译器提供了多种优化选项（如 -O2、-O3、-Ofast 等），合理选择和使用这些选项可以显著提升程序性能。

比如开启 `-O2` 与不开启时的性能差别很大，尤其是在计算密集型应用中。

同时像一些大型项目中，因为链接了很多动态库，这时可以尝试开启 `-flto` 进行链接时优化（Link Time Optimization），进一步提升性能。

如果编译器的版本很旧，也可以尝试升级编译器版本，因为新版本的编译器通常会带来更好的优化效果。
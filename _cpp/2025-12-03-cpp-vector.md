---
title: "浅谈 std::vector 扩容"
collection: cpp
permalink: /cpp/cpp-vector
excerpt: '介绍 C++ std::vector 的扩容机制'
date: 2025-12-03
---

在 C++ 标准库中，`std::vector` 是一种动态数组容器，它能够根据需要自动调整其大小。当向 `std::vector` 添加元素时，如果当前容量不足以容纳新元素，`std::vector` 会进行扩容操作。这里学习并记录下 `std::vector` 的扩容机制及其相关细节。

## std::vector 底层实现

`std::vector` 通常由以下几个部分组成：
- **指针**：指向动态分配的内存区域，用于存储元素。
- **大小**：当前存储的元素数量。
- **容量**：当前分配的内存区域可以容纳的最大元素数量。

以 clang++ 编译器为例，`std::vector` 的底层实现大致如下：

```cpp
template <class _Tp, class _Allocator /* = allocator<_Tp> */>
class _LIBCPP_TEMPLATE_VIS vector {
    ...
public:
  typedef vector __self;
  typedef _Tp value_type;
  typedef _Allocator allocator_type;
  typedef allocator_traits<allocator_type> __alloc_traits;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef typename __alloc_traits::size_type size_type;
  typedef typename __alloc_traits::difference_type difference_type;
  typedef typename __alloc_traits::pointer pointer;
  typedef typename __alloc_traits::const_pointer const_pointer;
  ...

private:
  pointer __begin_ = nullptr;
  pointer __end_   = nullptr;
  __compressed_pair<pointer, allocator_type> __end_cap_ =
      __compressed_pair<pointer, allocator_type>(nullptr, __default_init_tag());

  ...
};
```

## 初始化

当一个 `std::vector` 被创建时，如果没有指定大小，它的大小为 0，容量也为 0。随着元素的添加，`std::vector` 会根据需要分配内存。
此时 `__begin_ == __end_ == __end_cap_();`

如果在创建时指定了初始大小，例如 `std::vector<int> vec(10);`，则 `std::vector` 会分配足够的内存来容纳 10 个元素，并将大小设置为 10。
此时 `__begin_` 指向分配的内存起始位置，`__end_` 指向第 10 个元素的下一个位置，`__end_cap_()` 也指向同一位置。

## 扩容机制

当向 `std::vector` 添加元素时，如果当前容量不足以容纳新元素，`std::vector` 会进行扩容操作。

具体源码如下：

```cpp
template <class _Tp, class _Allocator>
_LIBCPP_CONSTEXPR_SINCE_CXX20 inline _LIBCPP_HIDE_FROM_ABI void
vector<_Tp, _Allocator>::push_back(const_reference __x) {
  pointer __end = this->__end_;
  if (__end < this->__end_cap()) {
    __construct_one_at_end(__x);
    ++__end;
  } else {
    __end = __push_back_slow_path(__x);
  }
  this->__end_ = __end;
}
```

如果当前容量足够，`std::vector` 直接在末尾使用 `placement new` 构造新元素。如果容量不足，则调用 `__push_back_slow_path` 进行扩容。

`__push_back_slow_path` 的实现如下：

```cpp
template <class _Tp, class _Allocator>
template <class _Up>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename vector<_Tp, _Allocator>::pointer
vector<_Tp, _Allocator>::__push_back_slow_path(_Up&& __x) {
  allocator_type& __a = this->__alloc();
  __split_buffer<value_type, allocator_type&> __v(__recommend(size() + 1), size(), __a);
  // __v.push_back(std::forward<_Up>(__x));
  __alloc_traits::construct(__a, std::__to_address(__v.__end_), std::forward<_Up>(__x));
  __v.__end_++;
  __swap_out_circular_buffer(__v);
  return this->__end_;
}
```

这里会调用 `__recommend` 函数来计算新的容量大小。通常情况下，新的容量是当前容量的两倍，以减少频繁的扩容操作。

具体实现：

```cpp
//  Precondition:  __new_size > capacity()
template <class _Tp, class _Allocator>
_LIBCPP_CONSTEXPR_SINCE_CXX20 inline _LIBCPP_HIDE_FROM_ABI typename vector<_Tp, _Allocator>::size_type
vector<_Tp, _Allocator>::__recommend(size_type __new_size) const {
  const size_type __ms = max_size();
  if (__new_size > __ms)
    this->__throw_length_error();
  const size_type __cap = capacity();
  if (__cap >= __ms / 2)
    return __ms;
  return std::max<size_type>(2 * __cap, __new_size);
}
```

可以看到 `__recommend` 函数会计算新的容量大小，通常是当前容量的两倍，除非达到最大容量限制。

`__push_back_slow_path` 首先分配了新的内存区域 `__v`，移动了旧地址的数据(`__split_buffer` 这里会做)，然后在 `__v.__end_` 的地址处原地构造新元素。

`__swap_out_circular_buffer` 这一步是把新的指针交换到 `std::vector` 的成员变量中，并释放旧的内存区域。

---

## 关于扩容的时间复杂度

`std::vector` 的扩容操作虽然涉及内存分配和数据移动，但由于扩容通常是按倍数增长的，因此在均摊意义下，`push_back` 操作的时间复杂度仍然是 O(1)。这意味着，尽管某些 `push_back` 操作可能需要 O(n) 的时间来进行扩容，但大多数情况下，`push_back` 操作只需 O(1) 的时间。

---

## 迭代器失效

扩容操作会导致 `std::vector` 内部的内存地址发生变化，因此所有指向 `std::vector` 元素的迭代器、指针和引用在扩容后都会失效。开发者在使用 `std::vector` 时需要注意这一点，避免在扩容后继续使用旧的迭代器或指针。

可能导致 std::vector 迭代器失效的操作包括：
- 添加元素（如 `push_back`、`emplace_back`）
- 插入元素（如 `insert`、`emplace`）
- 调用 `resize` 方法增加大小
- 调用 `reserve` 方法增加容量

要避免编写出易受迭代器失效影响的代码，比如可能有问题的代码：
```cpp
std::vector<int> vec = {1, 2, 3};
for (auto it = vec.begin(); it != vec.end(); ++it) {
    if (*it == 2) {
        vec.push_back(4); // 可能导致迭代器失效
    }
}
```

---
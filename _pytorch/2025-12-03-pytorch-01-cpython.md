---
title: "CPython 基础"
collection: pytorch
permalink: /pytorch/pytorch-01-cpython
excerpt: '介绍 CPython 基础类及其宏定义，帮助理解 Python 对象模型'
date: 2025-12-03
---

在学习 PyTorch 源码之前，了解 CPython 的基础类和宏定义是非常重要的。PyTorch C++ 部分源码使用大量的 CPython C API 来实现与 Python 的交互。本文将介绍一些常见的 CPython 基础类及其宏定义，帮助理解 Python 对象模型。

一些基本的类包括：
- `PyObject`：所有 Python 对象的基类，定义了对象的基本属性和方法。
- `PyTypeObject`：表示 Python 类型的结构体，定义了类型的属性
- `PyVarObject`：表示可变大小的对象，如列表和字符串。

同时 cpython 定义了一些辅助宏，用于操作这些基础类，例如：
- `Py_INCREF` 和 `Py_DECREF`：用于增加和减少对象的引用计数。
- `Py_TYPE`：用于获取对象的类型。

---

## 一些宏定义

1. `_PyObject_HEAD_EXTRA`
```c
#ifdef Py_TRACE_REFS
#define _PyObject_HEAD_EXTRA      \
    PyObject *_ob_next;           \
    PyObject *_ob_prev;

#define _PyObject_EXTRA_INIT _Py_NULL, _Py_NULL,

#else
#  define _PyObject_HEAD_EXTRA
#  define _PyObject_EXTRA_INIT
#endif
```

如果使用 `--with-trace-refs` 编译则 `Py_TRACE_REFS` 为 true。

这里定义了一个宏 `_PyObject_HEAD_EXTRA`，用于在调试模式下为每个 `PyObject` 添加额外的指针字段 `_ob_next` 和 `_ob_prev`，以便跟踪对象的引用关系。在非调试模式下，这些字段不会被添加。

2. `PyObject_HEAD`
```c
/* PyObject_HEAD defines the initial segment of every PyObject. */
#define PyObject_HEAD                   PyObject ob_base;
```
这个宏定义了 `PyObject` 结构体的头部部分，包含了一个 `PyObject` 类型的字段 `ob_base`，用于表示对象的基本信息。

3. `PyObject_HEAD_INIT`
```c
#define PyObject_HEAD_INIT(type)        \
    { _PyObject_EXTRA_INIT              \
    1, type },
```
这个宏用于初始化 `PyObject` 结构体的头部部分。它设置了引用计数为 1，并将类型指针设置为传入的 `type` 参数。

4. `PyVarObject_HEAD_INIT`
```c
#define PyVarObject_HEAD_INIT(type, size)       \
    { PyObject_HEAD_INIT(type) size },
```
这个宏用于初始化 `PyVarObject` 结构体的头部部分。它调用了 `PyObject_HEAD_INIT` 宏，并设置了可变大小对象的大小字段 `size`。

5. `PyObject_VAR_HEAD`
```c
#define PyObject_VAR_HEAD      PyVarObject ob_base;
```
这个宏定义了 `PyVarObject` 结构体的头部部分，包含了一个 `PyVarObject` 类型的字段 `ob_base`，用于表示可变大小对象的基本信息。

---

## PyObject

在 CPython 源码里的 `Include/pytypedefs.h`里会对 `_object` 用 typedef 重定义为 `PyObject`。

```c++
typedef struct PyModuleDef PyModuleDef;
typedef struct PyModuleDef_Slot PyModuleDef_Slot;
typedef struct PyMethodDef PyMethodDef;
typedef struct PyGetSetDef PyGetSetDef;
typedef struct PyMemberDef PyMemberDef;

typedef struct _object PyObject;
typedef struct _longobject PyLongObject;
typedef struct _typeobject PyTypeObject;
typedef struct PyCodeObject PyCodeObject;
typedef struct _frame PyFrameObject;

typedef struct _ts PyThreadState;
typedef struct _is PyInterpreterState;
```

实际声明&定义：

```c++
struct _object {
    _PyObject_HEAD_EXTRA
    Py_ssize_t ob_refcnt;
    PyTypeObject *ob_type;
};
```

PyTypeObject 为 _typeobject。

`PyObject` 是所有 Python 对象的基类。它定义了对象的基本属性和方法，如引用计数和类型信息。每个 Python 对象都包含一个指向其类型的指针，以及一个引用计数器，用于内存管理。

---

## PyVarObject

定义：
```c++
typedef struct {
    PyObject ob_base;
    Py_ssize_t ob_size; /* Number of items in variable part */
} PyVarObject;
```

`PyVarObject` 是 `PyObject` 的子类，表示可变大小的对象。它在 `PyObject` 的基础上添加了一个字段 `ob_size`，用于存储对象的大小（如元素数量）。这对于列表、字符串等可变长度的数据结构非常重要。

---

## PyTypeObject

定义：
```c++
// include/cpython/object.h

// If this structure is modified, Doc/includes/typestruct.h should be updated
// as well.
struct _typeobject {
    PyObject_VAR_HEAD
    const char *tp_name; /* For printing, in format "<module>.<name>" */
    Py_ssize_t tp_basicsize, tp_itemsize; /* For allocation */

    /* Methods to implement standard operations */

    destructor tp_dealloc;
    Py_ssize_t tp_vectorcall_offset;
    getattrfunc tp_getattr;
    setattrfunc tp_setattr;
    PyAsyncMethods *tp_as_async; /* formerly known as tp_compare (Python 2)
                                    or tp_reserved (Python 3) */
    reprfunc tp_repr;

    /* Method suites for standard classes */

    PyNumberMethods *tp_as_number;
    PySequenceMethods *tp_as_sequence;
    PyMappingMethods *tp_as_mapping;

    /* More standard operations (here for binary compatibility) */

    hashfunc tp_hash;
    ternaryfunc tp_call;
    reprfunc tp_str;
    getattrofunc tp_getattro;
    setattrofunc tp_setattro;

    /* Functions to access object as input/output buffer */
    PyBufferProcs *tp_as_buffer;

    /* Flags to define presence of optional/expanded features */
    unsigned long tp_flags;

    const char *tp_doc; /* Documentation string */

    /* Assigned meaning in release 2.0 */
    /* call function for all accessible objects */
    traverseproc tp_traverse;

    /* delete references to contained objects */
    inquiry tp_clear;

    /* Assigned meaning in release 2.1 */
    /* rich comparisons */
    richcmpfunc tp_richcompare;

    /* weak reference enabler */
    Py_ssize_t tp_weaklistoffset;

    /* Iterators */
    getiterfunc tp_iter;
    iternextfunc tp_iternext;

    /* Attribute descriptor and subclassing stuff */
    PyMethodDef *tp_methods;
    PyMemberDef *tp_members;
    PyGetSetDef *tp_getset;
    // Strong reference on a heap type, borrowed reference on a static type
    PyTypeObject *tp_base;
    PyObject *tp_dict;
    descrgetfunc tp_descr_get;
    descrsetfunc tp_descr_set;
    Py_ssize_t tp_dictoffset;
    initproc tp_init;
    allocfunc tp_alloc;
    newfunc tp_new;
    freefunc tp_free; /* Low-level free-memory routine */
    inquiry tp_is_gc; /* For PyObject_IS_GC */
    PyObject *tp_bases;
    PyObject *tp_mro; /* method resolution order */
    PyObject *tp_cache;
    PyObject *tp_subclasses;
    PyObject *tp_weaklist;
    destructor tp_del;

    /* Type attribute cache version tag. Added in version 2.6 */
    unsigned int tp_version_tag;

    destructor tp_finalize;
    vectorcallfunc tp_vectorcall;
};
```

`PyTypeObject` 结构体表示 Python 类型。它定义了类型的属性和方法，如名称、大小、方法表等。每个 Python 类型（如整数、字符串、列表等）都有一个对应的 `PyTypeObject` 实例。其中包含了大量的函数指针，用于实现类型的各种操作，如方法调用、属性访问、算术运算等。

---

## PyModuleDef

定义：
```c++
struct PyModuleDef {
  PyModuleDef_Base m_base;
  const char* m_name;
  const char* m_doc;
  Py_ssize_t m_size;
  PyMethodDef *m_methods;
  PyModuleDef_Slot *m_slots;
  traverseproc m_traverse;
  inquiry m_clear;
  freefunc m_free;
};
typedef struct PyModuleDef_Base {
  PyObject_HEAD
  PyObject* (*m_init)(void);
  Py_ssize_t m_index;
  PyObject* m_copy;
} PyModuleDef_Base;

#define PyModuleDef_HEAD_INIT {  \
    PyObject_HEAD_INIT(_Py_NULL) \
    _Py_NULL, /* m_init */       \
    0,        /* m_index */      \
    _Py_NULL, /* m_copy */       \
  }
```

`PyModuleDef` 结构体用于定义 Python 模块。它包含了模块的名称、文档字符串、方法表等信息。通过定义 `PyModuleDef`，可以创建自定义的 Python 模块，并将其与 C/C++ 代码进行绑定。

在 PyTorch 源码里就有：
```c++
// torch/csrc/Module.cpp initModule()

extern "C" TORCH_PYTHON_API PyObject* initModule{

  // ...
  static struct PyModuleDef torchmodule = {
      PyModuleDef_HEAD_INIT, "torch._C", nullptr, -1, methods.data()};
  module = PyModule_Create(&torchmodule);
 
  // ...
  return module;
}
```

---

## PyMethodDef

定义：
```c++
struct PyMethodDef {
    const char  *ml_name;   /* The name of the built-in function/method */
    PyCFunction ml_meth;    /* The C function that implements it */
    int         ml_flags;   /* Combination of METH_xxx flags, which mostly
                               describe the args expected by the C func */
    const char  *ml_doc;    /* The __doc__ attribute, or NULL */
};
```

`PyMethodDef` 结构体用于定义 Python 方法。它包含了方法的名称、实现该方法的 C 函数指针、参数标志以及文档字符串。通过定义 `PyMethodDef`，可以将 C/C++ 函数暴露为 Python 方法，使其可以在 Python 代码中调用。

---

## 自定义 Python 模块

可以使用 `PyInit_<modulename>` 函数来初始化一个自定义的 Python 模块。

python 在 import 模块时，会调用模块中的 `PyInit_<modulename>` 函数来初始化模块。这个函数需要返回一个指向 `PyObject` 的指针，表示模块对象。

**注意：**
1. 这里在 import 模块时，模块名需要和 `PyInit_<modulename>` 中的 `<modulename>` 一致，否则会导致导入失败。
2. 编译好的模块文件名也需要和模块名一致，例如 `mymodule` 模块对应的文件名应该是 `mymodule.so`（Linux）或 `mymodule.pyd`（Windows）。
3. 动态库的导入路径需要在 `PYTHONPATH` 环境变量中，或者放在当前工作目录下，否则 Python 解释器无法找到该模块。

比如 PyTorch 的 C++ 扩展模块，就是通过这种方式导入的。
```c++
// torch/csrc/stub.c

#include <Python.h>

extern PyObject* initModule(void);

#ifndef _WIN32
#ifdef __cplusplus
extern "C"
#endif
__attribute__((visibility("default"))) PyObject* PyInit__C(void);
#endif

PyMODINIT_FUNC PyInit__C(void)
{
  return initModule();
}
```

---

## PyObject 在 PyTorch 源码中的使用实例

关于 `PyObject`、`PyVarObject` 和 `PyTypeObject`，在 PyTorch 的源码里有很多使用实例。

### 1. `THPVariable` 结构体
```c
// torch/csrc/autograd/python_variable.h

// Python object that backs torch.autograd.Variable
struct THPVariable {
  PyObject_HEAD
  // Payload
  c10::MaybeOwned<at::Tensor> cdata;
  // Hooks to be run on backwards pass (corresponds to Python attr
  // '_backwards_hooks', set by 'register_hook')
  PyObject* backward_hooks = nullptr;
  // Hooks to be run in the backwards pass after accumulate grad,
  // i.e., after the .grad has been set (corresponds to Python attr
  // '_post_accumulate_grad_hooks', set by 'register_post_accumulate_grad_hook')
  PyObject* post_accumulate_grad_hooks = nullptr;
};
```
这里定义了一个 `THPVariable` 结构体，表示 PyTorch 中的 `Variable` 对象。它继承自 `PyObject`，并包含了一个 `cdata` 字段，用于存储底层的 `at::Tensor` 数据。此外，它还包含了两个钩子列表，用于在反向传播过程中执行自定义操作。

### 2. THPVariableMetaType 和 THPVariableType
```c
// torch/csrc/autograd/python_variable.cpp

static PyTypeObject THPVariableMetaType = {
    PyVarObject_HEAD_INIT(DEFERRED_ADDRESS(&PyType_Type), 0)
    "torch._C._TensorMeta", /* tp_name */
    sizeof(THPVariableMeta), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    DEFERRED_ADDRESS(&PyType_Type), /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    THPVariableMetaType_init, /* tp_init */
    nullptr, /* tp_alloc */
    nullptr, /* tp_new */
};

static PyTypeObject THPVariableType = {
    PyVarObject_HEAD_INIT(&THPVariableMetaType, 0)
    "torch._C.TensorBase", /* tp_name */
    sizeof(THPVariable), /* tp_basicsize */
    0, /* tp_itemsize */
    // This is unspecified, because it is illegal to create a THPVariableType
    // directly.  Subclasses will have their tp_dealloc set appropriately
    // by the metaclass
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    &THPVariable_as_mapping, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC, /* tp_flags */
    nullptr, /* tp_doc */
    // Also set by metaclass
    (traverseproc)THPFake_traverse, /* tp_traverse */
    (inquiry)THPFake_clear, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    THPVariable_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    // Although new is provided here, it is illegal to call this with cls ==
    // THPVariableMeta.  Instead, subclass it first and then construct it
    THPVariable_pynew, /* tp_new */
};
```

在 python 中打印 `type(torch.Tensor)`:
```python
>>> import torch
>>> print(type(torch.Tensor))
<class 'torch._C._TensorMeta'>
>>>
```

可以看到 `torch.Tensor` 的类型是 `torch._C._TensorMeta`，这正是上面定义的 `THPVariableMetaType`。

而 `torch.Tensor` 的实例对象类型是 `torch._C.TensorBase`，对应上面的 `THPVariableType`。

在 PyTorch 源码中，还有大量使用 CPython 的地方，还需要继续学习和探索。

---

## 实现一个 Python 迭代器

首先需要定义一个结构体，表示迭代器对象。这个结构体需要包含 `PyObject_HEAD` 宏，以便继承自 `PyObject`。

```cpp
struct MyIterator {
    PyObject_HEAD
    // other fields ...
};
```

然后需要实现迭代器的 `tp_iternext` 方法，这个方法用于返回下一个元素。如果没有更多元素可供迭代，应该返回 `NULL` 并设置 `StopIteration` 异常。

```cpp
PyObject* MyIterator_iternext(MyIterator* self) {
    // 这里可以通过 MyIterator 结构体中的其他字段来获取下一个元素
    // 具体实现就需要根据实际需求来编写
    if (no_more_elements) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
    // 返回下一个元素
    PyObject* next_element = ...;
    return next_element;
}
```

最后需要定义迭代器类型的 `PyTypeObject` 结构体，并将 `tp_iternext` 方法指针设置为上面实现的函数。

```cpp
PyTypeObject* GetIterType() {
    static auto type_getter = []() {
        static PyTypeObject iter_type = {
            PyVarObject_HEAD_INIT(NULL, 0)
            "MyIterator",               /* tp_name */
            sizeof(MyIterator),        /* tp_basicsize */
            0,                          /* tp_itemsize */
            nullptr,                    /* tp_dealloc */
            0,                          /* tp_vectorcall_offset */
            nullptr,                    /* tp_getattr */
            nullptr,                    /* tp_setattr */
            nullptr,                    /* tp_reserved */
            nullptr,                    /* tp_repr */
            nullptr,                    /* tp_as_number */
            nullptr,                    /* tp_as_sequence */
            nullptr,                    /* tp_as_mapping */
            nullptr,                    /* tp_hash  */
            nullptr,                    /* tp_call */
            nullptr,                    /* tp_str */
            nullptr,                    /* tp_getattro */
            nullptr,                    /* tp_setattro */
            nullptr,                    /* tp_as_buffer */
            Py_TPFLAGS_DEFAULT,         /* tp_flags */
            "My custom iterator",       /* tp_doc */
            nullptr,                    /* tp_traverse */
            nullptr,                    /* tp_clear */
            nullptr,                    /* tp_richcompare */
            0,                          /* tp_weaklistoffset */
            nullptr,                    /* tp_iter */
            (iternextfunc)MyIterator_iternext, /* tp_iternext */
            // other fields ...
        };

        if (PyType_Ready(&iter_type) < 0) {
            throw std::runtime_error("Failed to initialize MyIterator type");
        }
        return &iter_type;
    };

    static PyTypeObject* iter_type = type_getter();
    return iter_type;
}
```

通过以上步骤，就可以实现一个自定义的 Python 迭代器类型。可以根据实际需求，在 `MyIterator` 结构体中添加其他字段，以便在迭代过程中存储状态信息。

```cpp
inline PyObject* CreateMyIterator() {
    auto iter = PyObject_CallObject((PyObject*)GetIterType(), nullptr);
    if (PyErr_Occurred()) {
        return nullptr;
    }
    return iter;
}
```

---
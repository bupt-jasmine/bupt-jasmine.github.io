---
permalink: /markdown/
title: "使用指南"
author_profile: true
redirect_from: 
  - /md/
  - /markdown.html
---

{% include toc %}

## 网站结构

### 主要文件和目录

* **配置文件**: `_config.yml` - 网站基本配置
* **导航菜单**: `_data/navigation.yml` - 顶部导航栏配置
* **页面文件**: `_pages/` - 单独的页面（如首页、CV等）
* **内容集合**:
  * `_cpp/` - C++ 学习笔记
  * `_pytorch/` - PyTorch 学习笔记
  * `_portfolio/` - 项目作品集
  * `_posts/` - 博客文章
* **静态文件**: `files/` - PDF 等文件
* **头像图片**: `images/profile.jpeg`

## 如何添加内容

### C++ 学习笔记

在 `_cpp/` 目录下创建文件，命名格式：`YYYY-MM-DD-文件名.md`

```markdown
---
title: "笔记标题"
collection: cpp
permalink: /cpp/your-note-name
excerpt: '简短描述'
date: 2025-12-02
---

你的笔记内容...
```

### PyTorch 学习笔记

在 `_pytorch/` 目录下创建文件，格式同上

```markdown
---
title: "笔记标题"
collection: pytorch
permalink: /pytorch/your-note-name
excerpt: '简短描述'
date: 2025-12-02
---

你的笔记内容...
```

### 博客文章

在 `_posts/` 目录下创建文件，命名格式：`YYYY-MM-DD-文章名.md`

```markdown
---
title: '文章标题'
date: 2025-12-02
permalink: /posts/2025/12/article-name/
tags:
  - 标签1
  - 标签2
---

文章内容...
```

### 项目展示

在 `_portfolio/` 目录下创建文件

```markdown
---
title: "项目名称"
excerpt: "项目简介"
collection: portfolio
---

## 项目介绍
...
```

## 使用技巧

### 文件格式

* `.md` 文件会被解析为 Markdown
* `.html` 文件会被解析为 HTML

### 部署状态检查

访问你的 GitHub 仓库的 Actions 页面查看构建状态：
* ✅ 绿色对勾：构建成功
* 🟠 橙色圆圈：正在构建  
* ❌ 红色 X：构建失败

### Markdown 解析

本站使用 Jekyll Kramdown 解析器，支持 GitHub Flavored Markdown (GFM)。

## 常用 Markdown 语法

### 标题

```markdown
# 一级标题
## 二级标题
### 三级标题
```

### 列表

**无序列表**：
```markdown
* 项目 1
* 项目 2
  * 子项目 2.1
```

**有序列表**：
```markdown
1. 第一项
2. 第二项
```

### 代码

行内代码：`` `code` ``

代码块：
````markdown
```python
def hello():
    print("Hello, World!")
```
````

### 链接和图片

链接：`[链接文字](https://example.com)`

图片：`![图片描述](/images/example.png)`

### 表格

```markdown
| 列1 | 列2 | 列3 |
|-----|:---:|----:|
| 左对齐 | 居中 | 右对齐 |
```

### 引用

```markdown
> 这是一段引用文字
```

### 强调

```markdown
**粗体**
*斜体*
~~删除线~~
```

## 数学公式 (MathJax)

行内公式：`\\(a^2 + b^2 = c^2\\)`

块级公式：
```markdown
$$
E = mc^2
$$
```

## 更多资源

* [Jekyll 文档](https://jekyllrb.com/docs/)
* [Markdown 指南](https://www.markdownguide.org/)
* [GitHub Pages 文档](https://docs.github.com/en/pages)
* [Kramdown 语法](https://kramdown.gettalong.org/syntax.html)

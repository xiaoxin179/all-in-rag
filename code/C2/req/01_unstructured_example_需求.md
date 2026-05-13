# 01_unstructured_example.py 动手练习清单

## RAG 流程定位

当前代码属于 RAG 数据管道的**第一个环节：文档加载（Document Loading）**

- **输入**：PDF 文件 (`rag.pdf`)
- **处理**：使用 Unstructured 的 `partition()` 函数解析 PDF，识别文档结构
- **输出**：结构化的文档元素列表（包含类型、文本内容）

## 练习任务

### 练习 1：使用 partition_pdf 替代 partition

**学习目标**：理解 Unstructured 文档加载器中 `partition` 与 `partition_pdf` 的区别

**任务描述**：
大白话：原来用的是通用的文档解析器，现在换成专门解析 PDF 的版本。试试看换成 `partition_pdf` 后能不能正常工作。

可以

**实现思路**：

```python
# 原代码
from unstructured.partition.auto import partition

# 改成
from unstructured.partition.pdf import partition_pdf

# 调用方式相同
elements = partition_pdf(
    filename=pdf_path,
    # partition_pdf 的专属参数后续练习会用到
)
```

**验证方法**：
运行后观察：
1. 输出的元素数量是否相同
2. 元素类型分布是否有变化
3. 是否有报错信息

**扩展思考**：
- `partition_pdf` 比 `partition` 多支持哪些参数？

`partition_pdf` 多了 `strategy`、`OCR语言`、`表格识别`、`图片提取`、`页码控制` 这些 PDF 专有参数。如果只是简单解析一个标准 PDF，`partition` 其实够用了。

- 为什么说 `partition_pdf` 性能更优？

---

### 练习 2：尝试不同的 PDF 解析策略

**学习目标**：理解 Unstructured 的 PDF 解析策略参数（strategy）

**任务描述**：
大白话：`partition_pdf` 有三种解析策略 —— `auto`（自动选择）、`fast`（快速但可能丢失精度）、`hi_res`（高精度）。现在分别试试这三种策略，看看结果有什么不同。

**实现思路**：

```python
from unstructured.partition.pdf import partition_pdf

pdf_path = "../../data/C2/pdf/rag.pdf"

# 尝试三种策略
strategies = ["auto", "fast", "hi_res"]

for strategy in strategies:
    print(f"\n{'='*60}")
    print(f"策略: {strategy}")
    print('='*60)
    
    elements = partition_pdf(
        filename=pdf_path,
        strategy=strategy
    )
    
    print(f"元素数量: {len(elements)}")
    print(f"总字符数: {sum(len(str(e)) for e in elements)}")
    
    # 统计元素类型
    from collections import Counter
    types = Counter(e.category for e in elements)
    print(f"元素类型分布: {dict(types)}")
```

**验证方法**：
1. 对比三种策略的元素数量
2. 对比元素类型分布是否一致
3. 注意观察哪种策略速度更快

**扩展思考**：
- `hi_res` 策略为什么能获得更高精度？它内部用了什么技术？
- 什么时候应该用 `fast` 而非 `hi_res`？

---

### 练习 3：启用 OCR 模式解析扫描 PDF

**学习目标**：理解 OCR 在 PDF 解析中的作用

**任务描述**：
大白话：有些 PDF 是扫描件（图片组成的），没有可选择的文字。这时候需要 OCR 来识别文字。试试 `ocr_only` 模式，看看能不能解析出文字内容。

**实现思路**：

```python
from unstructured.partition.pdf import partition_pdf

pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用 OCR 专用策略
elements = partition_pdf(
    filename=pdf_path,
    strategy="ocr_only"  # 只用 OCR 模式
)

print(f"OCR 解析完成: {len(elements)} 个元素")
for i, element in enumerate(elements[:5], 1):  # 只打印前5个
    print(f"Element {i} ({element.category}):")
    print(element)
    print("-" * 40)
```

**验证方法**：
1. 对比 `ocr_only` 和 `hi_res` 的结果
2. 如果原 PDF 有文字层，两种方式应该都能识别
3. 如果 PDF 是纯图片扫描件，`ocr_only` 可能反而更准确

**扩展思考**：
- 什么情况下 `hi_res` + OCR 组合使用效果最好？
- OCR 的语言参数怎么设置？

---

### 练习 4：提取并保存元数据

**学习目标**：理解文档加载过程中的元数据提取

**任务描述**：
大白话：Unstructured 解析出来的每个元素都带元数据（metadata），里面包含页码、来源文件等信息。试试把第一个元素的元数据打印出来看看里面有什么。

**实现思路**：

```python
from unstructured.partition.pdf import partition_pdf

pdf_path = "../../data/C2/pdf/rag.pdf"

elements = partition_pdf(
    filename=pdf_path,
    strategy="auto"
)

# 查看第一个元素的详细信息
if elements:
    first_element = elements[0]
    print(f"元素类型: {first_element.category}")
    print(f"元素文本: {str(first_element)[:200]}...")
    print(f"\n元数据内容:")
    print(first_element.metadata)
    
    # 查看所有元素有哪些元数据字段
    print("\n所有元素的元数据字段（第一个非空示例）:")
    for key, value in first_element.metadata.__dict__.items():
        if value is not None:
            print(f"  {key}: {value}")
```

**验证方法**：
1. 观察元数据中包含哪些字段
2. 注意 `page_number`、`parent_id`、`source` 等字段
3. 这些元数据在后续 RAG 流程中有什么用？

**扩展思考**：
- 为什么要保留页码信息？
- 如何利用元数据来优化检索结果？

---

### 练习 5（可选）：尝试提取图片元素

**学习目标**：了解 PDF 中图片的提取能力

**任务描述**：
大白话：PDF 里可能包含图片，`partition_pdf` 可以提取图片信息。试试开启图片提取功能，看看能发现多少图片。

**实现思路**：

```python
from unstructured.partition.pdf import partition_pdf

pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用 hi_res 策略并提取图片
elements = partition_pdf(
    filename=pdf_path,
    strategy="hi_res",
    extract_image_block_types=["Image"],  # 提取图片
)

# 筛选出图片类型的元素
images = [e for e in elements if e.category == "Image"]
print(f"发现 {len(images)} 张图片")

for i, img in enumerate(images, 1):
    print(f"\n图片 {i}:")
    print(f"  元数据: {img.metadata}")
    print(f"  内容: {str(img)[:100]}...")
```

**验证方法**：
1. 检查 PDF 中实际包含多少图片
2. 观察图片元数据中的分辨率、位置等信息
3. 思考图片在 RAG 中的处理方式

**扩展思考**：
- 如果需要提取图片的实际文件（而非元数据），应该怎么做？
- 多模态 RAG 中图片如何处理？

---

## 相关知识点

| 概念 | 解释 |
|------|------|
| **文档加载器 (Document Loader)** | RAG 流水线的第一步，负责将各种格式的文档解析为程序可处理的结构化数据 |
| **Unstructured** | 专门为 RAG 场景设计的文档解析库，支持 PDF、Word、HTML 等多种格式 |
| **partition()** | Unstructured 的统一入口函数，自动识别文件类型并调用对应的解析器 |
| **partition_pdf()** | 专门用于 PDF 解析的函数，提供更多 PDF 特有参数 |
| **策略 (strategy)** | PDF 解析的不同策略：`auto` 自动选择、`fast` 快速模式、`hi_res` 高精度模式、`ocr_only` 仅 OCR |
| **OCR** | 光学字符识别，将图片中的文字转换为可选择的文本 |
| **元素类型 (Element Category)** | Unstructured 识别出的文档结构类型，如 Title、NarrativeText、Table、ListItem 等 |
| **元数据 (Metadata)** | 文档元素附带的额外信息，包括页码、来源、坐标等 |
| **Garbage In, Garbage Out** | RAG 核心原则：高质量输入才能带来高质量输出 |

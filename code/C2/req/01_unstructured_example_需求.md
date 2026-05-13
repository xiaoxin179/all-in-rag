# `01_unstructured_example.py` 动手练习清单

## RAG 流程定位

当前代码属于 **数据加载（Document Loading）** 环节：
- 输入：PDF 文件（`rag.pdf`）
- 处理：使用 `unstructured` 库解析 PDF，提取结构化元素
- 输出：`elements` 列表，每个元素包含文本内容和类型分类

---

## 练习任务

### 练习 1：统计 PDF 中的图片数量

**学习目标**：理解 PDF 解析后元素的多模态性

**任务描述**：
当前代码只打印了文本内容。请修改代码，统计并打印 PDF 中包含的图片（`Image` 类型）数量。

**关键代码提示**：
```python
# 统计图片数量
image_count = sum(1 for e in elements if e.category == "Image")
print(f"PDF 中包含 {image_count} 张图片")
```

**验证方法**：
运行后能看到类似 `PDF 中包含 3 张图片` 的输出即成功。

**扩展思考**：
- 在 RAG 中，图片如何参与检索？有没有办法把图片转成文字描述？
- 多模态 RAG 是什么？

---

### 练习 2：保存解析结果到 JSON 文件

**学习目标**：理解数据持久化的重要性，以及 RAG 管道中中间结果保存

**任务描述**：
将解析出来的所有元素保存到一个 JSON 文件中，包含每个元素的 `category`（类型）和 `text`（内容）。

**关键代码提示**：
```python
import json

# 准备数据
result = [
    {"category": e.category, "text": str(e)}
    for e in elements
]

# 保存到 JSON
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"已保存到 output.json")
```

**验证方法**：
生成一个 `output.json` 文件，可以用 VSCode 打开查看结构。

**扩展思考**：
- 如果数据量很大，JSON 存储有什么问题？
- RAG 项目中通常用什么方式存储向量数据？

---

### 练习 3：只提取"标题"和"段落"元素

**学习目标**：理解 PDF 解析后的元素过滤，理解为什么不是所有元素都适合进入 RAG

**任务描述**：
修改代码，只保留 `Title`（标题）和 `NarrativeText`（段落）类型的元素，过滤掉其他类型（如图片、表格、脚注等）。

**关键代码提示**：
```python
# 过滤：只保留标题和段落
target_categories = {"Title", "NarrativeText"}
filtered_elements = [e for e in elements if e.category in target_categories]

print(f"过滤后剩余 {len(filtered_elements)} 个元素")
for e in filtered_elements[:3]:
    print(f"[{e.category}] {e}")
```

**验证方法**：
运行后只看到 `Title` 和 `NarrativeText` 开头的元素。

**扩展思考**：
- 表格（Table）元素要不要保留？如果要，怎么转成文字？
- 脚注、页眉页脚通常对回答问题有没有帮助？

---

### 练习 4：添加 PDF 密码处理

**学习目标**：理解实际项目中 PDF 可能有各种"异常"情况

**任务描述**：
当前代码假设 PDF 是可直接打开的。请添加一个检查：如果 PDF 打开失败，尝试用空密码重试（有些 PDF 默认密码为空）。

**关键代码提示**：
```python
from unstructured.partition.api import partition_via_api
from unstructured.partition.pdf_image import partition_pdf

# 尝试无密码打开
try:
    elements = partition(filename=pdf_path, content_type="application/pdf")
except Exception as e:
    print(f"打开失败，尝试无密码重试: {e}")
    # partition_pdf 支持 parameters 参数
    elements = partition_pdf(
        filename=pdf_path,
        passwords={"": ""}  # 尝试空密码
    )
```

**验证方法**：
- 如果你的 PDF 没有密码：代码直接成功
- 如果有密码：需要找到正确密码才能通过

**扩展思考**：
- 在企业 RAG 项目中，如何批量处理各种"异常"文件？
- 有没有专门做文档预处理的库？

---

## 相关知识点

| 概念 | 解释 |
|------|------|
| 多模态 | 同时处理文字、图片、音频等多种类型数据 |
| 文档解析 | 把 PDF/Word 等文件转换成可处理的文本结构 |
| 元素类型 | 不同库对 PDF 内容有不同的分类方式（Title/NarrativeText/Table/Image） |
| 多模态 RAG | 用 OCR 或视觉模型处理图片，或将图片转成文字描述再检索 |

---

## 继续学习建议

完成这些练习后，下一步可以探索：
1. 对比 `01_unstructured_example.py`（PDF）和 `02/03/04`（TXT）的加载方式差异
2. 了解 `unstructured` 库还支持哪些文件格式（Word、HTML、邮件等）
3. 学习多模态 RAG：用 `marker` 或 `Docling` 等库处理更复杂的 PDF

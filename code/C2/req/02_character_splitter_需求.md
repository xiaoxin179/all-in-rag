# 02_character_splitter.py 动手练习清单

## RAG 流程定位

当前代码属于 RAG 的 **文本分块（Chunking）** 环节。

- **输入**：原始文档文本（通过 TextLoader 加载）
- **处理**：按固定字符数切分文本
- **输出**：多个文本块（Document 对象列表）

## 练习任务

### 练习 1：理解 chunk_size 和 chunk_overlap

**学习目标**：掌握分块器最基本的两个参数

**任务描述**：
修改 `chunk_size` 和 `chunk_overlap` 的值，观察分块数量和大小的变化。

**实现思路**：

```python
# 尝试不同的参数组合
test_cases = [
    (100, 0),    # 小块，无重叠
    (100, 20),   # 小块，有重叠
    (500, 50),   # 大块，有重叠
    (1000, 100), # 更大块
]

for size, overlap in test_cases:
    text_splitter = CharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap
    )
    chunks = text_splitter.split_documents(docs)
    print(f"size={size}, overlap={overlap} -> {len(chunks)} 块")
```

**验证方法**：
运行代码后，观察：
1. `chunk_size` 越大，块数越少
2. `chunk_overlap` 越大，相邻块重复内容越多

**扩展思考**：
- **扩展思考 1**：chunk_size 设太大（1000+）会有什么缺点？
  - 块内噪声增加：包含大量与查询不相关的上下文信息，导致向量相似度计算不准确
  - 语义稀释：核心信息被稀释在冗余内容中，降低检索精度
  - 上下文窗口浪费：输入给大模型的上下文包含过多无关内容，增加 token 消耗且可能影响回答质量

- **扩展思考 2**：chunk_overlap 设太大（接近 chunk_size）会怎样？
  - 冗余度飙升：相邻块之间高度重复，浪费存储和计算资源
  - 检索重复：可能返回多个内容高度相似的块，导致上下文冗余
  - 成本增加：重复内容被重复编码和检索，增加向量数据库存储成本和推理开销

---

### 练习 2：观察分隔符对中文分块的影响

**学习目标**：理解 CharacterTextSplitter 如何处理中文文本

**任务描述**：
在代码中添加打印语句，观察分块是否在标点符号处断开。

**实现思路**：

```python
# 修改代码，打印每个块的最后几个字符
for i, chunk in enumerate(chunks[:10]):
    text = chunk.page_content
    print(f"块 {i+1} 结尾: ...{text[-15:] if len(text) > 15 else text}")
```

**验证方法**：
观察分块是否在 `。！？，；` 等中文标点处断开，还是硬切断。

**扩展思考**：
- **问题 1**：如果文本里没有标点符号会怎样？
  - 分块器将无法在语义边界处断开，被迫进行硬切割
  - 文本会在精确的 chunk_size 位置被切断，可能将完整的词语或短语从中间拆开
  - 导致检索时块内语义不完整，影响向量表示质量和检索准确率

- **问题 2**：英文文本的分块效果和中文有什么不同？
  - 英文单词间天然以空格分隔，切分器更容易找到语义边界（如句号、逗号后的空格）
  - 英文句子结构更规则，分块结果更均匀；中文句子边界模糊，硬切割风险更高
  - 对于代码或特殊格式文本，两者表现类似，都可能产生硬切割

---

### 练习 3：统计分块大小分布

**学习目标**：理解分块大小的不均匀性问题

**任务描述**：
编写代码统计所有分块的大小分布，检查是否存在大小差异很大的情况。

**实现思路**：

```python
chunk_sizes = [len(chunk.page_content) for chunk in chunks]

print(f"块大小统计:")
print(f"  最小: {min(chunk_sizes)} 字符")
print(f"  最大: {max(chunk_sizes)} 字符")
print(f"  平均: {sum(chunk_sizes)/len(chunk_sizes):.1f} 字符")

# 统计各范围的块数量
size_ranges = {"<100": 0, "100-200": 0, "200-300": 0, ">300": 0}
for size in chunk_sizes:
    if size < 100: size_ranges["<100"] += 1
    elif size < 200: size_ranges["100-200"] += 1
    elif size < 300: size_ranges["200-300"] += 1
    else: size_ranges[">300"] += 1
print(f"分布: {size_ranges}")
```

**验证方法**：
观察块大小是否都在 `chunk_size`（200）附近，还是有较大的偏差。

**扩展思考**：
- 大小不均匀会影响什么？
- 什么情况下会产生很小的块（接近 0）？

---

### 练习 4：切换到 RecursiveCharacterTextSplitter 对比效果

**学习目标**：对比两种分块器的效果差异

**任务描述**：
分别用两种分块器处理同一文本，对比输出差异。

**实现思路**：

```python
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

text_splitter1 = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunks1 = text_splitter1.split_documents(docs)

text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunks2 = text_splitter2.split_documents(docs)

print(f"CharacterTextSplitter: {len(chunks1)} 块")
print(f"RecursiveCharacterTextSplitter: {len(chunks2)} 块")

# 对比第一个块的内容
print("\n=== CharacterTextSplitter 块1 ===")
print(chunks1[0].page_content[:100])
print("\n=== RecursiveCharacterTextSplitter 块1 ===")
print(chunks2[0].page_content[:100])
```

**验证方法**：
1. 块数量是否不同
2. 分块位置是否更合理（递归版本通常在句子边界断开）

**扩展思考**：
- **问题 1**：哪种分块器更适合你的实际文档？
  - 当前文档为 Markdown 格式，具有明确的结构化语义（标题层级、列表、代码块等）
  - `RecursiveCharacterTextSplitter` 仅基于通用分隔符（`\n\n`、`\n` 等）切分，无法识别 Markdown 语法结构
  - 建议使用 `MarkdownHeaderTextSplitter`，它能按 `# ## ###` 等标题层级进行语义感知的切分

- **问题 2**：为什么递归版本可能会产生更多或更少的块？
  - `RecursiveCharacterTextSplitter` 采用**多级回退策略**：依次尝试 `\n\n`、`\n`、` ` 等分隔符
  - Markdown 文档中换行符密集，导致在更细粒度上切分，块数量可能**更多**
  - 与 `CharacterTextSplitter`（固定字符数硬切）相比，块大小分布更不均匀

---

## 相关知识点

| 概念 | 解释 |
|------|------|
| **Chunk Size** | 每个文本块的目标字符数，越小检索越精准但上下文丢失 |
| **Chunk Overlap** | 相邻块之间的重叠字符数，用于保持上下文连续性 |
| **分隔符** | 文本断开的位置（`\n\n`, `\n`, `。`, `，` 等） |
| **硬切断** | 找不到分隔符时，直接在指定字符数处断开，可能切断句子 |
| **Document 对象** | LangChain 中表示文档的类，包含 `page_content` 和 `metadata` |

## 进一步探索

1. 尝试修改 `separator` 参数自定义分隔符
2. 了解 `length_function` 参数如何影响字符计数
3. 研究如何保存分块后的元数据（如来源文件、行号等）

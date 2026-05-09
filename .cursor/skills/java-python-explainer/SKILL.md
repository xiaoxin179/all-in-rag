---
name: java-python-explainer
description: 用 Java 思路解释 Python 代码，为 Java 开发者提供等价的 Java 代码对比。适用于用户选中 Python 代码并要求解释、或询问两个语言的对比时使用。
---

# Java-Python 代码解释器

当用户选中 Python 代码并要求解释时，使用本技能。

## 核心原则

1. **始终提供 Java 等价代码**：让 Java 开发者看到熟悉的语法
2. **标注语法对应关系**：Python 语法 → Java 等价写法
3. **使用表格对比**：复杂概念用表格呈现更清晰
4. **保持简洁**：只解释用户选中的代码片段

## 常用对应关系速查

### 数据类型

| Python | Java |
|--------|------|
| `x: int = 10` | `int x = 10;` |
| `x: str = "hello"` | `String x = "hello";` |
| `x: list = [1, 2, 3]` | `List<Integer> x = Arrays.asList(1, 2, 3);` |
| `x: dict = {"a": 1}` | `Map<String, Integer> x = new HashMap<>();` |
| `x: bool = True` | `boolean x = true;` |
| `x: float = 1.5` | `double x = 1.5;` |

### 流程控制

| Python | Java |
|--------|------|
| `if x > 0:` | `if (x > 0) {` |
| `elif x < 0:` | `} else if (x < 0) {` |
| `else:` | `} else {` |
| `for i in range(10):` | `for (int i = 0; i < 10; i++) {` |
| `for item in list:` | `for (Item item : list) {` |
| `while True:` | `while (true) {` |

### 函数定义

| Python | Java |
|--------|------|
| `def func(x, y):` | `public ReturnType func(Type1 x, Type2 y) {` |
| `return x + y` | `return x + y; }` |
| `def func(x: int) -> int:` | `public int func(int x) {` |

### 类定义

| Python | Java |
|--------|------|
| `class Dog:` | `public class Dog {` |
| `def __init__(self, name):` | `public Dog(String name) {` |
| `self.name = name` | `this.name = name;` |
| `def bark(self):` | `public void bark() {` |
| `print("woof")` | `System.out.println("woof");` |

### 特殊语法

| Python | Java |
|--------|------|
| `with open("file") as f:` | `try (FileReader f = new FileReader("file")) {` |
| `list comprehension` | `stream API` 或 `for` 循环 |
| `@decorator` | 无直接等价（用 AOP/注解模拟） |
| `*args, **kwargs` | `Object... args` / `Map<String, Object>` |
| `lambda x: x * 2` | `(x) -> x * 2` 或 `x -> x * 2` |

## 解释流程

### 1. 识别代码类型

```
函数定义 → 找出参数和返回值
类定义 → 找出属性和方法
库调用 → 找出库的职责
```

### 2. 逐行翻译

```python
# 示例 Python 代码
from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()
```

**翻译为：**

```java
// 导入语句 = Java 的 import
import com.langchain.community.loaders.UnstructuredMarkdownLoader;

// 创建加载器实例 = new 对象
UnstructuredMarkdownLoader loader = new UnstructuredMarkdownLoader(markdownPath);

// 调用 load() 方法 = 调用实例方法
List<Document> docs = loader.load();
```

### 3. 解释关键概念

对于不熟悉的 Python 概念，用 Java 类比解释：

| Python 概念 | Java 解释 |
|-------------|-----------|
| `self` | 等价于 Java 的 `this` |
| `__init__` | 等价于构造方法 |
| `None` | 等价于 `null` |
| `doc.page_content` | 类似 Map 的 `get("page_content")` |
| 切片 `list[1:3]` | `list.subList(1, 3)` |

## 输出格式模板

```markdown
## 代码解释

**Python 原始代码：**
```python
# 用户选中的代码
```

**Java 等价代码：**
```java
// Java 写法
```

**逐行说明：**

| 行号 | Python | Java 等价 | 说明 |
|------|--------|-----------|------|
| 1 | `x = 10` | `int x = 10;` | 变量定义 |

**关键概念：**
- 概念1: Java 中的等价解释
```

## 注意事项

- 只解释用户选中的代码，不要过度展开
- 如果代码涉及多个文件/模块，聚焦当前选中部分
- 对于 Python 特有的语法（如装饰器、生成器），明确说明 Java 无直接等价
- 保持解释简洁，避免引入过多额外概念

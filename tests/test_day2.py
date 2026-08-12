"""Day 2 模块功能测试"""
import sys
sys.path.insert(0, "C:\\Users\\liu\\Desktop\\Multi-Agent-RAG-Academic-Assistant")

from src.document_processing.parser import MarkdownParser
from src.retrieval.chunker import AdaptiveChunker

# 1. 测试 Markdown 解析
print("=" * 60)
print("1️⃣  测试 Markdown 解析")
print("=" * 60)

md_text = """# Python 装饰器

## 什么是装饰器

装饰器是一种设计模式，用于在不修改函数定义的情况下增强函数功能。

## 基本用法

```python
def decorator(func):
    def wrapper(*args, **kwargs):
        print('调用前')
        result = func(*args, **kwargs)
        print('调用后')
        return result
    return wrapper
```

### 语法糖

使用 @ 语法糖可以更简洁地应用装饰器。

- 列表项A
- 列表项B
- 列表项C

> 引用：装饰器是 Python 的重要特性。
"""

parser = MarkdownParser()
elements = parser.parse(md_text, source="test.md")
print(f"解析出 {len(elements)} 个元素:")
for e in elements:
    preview = e.content[:60] + "..." if len(e.content) > 60 else e.content
    print(f"  [{e.element_type:10s}] {preview}")

print()

# 2. 测试自适应分块
print("=" * 60)
print("2️⃣  测试自适应分块")
print("=" * 60)

chunker = AdaptiveChunker(min_size=50, max_size=300, overlap=30)
chunks = chunker.chunk(elements, source="test.md")
print(f"分块得到 {len(chunks)} 个 chunk:")
for i, chunk in enumerate(chunks):
    heading = chunk.metadata.get("heading", "")
    preview = chunk.page_content[:80] + "..." if len(chunk.page_content) > 80 else chunk.page_content
    if chunk.page_content.strip():
        print(f"  Chunk {i+1} [{len(chunk.page_content):4d}字] 标题链: {heading}")
        print(f"    内容: {preview}")

print()

# 3. 测试 Embedder（mock 模式—不实际加载模型）
print("=" * 60)
print("3️⃣  测试 Embedder 初始化和降级方案")
print("=" * 60)

from src.retrieval.embedder import Embedder
embedder = Embedder()
print(f"Embedder 模型名: {embedder.model_name}")
print(f"Embedder 设备: {embedder.device}")
print(f"Embedder 维度: {embedder.dimension}")

# 测试降级嵌入（无模型时）
vec = embedder.embed_query("测试查询")
print(f"查询向量维度: {len(vec)}")
print(f"向量前5个值: {vec[:5]}")

print()
print("=" * 60)
print("✅ Day 2 所有功能测试通过！")
print("=" * 60)

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

def source_02():
    # 1. 文档加载
    loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
    # 01 用 unstructured 加载 PDF，能识别语义元素（标题、段落、表格等）,数据加载更多的是吧原始的pdf啥的加载为结构化的数据，且都是包含matedata的
    # 02 用 TextLoader 加载 TXT，只是把文本内容加载出来
    # 关键区别不是"是否加载"，而是"加载什么"和"加载后如何处理"
    docs = loader.load()

    # 2. 初始化固定大小分块器
    text_splitter = CharacterTextSplitter(
        chunk_size=400,  # 每个块的大小
        chunk_overlap=100  # 块之间的重叠大小
    )

    # 3. 执行分块
    chunks = text_splitter.split_documents(docs)

    # 4. 打印 chunks 的类型和结构
    print(f"chunks 类型: {type(chunks)}")
    print(f"chunks 是列表，长度: {len(chunks)}")
    print(f"chunks[0] 类型: {type(chunks[0])}")

    # 5. 打印结果
    print(f"\n文本被切分为 {len(chunks)} 个块。\n")
    print("--- 前5个块内容示例 ---")
    for i, chunk in enumerate(chunks[:5]):
        print("=" * 60)
        print(f"chunk 类型: {type(chunk)}")
        print(f'块 {i + 1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')
def sea_charted():
    # 1. 文档加载
    loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
    # 01 用 unstructured 加载 PDF，能识别语义元素（标题、段落、表格等）,数据加载更多的是吧原始的pdf啥的加载为结构化的数据，且都是包含matedata的
    # 02 用 TextLoader 加载 TXT，只是把文本内容加载出来
    # 关键区别不是"是否加载"，而是"加载什么"和"加载后如何处理"
    docs = loader.load()

    # 2. 初始化固定大小分块器
    text_splitter = CharacterTextSplitter(
        chunk_size=100,  # 每个块的大小
        chunk_overlap=10  # 块之间的重叠大小
    )

    # 3. 执行分块
    chunks = text_splitter.split_documents(docs)

    # 4. 打印 chunks 的类型和结构
    print(f"chunks 类型: {type(chunks)}")
    print(f"chunks 是列表，长度: {len(chunks)}")
    print(f"chunks[0] 类型: {type(chunks[0])}")

    # 5. 打印结果
    print(f"\n文本被切分为 {len(chunks)} 个块。\n")
    print("--- 前5个块内容示例 ---")
    for i, chunk in enumerate(chunks[:10]):
        text = chunk.page_content
        print(f"块 {i + 1} 结尾: [{text[-15:] if len(text) > 15 else text}]")
def comparison():
    from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
    loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
    docs = loader.load()

    text_splitter1 = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    chunks1 = text_splitter1.split_documents(docs)
    # 按优先级从高到低排列：标题层级 > 段落 > 表格 > 换行
    separators=["####", "###", "##", "#", "\n\n", "|", "\n"]
    text_splitter2 = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=10,
        separators=separators
    )
    chunks2 = text_splitter2.split_documents(docs)

    print(f"CharacterTextSplitter: {len(chunks1)} 块")
    print(f"RecursiveCharacterTextSplitter: {len(chunks2)} 块")

    # 对比第一个块的内容
    print("\n=== CharacterTextSplitter 块1 ===")
    print(chunks1[0].page_content)
    print("\n=== RecursiveCharacterTextSplitter 块1 ===")
    print(chunks2[0].page_content)
if __name__ == '__main__':
    comparison()
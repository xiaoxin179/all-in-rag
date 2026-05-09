import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter() #分块器实例
chunks = text_splitter.split_documents(docs)

# 诊断：检查分块情况
total_chars = sum(len(chunk.page_content) for chunk in chunks)
print(f"原始文档总字符数: {len(docs[0].page_content)}")
print(f"分块后总字符数: {total_chars}")
print(f"丢失字符数: {len(docs[0].page_content) - total_chars}")
print(f"各块字符数: {[len(chunk.page_content) for chunk in chunks]}\n")

# 输出分块结果
print(f"chunks 类型: {type(chunks)}，元素类型: {type(chunks[0]) if chunks else 'N/A'}")
print(f"共分成 {len(chunks)} 个块：\n")
for i, chunk in enumerate(chunks):
    print(f"=== 块 {i+1} ===")
    print(f"字符数: {len(chunk.page_content)}")
    print(chunk.page_content)
    print()  # 空行分隔
    print()

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
result = vectorstore.add_documents(chunks)
print(f"vectorstore 类型: {type(vectorstore)}")
print(f"add_documents 返回值类型: {type(result)}\n")

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型

# 使用 AIHubmix
llm = ChatOpenAI(
    model="gpt-5.5-free",
    temperature=0.6,
    max_tokens=4096,
    api_key=os.getenv("AI-HUB-KEY"),
    base_url="https://aihubmix.com/v1"
)

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer.content)

# 打印 token 消耗统计
usage = answer.response_metadata.get('token_usage', {})
print(f"\n=== Token 消耗统计 ===")
print(f"输入 tokens: {usage.get('prompt_tokens', 'N/A')}")
print(f"输出 tokens: {usage.get('completion_tokens', 'N/A')}")
print(f"总 tokens: {usage.get('total_tokens', 'N/A')}")

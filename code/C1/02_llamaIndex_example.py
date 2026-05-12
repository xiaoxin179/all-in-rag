import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 配置 HuggingFace 镜像（解决国内访问超时问题）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 环境初始化
load_dotenv()

# 配置 Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 配置 LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.6,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 直接读取 markdown 文件
with open("../../data/C1/markdown/easy-rl-chapter1.md", "r", encoding="utf-8") as f:
    content = f.read()

# 简单分块（按段落）
paragraphs = content.split("\n\n")
chunks = [Document(page_content=p.strip()) for p in paragraphs if p.strip()]

# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知："抱歉，我无法根据提供的上下文找到相关信息来回答此问题。"

上下文:
{context}

问题: {question}

回答:""")

# 查询
question = "文中举了哪些例子?,请告诉我例子的内容是什么?"
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 获取回答
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer.content)

import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. 示例文本和嵌入模型
texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]
docs = [Document(page_content=t) for t in texts]
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
local_faiss_path = "./faiss_index_store"
# 2. 如果本地有的化就直接用本地embedding的结果

if os.path.exists(local_faiss_path):
    print("加载已有索引...")
    vectorstore = FAISS.load_local(
        local_faiss_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("创建新索引（首次需要 embedding）...")
    docs = [Document(page_content=t) for t in texts]
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(local_faiss_path)

print(f"FAISS index has been saved to {local_faiss_path}")

# 3. 加载索引并执行查询
# 加载时需指定相同的嵌入模型，并允许反序列化
loaded_vectorstore = FAISS.load_local(
    local_faiss_path,
    embeddings,
    allow_dangerous_deserialization=True
)

# 执行相似性搜索
query = "FAISS？"
results = loaded_vectorstore.similarity_search(query, k=1)

print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for doc in results:
    print(f"- {doc.page_content}")
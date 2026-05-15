from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import load_index_from_storage
from llama_index.core.storage import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import QueryBundle

# 持久化索引路径
PERSIST_PATH = "./llamaindex_index_store"

# 1. 配置全局嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")


def query_index(query_text: str, persist_path: str = PERSIST_PATH):
    """
    使用本地持久化的索引进行查询，返回相似度最高的一条结果

    Args:
        query_text: 查询条件
        persist_path: 索引持久化目录路径
    """
    storage_context = StorageContext.from_defaults(persist_dir=persist_path)
    index = load_index_from_storage(storage_context, embed_model=Settings.embed_model)

    retriever = VectorIndexRetriever(index=index, similarity_top_k=1)
    nodes = retriever.retrieve(QueryBundle(query_text))

    print(f"查询: {query_text}")
    print(f"最相似结果: {nodes[0].text}")
    return nodes[0].text


# 2. 创建示例文档
texts = [
    "张三是法外狂徒",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "它提供了数据连接、索引和查询接口等工具。"
]
docs = [Document(text=t) for t in texts]

# 3. 创建索引并持久化到本地
index = VectorStoreIndex.from_documents(docs)
persist_path = "./llamaindex_index_store"
index.storage_context.persist(persist_dir=persist_path)
print(f"LlamaIndex 索引已保存至: {persist_path}")

if __name__ == '__main__':
    # 示例查询
    query_index("法外狂徒是谁")

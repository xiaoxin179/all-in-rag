import json              # 导入标准库：JSON序列化/反序列化，用于读取 dragon.json 数据文件
import os                # 导入标准库：操作系统相关功能，这里用于检查数据文件是否存在
from pathlib import Path  # 导入标准库：Path对象，提供更直观的文件路径操作
import numpy as np        # 导入数值计算库：矩阵运算，向量归一化时用到
import torch             # 导入深度学习框架：SigLIP模型加载和推理
from transformers import AutoModel, AutoProcessor  # 从HuggingFace Transformers导入模型和处理器自动加载工具
from sklearn.feature_extraction.text import TfidfVectorizer  # Sklearn的TF-IDF向量化工具，生成稀疏向量
from scipy.sparse import csr_matrix  # Scipy稀疏矩阵格式，用于处理和转换TF-IDF输出
# 导入Milvus向量数据库相关组件：连接管理、客户端、字段定义、数据类型、集合操作、ANN搜索请求、RRF排序器
from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, AnnSearchRequest, RRFRanker

# 类解析
# ：v1 用 BGE-M3 一站式搞定，代码更简洁；v2 用 SigLIP + TF-IDF 分离实现，展示更清晰，适合学习理解混合检索的原理。
# 1. 初始化设置

COLLECTION_NAME = "dragon_siglip_demo"   # Milvus中的集合名称，用于存储龙主题的混合检索数据
MILVUS_URI = "http://localhost:19530"    # Milvus服务器地址，19530是Milvus默认端口；也可以换成 "./milvus_demo.db" 文件模式
DATA_PATH = "../../data/C4/metadata/dragon.json"  # 相对路径指向测试数据文件，..表示从code/C4回到项目根目录
BATCH_SIZE = 50   # 数据处理的批大小，这里实际未在嵌入生成时使用（嵌入用8），用于示意分批概念

# 2. 自定义SigLIP嵌入函数类
# 封装SigLIP模型和TF-IDF，提供统一的encode接口，兼容Milvus混合检索对两种向量的需求
class SigLIPEmbeddingFunction:
    def __init__(self, model_name="google/siglip-base-patch16-256-multilingual", device="cpu"):
        """
        初始化SigLIP嵌入函数
        Args:
            model_name: SigLIP模型名称或本地路径（默认HuggingFace上的多语言版本）
            device: 设备类型 ("cpu" 或 "cuda")；没有GPU时用cpu，推理速度较慢
        """
        self.model_name = model_name
        self.device = device

        # 支持本地路径：如果路径存在则使用本地模型
        # __file__是当前脚本路径，.parent取上级目录（code/C4的父目录即项目根目录）
        # 这样设计是为了优先使用本地缓存模型，避免重复下载，也能离线运行
        local_model_path = Path(__file__).parent.parent / "models" / "siglip-base-patch16-256-multilingual"
        if local_model_path.exists():
            self.model_path = str(local_model_path)
            print(f"--> 使用本地 SigLIP 模型: {self.model_path}")   # SigLIP密集向量模型
        else:
            self.model_path = model_name  # 回退到HuggingFace下载
            print(f"--> 正在从 HuggingFace 加载 SigLIP 模型: {model_name}")

        # 加载预训练模型和对应的数据处理器（processor包含tokenizer和图像预处理）
        self.model = AutoModel.from_pretrained(self.model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model.to(device)      # 将模型移到指定设备（CPU/GPU）
        self.model.eval()         # 切换到评估模式，禁用dropout等训练专属层，加速推理

        # 初始化TF-IDF作为稀疏向量生成器
        # max_features=10000: 限制词汇表大小，防止维度爆炸和内存过大
        # stop_words='english': 移除英文停用词（如the, a, is），减少噪音
        # ngram_range=(1, 2): 同时捕获单词和双词组合，如"dragon"和"fire dragon"，增强短语匹配能力
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,  # 限制词汇表大小以节省空间
            stop_words='english',  # 过滤英文停用词，保留中文为主的检索词
            ngram_range=(1, 2)   # 捕获单词和双词组合，增强短语表达能力
        )
        self.tfidf_fitted = False  # 标记TF-IDF是否已完成fit，防止在未学习语料的情况下transform

        # 获取文本编码器的输出维度
        # 用一个虚拟文本过一次前向传播来确定向量维度，Milvus建索引时需要提前知道维度
        with torch.no_grad():
            dummy_text = ["test"]
            inputs = self.processor(text=dummy_text, padding="max_length", return_tensors="pt")
            # pixel_values是图像输入特有的key，这里只做文本所以排除掉
            outputs = self.model.text_model(**{k: v.to(device) for k, v in inputs.items() if k != 'pixel_values'})
            self.dense_dim = outputs.pooler_output.shape[-1]  # pooler_output的最后一个维度就是向量维度

        print(f"--> SigLIP 模型加载完成。密集向量维度: {self.dense_dim}")

    @property
    def dim(self):
        """返回维度信息，兼容原BGE-M3接口"""
        # 提供统一的维度查询接口，返回一个字典包含dense和sparse两个维度的值
        # 如果TF-IDF还未fit，则用默认值10000作为占位
        return {
            "dense": self.dense_dim,
            "sparse": self.tfidf_vectorizer.max_features if self.tfidf_fitted else 10000
        }

    def fit_sparse(self, docs):
        """拟合稀疏向量模型（TF-IDF）"""
        # TF-IDF需要先在语料上fit学习词汇表和IDF值，然后再transform
        # fit只需在数据入库时调用一次，检索时不需要重新fit
        print("--> 正在拟合 TF-IDF 模型...")
        self.tfidf_vectorizer.fit(docs)
        self.tfidf_fitted = True
        print(f"--> TF-IDF 模型拟合完成。词汇表大小: {len(self.tfidf_vectorizer.vocabulary_)}")

    def encode_text_dense(self, texts):
        """使用SigLIP编码文本为密集向量"""
        # 单个字符串转为单元素列表的兼容处理，统一后续处理逻辑
        if isinstance(texts, str):
            texts = [texts]

        dense_vectors = []
        batch_size = 8  # 减小批次大小以节省内存；CPU模式下8已足够，GPU可适当增大

        with torch.no_grad():  # 推理阶段禁用梯度计算，节省显存和计算时间
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.processor(text=batch_texts, padding="max_length", truncation=True, return_tensors="pt")
                # processor返回的dict中包含input_ids、attention_mask等，排除pixel_values（图像用）
                inputs = {k: v.to(self.device) for k, v in inputs.items() if k != 'pixel_values'}

                outputs = self.model.text_model(**inputs)  # 过文本编码器得到hidden states
                embeddings = outputs.pooler_output  # 取[CLS] token的输出作为整个句子的表示向量

                # 归一化向量：p=2表示L2归一化，dim=1在行维度上操作
                # 归一化后余弦相似度等价于内积，Milvus用IP（内积）度量时结果更稳定
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                dense_vectors.extend(embeddings.cpu().numpy())  # 转回CPU numpy数组以兼容Milvus

        return np.array(dense_vectors)

    def encode_text_sparse(self, texts):
        """使用TF-IDF编码文本为稀疏向量"""
        # 必须先fit过词汇表才能transform，否则抛出明确错误而不是静默返回错误结果
        if not self.tfidf_fitted:
            raise ValueError("请先调用 fit_sparse() 方法拟合TF-IDF模型")

        if isinstance(texts, str):
            texts = [texts]

        # transform用已学习的词汇表将文本转为稀疏向量，fit_transform不同这里不会重新学习
        sparse_matrix = self.tfidf_vectorizer.transform(texts)
        return sparse_matrix

    def __call__(self, texts):
        """主调用方法，返回密集和稀疏向量"""
        # 这是类的核心入口，使得ef(docs)这样的调用成为可能（类似函数式调用）
        # 隐藏了内部两个模型分别处理的细节，对外只暴露一个接口
        if isinstance(texts, str):
            texts = [texts]

        # 如果还没有拟合稀疏模型，先拟合
        # 自动完成fit_sparse，避免外部遗漏调用导致后续出错
        if not self.tfidf_fitted:
            self.fit_sparse(texts)

        dense_vectors = self.encode_text_dense(texts)    # SigLIP生成语义密集向量
        sparse_vectors = self.encode_text_sparse(texts)  # TF-IDF生成关键词稀疏向量

        return {
            "dense": dense_vectors,
            "sparse": sparse_vectors
        }

# 3. 连接 Milvus 并初始化嵌入模型
# 打印连接信息便于调试，确认Milvus服务是否正常响应
print(f"--> 正在连接到 Milvus: {MILVUS_URI}")
connections.connect(uri=MILVUS_URI)  # 建立到Milvus服务器的持久连接

print("--> 正在初始化 SigLIP 嵌入模型...")
ef = SigLIPEmbeddingFunction(device="cpu")  # 如果有GPU可以改为"cuda"以加速推理

# 4. 创建 Collection
milvus_client = MilvusClient(uri=MILVUS_URI)  # MilvusClient提供高级API封装，适合普通操作
if milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在删除已存在的 Collection '{COLLECTION_NAME}'...")
    milvus_client.drop_collection(COLLECTION_NAME)  # 删除旧集合以确保干净的实验环境

# 定义集合的Schema，每个FieldSchema对应一列字段
fields = [
    # pk: 主键，自增ID，Milvus自动生成，VARCHAR类型最大100字符
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),   # 图片唯一标识
    FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256),     # 文件路径
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),    # 标题
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),  # 描述，允许较长文本
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),  # 类别
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128), # 地点
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),  # 环境
    # 稀疏向量字段：类型必须是SPARSE_FLOAT_VECTOR，Milvus会自动选择稀疏索引
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    # 密集向量字段：维度从ef.dim["dense"]获取（由模型决定），float类型向量
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"])
]

# 如果集合不存在，则创建它及索引
if not milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在创建 Collection '{COLLECTION_NAME}'...")
    schema = CollectionSchema(fields, description="使用SigLIP的龙混合检索示例")
    # 创建集合：consistency_level="Strong"保证最强一致性，所有副本读取相同数据
    collection = Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Strong")
    print("--> Collection 创建成功。")

    # 5. 创建索引
    print("--> 正在为新集合创建索引...")
    # 稀疏向量索引：SPARSE_INVERTED_INDEX专为稀疏高维向量设计，适合TF-IDF输出
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    collection.create_index("sparse_vector", sparse_index)
    print("稀疏向量索引创建成功。")

    # 密集向量索引：AUTOINDEX让Milvus自动选择最优索引类型，比手动指定更省心
    dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    collection.create_index("dense_vector", dense_index)
    print("密集向量索引创建成功。")

collection = Collection(COLLECTION_NAME)  # 获取Collection实例用于后续操作

# 6. 加载数据并插入
collection.load()  # 将集合加载到内存中，才能进行搜索操作
print(f"--> Collection '{COLLECTION_NAME}' 已加载到内存。")

if collection.is_empty:  # 检查集合是否为空，只在空时插入数据
    print(f"--> Collection 为空，开始插入数据...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")  # 数据文件不存在则报错终止
    with open(DATA_PATH, 'r', encoding='utf-8') as f:  # 以UTF-8编码打开JSON文件
        dataset = json.load(f)  # 解析JSON为Python列表

    docs, metadata = [], []
    for item in dataset:
        # 将多个字段拼接成一个长字符串作为文档文本，用于生成统一的向量表示
        # 这样做可以让向量同时包含标题、描述、地点等多方面信息
        parts = [
            item.get('title', ''),         # 标题字段
            item.get('description', ''),   # 描述字段
            item.get('location', ''),       # 地点字段
            item.get('environment', ''),   # 环境字段
            # *item.get('combat_details', {}).get('combat_style', []),  # 战斗风格，已注释掉
            # *item.get('combat_details', {}).get('abilities_used', []), # 战斗技能，已注释掉
            # item.get('scene_info', {}).get('time_of_day', '')          # 时间场景，已注释掉
            # 注释掉的字段是为了简化处理，完整版可取消注释包含更多信息
        ]
        # filter(None)移除空字符串，' '.join将多段文本用空格连接成一个长字符串
        docs.append(' '.join(filter(None, parts)))
        metadata.append(item)  # 保留原始完整字典用于后续提取字段值
    print(f"--> 数据加载完成，共 {len(docs)} 条。")

    print("--> 正在生成向量嵌入...")
    # ef(docs)调用__call__方法，同时生成SigLIP密集向量和TF-IDF稀疏向量
    embeddings = ef(docs)
    print("--> 向量生成完成。")

    print("--> 正在分批插入数据...")
    # 从metadata列表中提取每个字段的列表，用于Milvus的批量插入
    img_ids = [doc["img_id"] for doc in metadata]
    paths = [doc["path"] for doc in metadata]
    titles = [doc["title"] for doc in metadata]
    descriptions = [doc["description"] for doc in metadata]
    categories = [doc["category"] for doc in metadata]
    locations = [doc["location"] for doc in metadata]
    environments = [doc["environment"] for doc in metadata]

    # 获取向量 - 注意SigLIP返回的格式与BGE-M3不同
    # BGE-M3返回可直接插入的格式，SigLIP+TF-IDF需要手动转换
    sparse_vectors = []
    dense_vectors = embeddings["dense"].tolist()  # 密集向量直接转list即可

    # 将稀疏矩阵转换为Milvus可接受的格式
    # Milvus的SPARSE_FLOAT_VECTOR需要{索引: 值}的字典格式，而不是scipy的稀疏矩阵对象
    sparse_matrix = embeddings["sparse"]
    for i in range(sparse_matrix.shape[0]):
        row = sparse_matrix.getrow(i)  # 获取第i行的稀疏向量（scipy sparse matrix格式）
        # 创建稀疏向量字典格式
        sparse_dict = {}
        for j in range(row.nnz):  # nnz是非零元素个数
            sparse_dict[row.indices[j]] = float(row.data[j])  # 索引->值 存入字典
        sparse_vectors.append(sparse_dict)

    # 插入数据：按fields定义顺序提供数据列表，Milvus会自动匹配字段
    collection.insert([
        img_ids,       # 对应img_id字段
        paths,         # 对应path字段
        titles,        # 对应title字段
        descriptions,  # 对应description字段
        categories,    # 对应category字段
        locations,     # 对应location字段
        environments,  # 对应environment字段
        sparse_vectors,  # 对应sparse_vector字段
        dense_vectors    # 对应dense_vector字段
    ])

    collection.flush()  # 强制将内存中的数据写入磁盘，确保数据持久化
    print(f"--> 数据插入完成，总数: {collection.num_entities}")
else:
    print(f"--> Collection 中已有 {collection.num_entities} 条数据，跳过插入。")

# 7. 执行搜索
search_query = "悬崖上的巨龙"   # 中文搜索查询，测试语义理解能力（悬崖、巨龙）
search_filter = 'category in ["western_dragon", "chinese_dragon", "movie_character"]'  # 过滤条件，只搜索特定类别的龙
top_k = 5   # 返回最相似的5个结果

print(f"\n{'='*20} 开始混合搜索 {'='*20}")
print(f"查询: '{search_query}'")
print(f"过滤器: '{search_filter}'")

# 生成查询向量：同样用ef生成两种向量，保持与入库时一致的编码方式
query_embeddings = ef([search_query])
dense_vec = query_embeddings["dense"][0].tolist()  # 取第一个（也是唯一）结果转list

# 处理稀疏向量：转换为Milvus期望的字典格式（与插入时的转换逻辑一致）
sparse_matrix = query_embeddings["sparse"]
sparse_row = sparse_matrix.getrow(0)
sparse_dict = {}
for j in range(sparse_row.nnz):
    sparse_dict[sparse_row.indices[j]] = float(sparse_row.data[j])

# 打印向量信息：帮助理解向量生成结果，便于调试
print("\n=== 向量信息 ===")
print(f"密集向量维度: {len(dense_vec)}")
print(f"密集向量前5个元素: {dense_vec[:5]}")
print(f"密集向量范数: {np.linalg.norm(dense_vec):.4f}")  # 验证是否为1（已归一化）

print(f"\n稀疏向量维度: {sparse_matrix.shape[1]}")  # 词汇表大小
print(f"稀疏向量非零元素数量: {sparse_row.nnz}")  # 非零词数
print("稀疏向量前5个非零元素:")
for i, (idx, val) in enumerate(list(sparse_dict.items())[:5]):
    print(f"  - 索引: {idx}, 值: {val:.4f}")
density = (sparse_row.nnz / sparse_matrix.shape[1] * 100)
print(f"\n稀疏向量密度: {density:.8f}%")  # 极低密度说明TF-IDF的稀疏特性

# 定义搜索参数：IP（内积）作为相似度度量，归一化后等价于余弦相似度
search_params = {"metric_type": "IP", "params": {}}

# 先执行单独的搜索：分别测试两种检索的效果，观察差异
print("\n--- [单独] 密集向量搜索结果 ---")
dense_results = collection.search(
    [dense_vec],               # 查询向量列表（必须是list）
    anns_field="dense_vector",  # 指定在哪个向量字段上搜索
    param=search_params,       # 搜索参数
    limit=top_k,               # 返回结果数量
    expr=search_filter,        # 过滤表达式
    # 指定返回时包含哪些标量字段，搜索结果中也会返回这些字段的值
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]  # search返回的是list of list，外层list对应多个查询向量，这里只有一个所以取[0]

for i, hit in enumerate(dense_results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")  # 只显示前100字符避免过长

print("\n--- [单独] 稀疏向量搜索结果 ---")
sparse_results = collection.search(
    [sparse_dict],              # 稀疏向量用字典格式传入
    anns_field="sparse_vector",
    param=search_params,
    limit=top_k,
    expr=search_filter,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

for i, hit in enumerate(sparse_results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

print("\n--- [混合] 稀疏+密集向量搜索结果 ---")
# 创建 RRF 融合器：Reciprocal Rank Fusion，通过排名融合两个检索结果
# k=60是RRF公式中的参数，较大的k会减少排名差异的影响；默认60是常用值
rerank = RRFRanker(k=60)

# 创建搜索请求：AnnSearchRequest封装了单个向量字段的搜索请求
dense_req = AnnSearchRequest([dense_vec], "dense_vector", search_params, limit=top_k)
sparse_req = AnnSearchRequest([sparse_dict], "sparse_vector", search_params, limit=top_k)

# 执行混合搜索：同时搜索两个向量字段，用RRF融合结果
results = collection.hybrid_search(
    [sparse_req, dense_req],  # 搜索请求列表，顺序不影响最终结果
    rerank=rerank,             # 指定排序器
    limit=top_k,              # 返回融合后的top_k结果
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

# 打印最终结果
for i, hit in enumerate(results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

# 8. 清理资源
# 释放集合占用的内存，防止长时间运行时内存泄漏
milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")
milvus_client.drop_collection(COLLECTION_NAME)  # 删除集合，清理磁盘空间
print(f"已删除 Collection: '{COLLECTION_NAME}'")

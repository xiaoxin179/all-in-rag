import json
import os
from tqdm import tqdm
from glob import glob
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
# 多模态龙搜索，，单模态版本
@dataclass
class DragonImage:
    """龙类图像数据类"""
    img_id: str
    path: str
    title: str
    description: str
    category: str
    location: str
    environment: str
    combat_details: Dict[str, Any] = None    # 战斗细节，风格等
    scene_info: Dict[str, Any] = None    #使用的技能

class DragonDataset:
    """龙类图像数据集管理类"""
    def __init__(self, data_dir: str, metadata_path: str):
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.images: List[DragonImage] = []
        self._load_metadata()
    
    def _load_metadata(self):
        """加载图像元数据"""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for img_data in data:
                # 确保图片路径是完整的
                if not img_data['path'].startswith(self.data_dir):
                    img_data['path'] = os.path.join(self.data_dir, img_data['path'].split('/')[-1])
                self.images.append(DragonImage(**img_data))
    
    def get_image_paths(self) -> List[str]:
        """获取所有图像路径"""
        return [img.path for img in self.images]
    
    def get_metadata_by_path(self, path: str) -> DragonImage:
        """根据路径获取元数据"""
        for img in self.images:
            if img.path == path:
                return img
        return None
    
    def get_text_content(self, img: DragonImage) -> str:
        """获取图像的文本描述内容"""
        parts = [
            img.title, img.description,
            img.location, img.environment
        ]
        if img.combat_details:
            parts.extend(img.combat_details.get('combat_style', []))
            parts.extend(img.combat_details.get('abilities_used', []))
        if img.scene_info:
            parts.append(img.scene_info.get('time_of_day', ''))
        return ' '.join(filter(None, parts))

class Encoder:
    """编码器类，用于将图像和文本编码为向量"""
    def __init__(self, model_name: str, model_path: str, use_local_bge_m3: bool = True):
        # 初始化 Visual-BGE 模型（用于多模态）
        self.visual_model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.visual_model.eval()

        # 初始化 BGE-M3 模型（用于混合检索），优先使用本地模型
        if use_local_bge_m3:
            local_bge_m3_path = Path(__file__).parent.parent.parent / "models" / "bge-m3"
            if local_bge_m3_path.exists():
                print(f"--> 使用本地 BGE-M3 模型: {local_bge_m3_path}")
                self.bge_m3 = BGEM3EmbeddingFunction(
                    model_name=str(local_bge_m3_path),
                    use_fp16=False,
                    device="cpu"
                )
                print(f"BGE-M3 密集向量维度: {self.bge_m3.dim['dense']}")
            else:
                print(f"⚠️ 本地 BGE-M3 模型不存在: {local_bge_m3_path}，文本混合检索将被禁用")
                self.bge_m3 = None
        else:
            self.bge_m3 = None

    def encode_query(self, image_path: str = None, text: str = None, mode: str = "multimodal") -> list[float] | dict:
        """编码查询（支持图像+文本或仅文本），mode 可选 'multimodal' 或 'hybrid'"""
        with torch.no_grad():
            if mode == "multimodal":
                if image_path and text:
                    query_emb = self.visual_model.encode(image=image_path, text=text)
                elif image_path:
                    query_emb = self.visual_model.encode(image=image_path)
                elif text:
                    query_emb = self.visual_model.encode(text=text)
                else:
                    raise ValueError("必须提供图像路径或文本内容")
                return query_emb.tolist()[0]
            elif mode == "hybrid" and self.bge_m3 and text:
                # 使用 BGE-M3 返回稀疏+密集向量
                embeddings = self.bge_m3([text])
                return {
                    'sparse': embeddings["sparse"]._getrow(0),
                    'dense': embeddings["dense"][0]
                }
            else:
                raise ValueError(f"无效的 mode '{mode}' 或 BGE-M3 模型未初始化")

    def encode_multimodal(self, image_path: str, text: str) -> list[float]:
        """编码多模态内容（图像+文本）"""
        with torch.no_grad():
            query_emb = self.visual_model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]

    def encode_text_hybrid(self, text: str) -> dict:
        """使用 BGE-M3 编码文本，返回稀疏和密集向量"""
        if not self.bge_m3:
            raise ValueError("BGE-M3 模型未初始化，无法使用混合编码")
        embeddings = self.bge_m3([text])
        return {
            'sparse': embeddings["sparse"]._getrow(0),
            'dense': embeddings["dense"][0]
        }

def visualize_results(query_image_path: str, retrieved_results: list, img_height: int = 300, img_width: int = 300, row_count: int = 3) -> np.ndarray:
    """从检索到的结果创建一个全景图用于可视化"""
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # 处理查询图像
    if query_image_path and os.path.exists(query_image_path):
        query_pil = Image.open(query_image_path).convert("RGB")
        query_cv = np.array(query_pil)[:, :, ::-1]
        resized_query = cv2.resize(query_cv, (img_width, img_height))
        bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
        query_display_area[img_height * (row_count - 1):, :] = cv2.resize(bordered_query, (img_width, img_height))
        cv2.putText(query_display_area, "Query", (10, panoramic_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 处理检索到的图像
    for i, result in enumerate(retrieved_results):
        row, col = i // row_count, i % row_count
        start_row, start_col = row * img_height, col * img_width
        
        img_path = result['image_path']
        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic_image[start_row:start_row + img_height, start_col:start_col + img_width] = bordered_retrieved
        
        # 添加索引号和相似度
        cv2.putText(panoramic_image, f"{i+1}", (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(panoramic_image, f"{result['distance']:.3f}", (start_col + 10, start_row + img_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return np.hstack([query_display_area, panoramic_image])

# 1. 初始化设置
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth"
DATA_DIR = "../../data/C3/dragon"
METADATA_PATH = "../../data/C4/metadata/dragon.json"
COLLECTION_NAME = "multimodal_dragon_demo"
MILVUS_URI = "http://localhost:19530"

# 2. 初始化数据集和编码器
print("--> 正在初始化数据集...")
dataset = DragonDataset(DATA_DIR, METADATA_PATH)
print(f"加载了 {len(dataset.images)} 张龙类图像")

print("--> 正在初始化编码器和 Milvus 客户端...")
encoder = Encoder(MODEL_NAME, MODEL_PATH, use_local_bge_m3=True)
milvus_client = MilvusClient(uri=MILVUS_URI)

# 3. 创建 Milvus Collection（支持多模态、密集、稀疏三种向量）
print(f"\n--> 正在创建 Collection '{COLLECTION_NAME}'")
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)
    print(f"已删除已存在的 Collection: '{COLLECTION_NAME}'")

# 获取各向量维度
sample_text = dataset.get_text_content(dataset.images[0])
sample_path = dataset.images[0].path
multimodal_dim = len(encoder.encode_multimodal(sample_path, sample_text))
dense_dim = encoder.bge_m3.dim["dense"] if encoder.bge_m3 else None

print(f"多模态向量维度: {multimodal_dim}")
if dense_dim:
    print(f"密集向量维度: {dense_dim}")

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),
    # 三种向量字段
    FieldSchema(name="multimodal_vector", dtype=DataType.FLOAT_VECTOR, dim=multimodal_dim),
]

# 如果 BGE-M3 可用，添加稀疏和密集向量字段
if encoder.bge_m3 and dense_dim:
    fields.append(FieldSchema(name="text_sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR))
    fields.append(FieldSchema(name="text_dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim))

schema = CollectionSchema(fields, description="多模态龙类图像检索（支持混合向量）")

# 创建集合
milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
print(f"成功创建 Collection: '{COLLECTION_NAME}'")

# 4. 准备并插入数据
print(f"\n--> 正在向 '{COLLECTION_NAME}' 插入数据")
img_ids, image_paths, titles, descriptions = [], [], [], []
categories, locations, environments = [], [], [], []
multimodal_vectors = []
text_sparse_vectors = [] if encoder.bge_m3 else None
text_dense_vectors = [] if encoder.bge_m3 else None

for img_data in tqdm(dataset.images, desc="生成向量嵌入"):
    text_content = dataset.get_text_content(img_data)

    # 多模态向量（Visual BGE：图像+文本）
    multimodal_vector = encoder.encode_multimodal(img_data.path, text_content)

    # 混合向量（BGE-M3：稀疏+密集）
    if encoder.bge_m3:
        text_embeddings = encoder.encode_text_hybrid(text_content)

    # 收集元数据
    img_ids.append(img_data.img_id)
    image_paths.append(img_data.path)
    titles.append(img_data.title)
    descriptions.append(img_data.description)
    categories.append(img_data.category)
    locations.append(img_data.location)
    environments.append(img_data.environment)

    multimodal_vectors.append(multimodal_vector)
    if encoder.bge_m3:
        text_sparse_vectors.append(text_embeddings['sparse']._getrow(0))
        text_dense_vectors.append(text_embeddings['dense'][0])

# 组装插入数据
if encoder.bge_m3:
    insert_data = [
        img_ids, image_paths, titles, descriptions, categories, locations, environments,
        multimodal_vectors, text_sparse_vectors, text_dense_vectors
    ]
else:
    insert_data = [
        img_ids, image_paths, titles, descriptions, categories, locations, environments,
        multimodal_vectors
    ]

result = milvus_client.insert(collection_name=COLLECTION_NAME, data=insert_data)
print(f"成功插入 {result['insert_count']} 条数据。")

# 5. 创建索引
print(f"\n--> 正在为 '{COLLECTION_NAME}' 创建索引")
index_params = milvus_client.prepare_index_params()

# 多模态向量索引（HNSW + COSINE）
index_params.add_index(
    field_name="multimodal_vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256}
)
print("多模态向量索引创建成功")

# 稀疏向量索引（仅当 BGE-M3 可用时）
if encoder.bge_m3:
    index_params.add_index(
        field_name="text_sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP"
    )
    print("稀疏向量索引创建成功")

    # 密集向量索引（AUTOINDEX + IP）
    index_params.add_index(
        field_name="text_dense_vector",
        index_type="AUTOINDEX",
        metric_type="IP"
    )
    print("密集向量索引创建成功")

milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
milvus_client.load_collection(collection_name=COLLECTION_NAME)
print("已加载 Collection 到内存中。")

# 6. 执行多模态检索
print(f"\n--> 正在 '{COLLECTION_NAME}' 中执行多模态检索")

# 示例1：图像+文本查询
query_image_path = os.path.join(DATA_DIR, "query.png")
query_text = "悬崖上的巨龙"
query_vector = encoder.encode_query(image_path=query_image_path, text=query_text, mode="multimodal")

print(f"\n=== 多模态查询（图像+文本）===")
print(f"查询图像: {query_image_path}")
print(f"查询文本: {query_text}")

search_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    output_fields=["img_id", "image_path", "title", "description", "category", "location", "environment"],
    limit=6,
    search_params={"metric_type": "COSINE", "params": {"ef": 128}},
    anns_field="multimodal_vector"
)[0]

retrieved_results = []
print("检索结果:")
for i, hit in enumerate(search_results):
    print(f"  Top {i+1}: ID={hit['id']}, 距离={hit['distance']:.4f}")
    print(f"    标题: {hit['entity']['title']}")
    print(f"    描述: {hit['entity']['description'][:100]}...")
    print(f"    类别: {hit['entity']['category']}")
    print(f"    路径: {hit['entity']['image_path']}")
    print("-" * 50)
    retrieved_results.append({
        'image_path': hit['entity']['image_path'],
        'distance': hit['distance']
    })

# 示例2：纯文本混合检索（使用 BGE-M3 的密集+稀疏向量）
if encoder.bge_m3:
    print(f"\n=== 纯文本混合检索（BGE-M3）===")
    text_query = "悬崖上的巨龙"

    # 使用 BGE-M3 编码查询文本
    text_embeddings = encoder.encode_text_hybrid(text_query)
    dense_vec = text_embeddings['dense']
    sparse_vec = text_embeddings['sparse']

    # 密集向量检索
    print(f"\n[密集向量检索] 查询文本: {text_query}")
    dense_search_results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[dense_vec],
        output_fields=["img_id", "image_path", "title", "description", "category"],
        limit=3,
        search_params={"metric_type": "IP", "params": {}},
        anns_field="text_dense_vector"
    )[0]
    print("密集检索结果:")
    for i, hit in enumerate(dense_search_results):
        print(f"  Top {i+1}: {hit['entity']['title']} (距离: {hit['distance']:.4f})")

    # 稀疏向量检索
    print(f"\n[稀疏向量检索] 查询文本: {text_query}")
    sparse_search_results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[sparse_vec],
        output_fields=["img_id", "image_path", "title", "description", "category"],
        limit=3,
        search_params={"metric_type": "IP", "params": {}},
        anns_field="text_sparse_vector"
    )[0]
    print("稀疏检索结果:")
    for i, hit in enumerate(sparse_search_results):
        print(f"  Top {i+1}: {hit['entity']['title']} (距离: {hit['distance']:.4f})")
else:
    print("\n⚠️ BGE-M3 模型未初始化，跳过纯文本混合检索")

# 示例3：纯图像查询
print(f"\n=== 纯图像查询 ===")
image_query_path = os.path.join(DATA_DIR, "query.png")
image_query_vector = encoder.encode_query(image_path=image_query_path, text=None, mode="multimodal")

print(f"查询图像: {image_query_path}")

image_search_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[image_query_vector],
    output_fields=["img_id", "image_path", "title", "description", "category", "location", "environment"],
    limit=3,
    search_params={"metric_type": "COSINE", "params": {"ef": 128}},
    anns_field="multimodal_vector"
)[0]

print("图像检索结果:")
for i, hit in enumerate(image_search_results):
    print(f"  Top {i+1}: {hit['entity']['title']} (距离: {hit['distance']:.4f})")
    print(f"    类别: {hit['entity']['category']}")
    print(f"    描述: {hit['entity']['description'][:80]}...")
    print(f"    路径: {hit['entity']['image_path']}")
    print("-" * 30)

# 7. 可视化与清理
print(f"\n--> 正在可视化结果并清理资源")
if retrieved_results:
    panoramic_image = visualize_results(query_image_path, retrieved_results)
    combined_image_path = "../../data/C4/multimodal_search.png"
    cv2.imwrite(combined_image_path, panoramic_image)
    print(f"结果图像已保存到: {combined_image_path}")
    # Image.open(combined_image_path).show()

# 8. 清理资源
milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'") 
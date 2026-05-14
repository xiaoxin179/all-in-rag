import torch
from cv2.typing import map_int_and_double

from visual_bge.visual_bge.modeling import Visualized_BGE
# 初始化一个多模态的嵌入模型
model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",
                      model_weight="../../models/bge/Visualized_base_en_v1.5.pth"  #加载微调权重
                       )
# 切换到推理模式
model.eval()
def calculate_source():
    with torch.no_grad():  # 禁用梯度计算，节省计算资源，代码只做预测末座学习
        # model.encode会把图片或者文本转换为向量的同时做归一化
        text_emb = model.encode(text="datawhale开源组织的logo")  # 编码为文本向量
        img_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png")  # 编码为图像向量
        multi_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png", text="datawhale开源组织的logo")
        img_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png")
        multi_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png", text="datawhale开源组织的logo")

    # 计算相似度 为什么矩阵乘法得到的就是相似度
    # 归一化之后使用向量的点积得到的就是相似度，为了可以向量可以乘法，就需要把其中一个矩阵转置
    sim_1 = img_emb_1 @ img_emb_2.T
    sim_2 = img_emb_1 @ multi_emb_1.T
    sim_3 = text_emb @ multi_emb_1.T
    sim_4 = multi_emb_1 @ multi_emb_2.T

    print("=== 相似度计算结果 ===")
    print(f"纯图像 vs 纯图像: {sim_1}")
    print(f"图文结合1 vs 纯图像: {sim_2}")
    print(f"图文结合1 vs 纯文本: {sim_3}")
    print(f"图文结合1 vs 图文结合2: {sim_4}")

    # 向量信息分析
    print("\n=== 嵌入向量信息 ===")
    print(f"多模态向量维度: {multi_emb_1.shape}")
    print(f"图像向量维度: {img_emb_1.shape}")
    print(f"多模态向量示例 (前10个元素): {multi_emb_1[0][:10]}")
    print(f"图像向量示例 (前10个元素):   {img_emb_1[0][:10]}")
def calculate_customer():
    with torch.no_grad():  # 禁用梯度计算，节省计算资源，代码只做预测末座学习
        # model.encode会把图片或者文本转换为向量的同时做归一化
        car_text = model.encode(text="limousine")  # 编码为文本向量
        car_image = model.encode(image="./visual_bge/imgs/cus/img.png")  # 编码为图像向量
        seal_car = model.encode(image="./visual_bge/imgs/cus/img.png", text="BYD Car")
        ts_car = model.encode(image="./visual_bge/imgs/cus/ts-car.png", text="TS Car")

    # 计算相似度 为什么矩阵乘法得到的就是相似度
    # 归一化之后使用向量的点积得到的就是相似度，为了可以向量可以乘法，就需要把其中一个矩阵转置
    sim_1 = car_text @ car_image.T
    sim_2 = car_text @ seal_car.T
    sim_3 = car_text @ ts_car.T
    sim_4 = seal_car @ ts_car.T

    print("=== 相似度计算结果 ===")
    print(f"文本 vs 纯图像: {sim_1}")
    print(f"文本 vs 图文结合(匹配): {sim_2}")
    print(f"文本 vs 图文结合(不匹配): {sim_3}")
    print(f"图文结合(匹配) vs 图文结合(不匹配): {sim_4}")

    # 向量信息分析
    print("\n=== 嵌入向量信息 ===")
    print(f"文本向量维度: {car_text.shape}")
    print(f"纯图像向量维度: {car_image.shape}")
    print(f"图文结合向量维度: {seal_car.shape}")
    print(f"\n文本向量示例 (前10个元素): {car_text[0][:10]}")
    print(f"纯图像向量示例 (前10个元素): {car_image[0][:10]}")
    print(f"图文结合向量示例 (前10个元素): {seal_car[0][:10]}")

if __name__ == '__main__':
    calculate_customer()
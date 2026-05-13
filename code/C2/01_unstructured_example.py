def simple():
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.auto import partition

    # PDF文件路径
    pdf_path = "../../data/C2/pdf/rag.pdf"
    strategies = ["fast", "auto"]  # 修正 hi_req → hi_res
    for strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"策略: {strategy}")
        print('=' * 60)

        elements = partition_pdf(
            filename=pdf_path,
            strategy=strategy
        )

        print(f"元素数量: {len(elements)}")
        print(f"总字符数: {sum(len(str(e)) for e in elements)}")

        # 统计元素类型
        from collections import Counter

        types = Counter(e.category for e in elements)
        print(f"元素类型分布: {dict(types)}")
def sea_metadata():
    from unstructured.partition.pdf import partition_pdf

    pdf_path = "../../data/C2/pdf/rag.pdf"

    elements = partition_pdf(
        filename=pdf_path,
        strategy="auto"
    )

    # 查看第一个元素的详细信息
    if elements:
        first_element = elements[0]
        print(f"元素类型: {first_element.category}")
        print(f"元素文本: {str(first_element)[:200]}...")
        print(f"\n元数据内容:")
        print(first_element.metadata)

        # 查看所有元素有哪些元数据字段
        print("\n所有元素的元数据字段（第一个非空示例）:")
        for key, value in first_element.metadata.__dict__.items():
            if value is not None:
                print(f"  {key}: {value}")
def get_image():
    from unstructured.partition.pdf import partition_pdf

    pdf_path = "../../data/C2/pdf/rag.pdf"

    # 使用 hi_res 策略并提取图片
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        extract_image_block_types=["Image"],  # 提取图片
    )

    # 筛选出图片类型的元素
    images = [e for e in elements if e.category == "Image"]
    print(f"发现 {len(images)} 张图片")

    for i, img in enumerate(images, 1):
        print(f"\n图片 {i}:")
        print(f"  元数据: {img.metadata}")
        print(f"  内容: {str(img)[:100]}...")
 # 使用Unstructured加载并解析PDF文档
# elements = partition(
#     filename=pdf_path,
# content_type="application/pdf",
# )
# # 文档是否直接被切块了？
# # 他只是把文档拆分为结构化的语义单元
# # 打印解析结果
# print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")
#
# # 统计元素类型
# from collections import Counter
# types = Counter(e.category for e in elements)
# print(f"元素类型: {dict(types)}")
#
# # 显示所有元素
# print("\n所有元素:")
# for i, element in enumerate(elements, 1):
#     print(f"Element {i} ({element.category}):")
#     print(element)
#     print("=" * 60)
# from unstructured.partition.auto import partition
# import inspect

if __name__ == '__main__':
    get_image()
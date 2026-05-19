import requests
from pathlib import Path
from tqdm import tqdm


def download_bge_small_en_model():
    """
    下载 BAAI/bge-small-en Embedding 模型
    如果模型文件不存在，则从 Hugging Face 下载
    下载到本地 ../../models/bge-small-en/ 目录
    """
    model_dir = Path("../../models/bge-small-en")
    # 核心文件存在即认为模型已就绪
    required_files = ["pytorch_model.bin", "config.json", "tokenizer.json"]
    if all((model_dir / f).exists() for f in required_files):
        print(f"模型文件已存在: {model_dir}")
        print(f"模型大小: {sum(f.stat().st_size for f in model_dir.glob('*') if f.is_file()) / (1024 * 1024):.1f} MB")
        return str(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"创建模型目录: {model_dir}")

    download_url = "https://huggingface.co/BAAI/bge-small-en/resolve/main/pytorch_model.bin?download=true"
    config_url = "https://huggingface.co/BAAI/bge-small-en/resolve/main/config.json?download=true"
    tokenizer_files = [
        ("tokenizer.json", "tokenizer.json"),
        ("tokenizer_config.json", "tokenizer_config.json"),
        ("special_tokens_map.json", "special_tokens_map.json"),
        ("vocab.txt", "vocab.txt"),
    ]

    files_to_download = [
        (download_url, "pytorch_model.bin"),
        (config_url, "config.json"),
        *[(f"https://huggingface.co/BAAI/bge-small-en/resolve/main/{src}", dst) for src, dst in tokenizer_files],
    ]

    for url, filename in files_to_download:
        target_path = model_dir / filename
        if target_path.exists():
            print(f"  [跳过] {filename} 已存在")
            continue

        print(f"\n开始下载 {filename}...")
        print(f"  地址: {url}")

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(target_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=filename,
                    ncols=80,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print(f"  ✅ {filename} 下载完成 ({target_path.stat().st_size / (1024 * 1024):.1f} MB)")

        except requests.exceptions.RequestException as e:
            print(f"  ❌ {filename} 下载失败: {e}")
            if target_path.exists():
                target_path.unlink()
            return None
        except Exception as e:
            print(f"  ❌ {filename} 发生错误: {e}")
            if target_path.exists():
                target_path.unlink()
            return None

    print(f"\n模型下载完成: {model_dir}")
    total_size = sum(f.stat().st_size for f in model_dir.glob("*") if f.is_file())
    print(f"总大小: {total_size / (1024 * 1024):.1f} MB")
    return str(model_dir)


def download_visualized_bge_model():
    """
    下载 Visual BGE 模型权重文件
    如果模型文件不存在，则从 Hugging Face 下载
    """
    # 定义模型路径和下载URL
    model_dir = Path("../../models/bge")
    model_file = model_dir / "Visualized_base_en_v1.5.pth"
    download_url = "https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_base_en_v1.5.pth?download=true"
    
    # 检查模型文件是否已存在
    if model_file.exists():
        print(f"模型文件已存在: {model_file}")
        print(f"文件大小: {model_file.stat().st_size / (1024*1024):.1f} MB")
        return str(model_file)
    
    # 创建目录
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"创建模型目录: {model_dir}")
    
    # 下载模型
    print(f"开始下载模型...")
    print(f"下载地址: {download_url}")
    
    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(model_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # 显示下载进度
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r下载进度: {progress:.1f}% ({downloaded_size/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)", end='')
        
        print(f"\n模型下载完成: {model_file}")
        print(f"文件大小: {model_file.stat().st_size / (1024*1024):.1f} MB")
        return str(model_file)
        
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        # 如果下载失败，删除不完整的文件
        if model_file.exists():
            model_file.unlink()
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        if model_file.exists():
            model_file.unlink()
        return None

if __name__ == "__main__":
    import sys
    print("=" * 40)
    print("         模型下载工具")
    print("=" * 40)
    print("请选择要下载的模型：")
    print("  [1] bge-small-en      - BGE 轻量级英文 Embedding 模型")
    print("  [2] visualized-bge    - Visual BGE 模型")
    print("=" * 40)

    choice = input("请输入选项 (1/2): ").strip()

    if choice == "1":
        model_path = download_bge_small_en_model()
    elif choice == "2":
        model_path = download_visualized_bge_model()
    else:
        print(f"\n❌ 无效选项: {choice}")
        sys.exit(1)

    if model_path:
        print(f"\n✅ 模型准备就绪: {model_path}")
    else:
        print("\n❌ 模型下载失败")

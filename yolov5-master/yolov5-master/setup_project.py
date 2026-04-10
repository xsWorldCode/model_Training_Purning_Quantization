import os
import shutil
from pathlib import Path

def find_file_in_dir(filename, search_path):
    """递归搜索文件名并返回所在目录"""
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return root
    return None

def organize_my_vision_project():
    # --- 1. 定义目标路径 ---
    project_root = Path.cwd()
    sam_dir = project_root / "models" / "sam"
    sam_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 开始扫描系统环境...")

    # --- 2. 寻找 YOLOv5 源码与 hubconf.py ---
    # 这里扫描你之前提到的 F 盘路径
    yolo_search_base = r"F:\python_code\yolov5-master"
    real_yolo_path = find_file_in_dir("hubconf.py", yolo_search_base)
    
    if real_yolo_path:
        print(f"✅ 找到 YOLOv5 核心目录: {real_yolo_path}")
        print(f"👉 请在你的 Gradio 代码中修改: repo_dir = r'{real_yolo_path}'")
    else:
        print("❌ 未能在 F 盘找到包含 hubconf.py 的 YOLOv5 文件夹，请确认源码是否完整。")

    # --- 3. 寻找并提取 SAM 权重 (从 C 盘缓存) ---
    user_profile = os.environ.get('USERPROFILE')
    x_labeling_path = Path(user_profile) / ".xanylabeling" / "models"
    
    sam_files = {
        "efficientvit_sam_l0_encoder.onnx": "sam_encoder.onnx",
        "efficientvit_sam_l0_decoder.onnx": "sam_decoder.onnx"
    }

    print(f"\n📂 正在检查 X-AnyLabeling 缓存: {x_labeling_path}")
    
    found_count = 0
    if x_labeling_path.exists():
        for remote_name, local_name in sam_files.items():
            source_file = x_labeling_path / remote_name
            target_file = sam_dir / local_name
            
            if source_file.exists():
                if not target_file.exists():
                    print(f"🚚 发现缓存，正在复制 {remote_name} -> {local_name}...")
                    shutil.copy(source_file, target_file)
                else:
                    print(f"✨ {local_name} 已存在于项目目录，无需重复复制。")
                found_count += 1
            else:
                print(f"❓ 缓存中缺少: {remote_name}")
    else:
        print("ℹ️ 未发现 X-AnyLabeling 缓存目录。")

    if found_count == 2:
        print("\n🎉 大模型组件已全部就绪！你的 5060 Ti 准备好起飞了。")
    else:
        print("\n⚠️ 仍有模型缺失，建议手动下载或在 X-AnyLabeling 中加载一次 SAM 模型。")

if __name__ == "__main__":
    organize_my_vision_project()
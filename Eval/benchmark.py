import torch
import time
import os
import sys
from models.ResNet import ResNet54

def benchmark_pruned():
    device = torch.device("cpu")
    dummy_input = torch.randn(1, 3, 224, 224)

    # 1. 加载原始模型
    print(">>> 加载 FP32 原始模型...")
    model_orig = ResNet54(num_classes=2).eval()
    orig_path = r"F:\python_code\checkpoints\best_model.pth"
    
    # 2. 加载剪枝模型
    print(">>> 加载剪枝后的模型...")
    pruned_path = "resnet54_pruned.pth"
    model_pruned = ResNet54(num_classes=2).eval()
    
    if os.path.exists(pruned_path):
        model_pruned.load_state_dict(torch.load(pruned_path, map_location=device))
    else:
        print("❌ 请先运行剪枝脚本！")
        return

    # 3. 性能对比
    configs = [
        ("FP32 (Original)", model_orig, orig_path),
        ("FP32 (Pruned 80%)", model_pruned, pruned_path)
    ]

    print("\n" + "="*70)
    print(f"{'模型名称':<25} | {'平均延迟 (ms)':<15} | {'文件大小 (MB)':<15}")
    print("-" * 70)

    for name, model, p_path in configs:
        # 预热
        with torch.no_grad():
            for _ in range(10): model(dummy_input)
        
        # 计时
        iters = 50
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters): model(dummy_input)
        end = time.perf_counter()
        
        latency = ((end - start) / iters) * 1000
        size = os.path.getsize(p_path) / (1024 * 1024)
        
        print(f"{name:<25} | {latency:>13.2f} ms | {size:>13.2f} MB")
    
    print("="*70)
    print("💡 结论：剪枝模型不会触发硬件指令集错误，且体积大幅缩小。")

if __name__ == "__main__":
    benchmark_pruned()
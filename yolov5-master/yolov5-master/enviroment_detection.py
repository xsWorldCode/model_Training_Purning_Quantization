import os
# 针对 50 系列显卡的特殊补丁：尝试开启延迟加载
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import torch
import torch.nn as nn
import time

def test_cuda_training():
    print("="*50)
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 1. 基础环境检查
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")
    
    if not cuda_available:
        print("错误: CUDA 不可用。请检查驱动和 PyTorch 版本。")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    print(f"正在使用的显卡: {gpu_name}")
    print(f"显卡计算能力 (Compute Capability): {compute_cap}")
    print("="*50)

    try:
        # 2. 算子测试 (很多 50 系报错会在这里触发)
        print("正在进行基础算子测试 (Matrix Multiplication)...")
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize() # 等待 GPU 计算完成
        print(f"基础计算成功！耗时: {time.time() - start:.4f}s")

        # 3. 模拟训练测试 (测试反向传播和梯度更新)
        print("\n正在启动模拟训练测试...")
        # 定义一个微型网络
        model = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 模拟数据
        data = torch.randn(32, 128).to(device)
        target = torch.randn(32, 10).to(device)
        
        # 运行一步训练
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        print(f"模拟训练成功！Loss 值: {loss.item():.4f}")
        print("\n>>> 恭喜！你的 RTX 5060 Ti 已完全适配当前环境，可以进行 Deep Learning 训练。")

    except Exception as e:
        print("\n" + "!"*50)
        print(f"测试失败！具体报错如下:\n{str(e)}")
        print("!"*50)
        if "no kernel image" in str(e).lower():
            print("\n诊断方案:")
            print("1. 你的 PyTorch 预编译包不包含 Blackwell (50系) 架构。")
            print("2. 尝试更新 NVIDIA 驱动至 570.xx 以上。")
            print("3. 尝试安装 torch-nightly 版本或等待官方发布完善的 2.8+ 补丁。")

if __name__ == "__main__":
    test_cuda_training()
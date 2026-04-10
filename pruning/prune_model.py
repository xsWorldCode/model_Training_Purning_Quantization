import torch
import torch.nn.utils.prune as prune
import os
from models.ResNet import ResNet54   # 确保路径正确

def prune_resnet54():
    device = torch.device("cpu")
    model = ResNet54(num_classes=2)
    model_path = r"F:\python_code\checkpoints\model.pth"
    
    if not os.path.exists(model_path):
        print("找不到原始权重文件")
        return

    # 加载原始权重
    checkpoint = torch.load(model_path, map_location=device)
    raw_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    # 仅移除可能存在的 'module.' 前缀（DataParallel 训练时会有）
    state_dict = {k.replace('module.', ''): v for k, v in raw_dict.items()}
    
    # 直接加载，strict=True 确保参数完整匹配
    try:
        model.load_state_dict(state_dict, strict=True)
        print("原始权重加载成功，参数完全匹配")
    except RuntimeError as e:
        print("加载失败，请检查 checkpoint 与模型结构是否一致")
        print(e)
        return

    print(">>> 正在对模型进行全局非结构化剪枝 (50%)...")
    # 收集所有卷积层和线性层的权重进行剪枝
    parameters_to_prune = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            parameters_to_prune.append((m, 'weight'))

    # 全局剪枝，保留最重要的 20% 权重
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )

    # 永久化剪枝（移除 mask，直接将权重中低于阈值的置零）
    for m, name in parameters_to_prune:
        prune.remove(m, name)

    # 保存剪枝后的模型
    save_path = "Pruned.pth"
    torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=True)
    
    print(f"剪枝完成！模型已保存至: {save_path}")
    print(f"原始模型大小: {os.path.getsize(model_path)/1024/1024:.2f} MB")
    print(f"剪枝后模型大小: {os.path.getsize(save_path)/1024/1024:.2f} MB")

if __name__ == "__main__":
    prune_resnet54()
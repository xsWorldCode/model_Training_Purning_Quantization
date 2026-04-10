import torch
import torch.nn.functional as F
import os
import sys

# 将项目根目录添加到 sys.path，确保可以导入 Eval 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ResNet import ResNet54

def eval_pruning_quality():
    device = torch.device("cpu")
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 3, 224, 224)

    print(">>> 正在加载原始模型...")
    model_orig = ResNet54(num_classes=2).eval()
    orig_path = r"F:\python_code\checkpoints\model.pth"
    
    checkpoint = torch.load(orig_path, map_location=device)
    state_dict_orig = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    state_dict_orig = {k.replace('module.', ''): v for k, v in state_dict_orig.items()}
    model_orig.load_state_dict(state_dict_orig, strict=True)
    print("原始模型加载成功")

    print(">>> 正在加载剪枝模型...")
    # 剪枝模型位于项目根目录
    pruned_path = r"F:\python_code\Pruned_Structured.pth"
    if not os.path.exists(pruned_path):
        print(f"错误：找不到剪枝文件 {pruned_path}")
        return
    model_pruned = torch.load(pruned_path, map_location=device, weights_only=False)
    model_pruned.eval()
    print("剪枝模型加载成功")

    with torch.no_grad():
        # 原始特征
        x_o = model_orig.conv1(dummy_input)
        x_o = model_orig.bn1(x_o)
        x_o = model_orig.relu(x_o)
        x_o = model_orig.maxpool(x_o)
        x_o = model_orig.layer1(x_o)
        x_o = model_orig.layer2(x_o)
        x_o = model_orig.layer3(x_o)
        x_o = model_orig.layer4(x_o)
        x_o = model_orig.avgpool(x_o)
        feat_orig = torch.flatten(x_o, 1)

        # 剪枝特征
        x_p = model_pruned.conv1(dummy_input)
        x_p = model_pruned.bn1(x_p)
        x_p = model_pruned.relu(x_p)
        x_p = model_pruned.maxpool(x_p)
        x_p = model_pruned.layer1(x_p)
        x_p = model_pruned.layer2(x_p)
        x_p = model_pruned.layer3(x_p)
        x_p = model_pruned.layer4(x_p)
        x_p = model_pruned.avgpool(x_p)
        feat_pruned = torch.flatten(x_p, 1)

    similarity = F.cosine_similarity(feat_orig, feat_pruned).item()

    print("\n" + "="*50)
    print(f"{'对比项':<20} | {'数值':<20}")
    print("-"*50)
    print(f"{'特征向量维度':<20} | {feat_orig.shape[1]:<20}")
    print(f"{'余弦相似度':<20} | {similarity:.6f}")
    
    if similarity > 0.98:
        status = "极佳 (剪枝几乎无损)"
    elif similarity > 0.90:
        status = "良好 (存在轻微偏差)"
    else:
        status = "警告 (精度下降风险大)"
    print(f"{'剪枝质量评估':<20} | {status:<20}")
    print("="*50)

if __name__ == "__main__":
    eval_pruning_quality()
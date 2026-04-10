import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ResNet import ResNet54   # 你的模型定义

def get_file_size_mb(file_path):
    """返回文件大小（MB）"""
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    else:
        return None

def evaluate_accuracy(model, dataloader, device):
    """计算模型在数据集上的 Top-1 准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def eval_classification_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 定义数据预处理和加载器
    data_dir = r"F:\python_code\DataSets\split"   # 替换为你的数据集根目录
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 2. 加载原始模型
    print(">>> 加载原始模型...")
    orig_path = r"F:\python_code\checkpoints\model.pth"
    orig_size = get_file_size_mb(orig_path)
    if orig_size is not None:
        print(f"原始模型磁盘大小: {orig_size:.2f} MB")
    else:
        print("原始模型文件未找到")
    
    model_orig = ResNet54(num_classes=2).to(device)
    checkpoint = torch.load(orig_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model_orig.load_state_dict(state_dict, strict=True)
    model_orig.eval()
    
    # 3. 加载剪枝模型（微调后的完整模型）
    print(">>> 加载剪枝模型...")
    pruned_path = r"F:\python_code\Pruned_Structured_FineTuned.pth"
    pruned_size = get_file_size_mb(pruned_path)
    if pruned_size is not None:
        print(f"剪枝模型磁盘大小: {pruned_size:.2f} MB")
    else:
        print("剪枝模型文件未找到，请检查路径")
        return
    
    model_pruned = torch.load(pruned_path, map_location=device, weights_only=False)
    model_pruned.eval()
    
    # 4. 评估准确率
    print(">>> 评估原始模型准确率...")
    acc_orig = evaluate_accuracy(model_orig, val_loader, device)
    print(f"原始模型 Top-1 准确率: {acc_orig:.4f} ({acc_orig*100:.2f}%)")
    
    print(">>> 评估剪枝模型准确率...")
    acc_pruned = evaluate_accuracy(model_pruned, val_loader, device)
    print(f"剪枝模型 Top-1 准确率: {acc_pruned:.4f} ({acc_pruned*100:.2f}%)")
    
    print(f"\n精度变化: {acc_pruned - acc_orig:+.4f}")
    print(f"模型大小变化: {pruned_size - orig_size:+.2f} MB (剪枝后/微调后)")

if __name__ == "__main__":
    eval_classification_accuracy()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_model_size_mb(path):
    """返回模型文件的磁盘大小（MB）"""
    if os.path.exists(path):
        size_bytes = os.path.getsize(path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    else:
        return None

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return correct / total

def fine_tune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载剪枝模型（请确保路径正确）
    pruned_path = r"F:\python_code\Pruned_Structured.pth"
    model = torch.load(pruned_path, map_location=device, weights_only=False)
    model = model.to(device)
    print("剪枝模型加载成功")
    
    # 显示剪枝模型的磁盘大小
    pruned_size = get_model_size_mb(pruned_path)
    if pruned_size is not None:
        print(f"剪枝模型磁盘大小: {pruned_size:.2f} MB")
    else:
        print("剪枝模型文件未找到，无法获取大小")
    
    # 数据加载器（根据你的实际情况修改）
    data_dir = r"F:\python_code\DataSets\split"   # 替换为你的数据集根目录
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform_train)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # 微调前评估
    acc_before = evaluate_accuracy(model, val_loader, device)
    print(f"微调前准确率: {acc_before:.4f}")
    
    num_epochs = 10
    best_acc = acc_before
    best_model_path = "Pruned_Structured_FineTuned.pth"
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        acc = evaluate_accuracy(model, val_loader, device)
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Val Acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model, best_model_path)
            # 保存后立即显示最佳模型的大小
            best_size = get_model_size_mb(best_model_path)
            if best_size is not None:
                print(f"  -> 保存最佳模型，当前磁盘大小: {best_size:.2f} MB")
        scheduler.step()
    
    # 最终输出最佳模型的大小
    final_size = get_model_size_mb(best_model_path)
    print(f"\n微调完成，最佳准确率: {best_acc:.4f}，恢复提升: {best_acc - acc_before:+.4f}")
    if final_size is not None:
        print(f"最佳模型磁盘大小: {final_size:.2f} MB")
    else:
        print("最佳模型文件未找到")

if __name__ == "__main__":
    fine_tune()
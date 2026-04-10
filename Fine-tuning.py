import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
from models.ResNet import ResNet54   # 仅用于原始模型加载（如果需要）

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

def fine_tune_pruned_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载剪枝后的完整模型
    pruned_path = r"F:\python_code\Pruned_Structured.pth"
    model = torch.load(pruned_path, map_location=device, weights_only=False)
    model = model.to(device)
    print("剪枝模型加载成功")
    
    # 2. 准备数据加载器（请根据你的实际数据集修改）
    data_dir = r"F:\python_code\DataSets\split"   # 替换为数据集根目录
    # 训练集预处理（可包含数据增强）
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 3. 定义损失函数和优化器（使用很小的学习率）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)   # 原始学习率的1/10左右
    # 可选：学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # 4. 微调前评估
    print("微调前剪枝模型准确率:")
    acc_before = evaluate_accuracy(model, val_loader, device)
    print(f"Top-1 Acc: {acc_before:.4f} ({acc_before*100:.2f}%)")
    
    # 5. 微调训练
    num_epochs = 10   # 通常 5-10 个 epoch 足够
    best_acc = acc_before
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 每个 epoch 后验证
        acc = evaluate_accuracy(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            # 保存最佳模型
            torch.save(model, "Pruned_Structured_FineTuned.pth")
        scheduler.step()
    
    print(f"\n微调完成！最佳验证准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"精度恢复: {best_acc - acc_before:+.4f}")

if __name__ == "__main__":
    fine_tune_pruned_model()
import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.ResNet import ResNet54

def run_accuracy_test():
    device = torch.device("cpu")
    
    # ✅ 修正1：使用 val 或 test 文件夹
    val_dir = r"F:\python_code\DataSets\split\val"
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(val_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # ✅ 修正2：两个模型都加载各自的权重
    model_orig = ResNet54(num_classes=2).eval()
    model_orig.load_state_dict(torch.load(r"F:\python_code\checkpoints\final_model.pth", map_location=device))

    model_pruned = ResNet54(num_classes=2).eval()
    model_pruned.load_state_dict(torch.load(r"F:\python_code\checkpoints\resnet54_pruned.pth", map_location=device))

    print(f"对 {len(dataset)} 张图片进行对比测试")
    
    correct_orig = 0
    correct_pruned = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs_orig = model_orig(images)
            outputs_pruned = model_pruned(images)
            
            _, pred_orig = torch.max(outputs_orig, 1)
            _, pred_pruned = torch.max(outputs_pruned, 1)
            
            total += labels.size(0)
            correct_orig += (pred_orig == labels).sum().item()
            correct_pruned += (pred_pruned == labels).sum().item()
            
            if total % 128 == 0:
                print(f"进度: {total}/{len(dataset)}")

    # ✅ 修正3：加入模型大小对比
    orig_size = os.path.getsize("resnet54_original.pth") / (1024 * 1024)
    pruned_size = os.path.getsize("resnet54_pruned.pth") / (1024 * 1024)

    print("\n" + "="*50)
    print(f"原始模型准确率: {100 * correct_orig / total:.2f}%  (大小: {orig_size:.2f} MB)")
    print(f"剪枝模型准确率: {100 * correct_pruned / total:.2f}%  (大小: {pruned_size:.2f} MB)")
    print(f"准确率掉落: {100 * (correct_orig - correct_pruned) / total:.2f}%")
    print(f"体积压缩: {100 * (1 - pruned_size / orig_size):.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_accuracy_test()

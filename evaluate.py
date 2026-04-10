import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from models.ResNet import ResNet54, Bottleneck

# 1. 環境設置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.backends.quantized.engine = 'fbgemm'

def load_quantized_model(model_path, device='cpu'):
    print(f">>> 正在初始化量化模型結構...")
    model = ResNet54(num_classes=2)
    model.eval() 
    
    # --- 算子融合修正 ---
    # 融合基礎層
    torch.ao.quantization.fuse_modules(model, ['conv1', 'bn1', 'relu'], inplace=True)
    
    # 遍歷所有模塊進行融合
    for m in model.modules():
        if isinstance(m, Bottleneck):
            # 融合 Bottleneck 內部的 conv 和 bn (不強求融合不存在的 relu1)
            torch.ao.quantization.fuse_modules(m, ['conv1', 'bn1'], inplace=True)
            torch.ao.quantization.fuse_modules(m, ['conv2', 'bn2'], inplace=True)
            torch.ao.quantization.fuse_modules(m, ['conv3', 'bn3'], inplace=True)
            if m.downsample:
                torch.ao.quantization.fuse_modules(m.downsample, ['0', '1'], inplace=True)
    
    # --- 量化準備 ---
    model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.ao.quantization.prepare(model)
    with torch.no_grad():
        model_prepared(torch.randn(1, 3, 224, 224)) # 校準 observer
    model_int8 = torch.ao.quantization.convert(model_prepared)
    
    # --- 關鍵：權重名稱自動映射 ---
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    
    # 獲取模型當前的鍵名
    current_model_keys = list(model_int8.state_dict().keys())
    
    # 嘗試將權重文件中的 layers.0.x 映射到 layer1.x
    for i, (k, v) in enumerate(state_dict.items()):
        if i < len(current_model_keys):
            new_state_dict[current_model_keys[i]] = v
            
    print(f">>> 正在加載權重 (映射匹配)...")
    model_int8.load_state_dict(new_state_dict, strict=False)
    return model_int8

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    
    # 量化參數（需與訓練時一致，若不確定則使用標準值）
    input_scale = 0.0078125 
    input_zero_point = 128
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            
            # 1. 量化輸入
            x_quant = torch.quantize_per_tensor(inputs, scale=input_scale, 
                                              zero_point=input_zero_point, 
                                              dtype=torch.quint8)
            
            # 2. 推理：直接調用 model(x_quant) 而不是手動遍歷，這樣最安全
            # 如果你的模型定義中包含了 QuantStub，可以直接輸入 inputs
            # 否則我們使用手動定義的 forward 邏輯
            outputs = model(x_quant)
            
            # 3. 如果輸出是量化的，則反量化
            if outputs.is_quantized:
                outputs = outputs.dequantize()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
            
    return 100. * correct / total

def main():
    model_path = r'F:\python_code\checkpoints\resnet54_pruned_int8.pth'
    data_dir = r'F:\python_code\DataSets\split'
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    try:
        model = load_quantized_model(model_path)
        accuracy = evaluate_model(model, test_loader)
        print(f"\n🎯 評估完成！測試準確率: {accuracy:.2f}%")
    except Exception as e:
        print(f"❌ 評估失敗: {e}")

if __name__ == "__main__":
    main()
import torch
import torch_pruning as tp
from models.ResNet import ResNet54

def prune_resnet54_structured():
    print(">>> 正在加载原始模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet54(num_classes=2).to(device)
    model_path = r"F:\python_code\checkpoints\model.pth"
    
    checkpoint = torch.load(model_path, map_location=device)
    raw_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in raw_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    imp = tp.importance.MagnitudeImportance(p=2)
    ignored_layers = [model.fc]
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=0.2,
        ignored_layers=ignored_layers,
        round_to=8
    )
    
    print(">>> 正在执行结构化剪枝...")
    pruner.step()
    
    save_path = "Pruned_Structured.pth"
    torch.save(model, save_path)
    print(f"结构化剪枝完成！完整模型已保存至: {save_path}")
    
    # 验证加载（必须加 weights_only=False）
    test_load = torch.load(save_path, map_location=device, weights_only=False)
    print(f"保存的对象类型: {type(test_load)}")

if __name__ == "__main__":
    prune_resnet54_structured()
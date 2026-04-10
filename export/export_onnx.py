import torch
import onnx
import os
import sys

# 如果需要导入自定义模型类，取消注释并修改路径
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from Eval.ResNet import ResNet54

def export_model_to_onnx():
    # ========== 配置参数 ==========
    # 输入模型路径（支持完整模型或 state_dict）
    MODEL_PATH = r"F:\python_code\Pruned_Structured_FineTuned.pth"
    # 输出 ONNX 路径
    ONNX_PATH = r"F:\python_code\pt_to_onnx\model_pruned.onnx"
    
    # 模型参数（根据你的模型修改）
    NUM_CLASSES = 2
    INPUT_CHANNELS = 3
    INPUT_HEIGHT = 224
    INPUT_WIDTH = 224
    BATCH_SIZE = 1   # 静态 batch size
    
    # ONNX 导出参数
    OPSET_VERSION = 14
    DO_CONSTANT_FOLDING = True
    
    # ========== 1. 加载模型 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在 {MODEL_PATH}")
        return
    
    # 方法1：如果是完整保存的模型（推荐）
    try:
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        print("成功加载完整模型 (torch.load)")
    except Exception as e:
        print(f"加载完整模型失败: {e}")
        print("尝试以 state_dict 方式加载...")
        # 方法2：如果是 state_dict，需要先创建模型实例
        # 注意：结构化剪枝后的模型通道数已变，不能直接使用原始 ResNet54 结构！
        # 如果你保存的是 state_dict，请确保下面的模型定义与剪枝后结构一致
        # 这里假设你保存的是完整模型，所以不会走到这个分支
        try:
            # 如果你有剪枝后的模型类定义，可以在这里导入
            # from your_model import YourPrunedResNet
            # model = YourPrunedResNet(num_classes=NUM_CLASSES)
            # model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("错误: 当前脚本仅支持完整模型加载。如需加载 state_dict，请手动构建模型结构。")
            return
        except:
            return
    
    model = model.to(device)
    model.eval()
    print(f"模型类型: {type(model)}")
    
    # 可选：打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # ========== 2. 创建虚拟输入 ==========
    dummy_input = torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH).to(device)
    
    # ========== 3. 导出 ONNX ==========
    print(f"正在导出 ONNX 到: {ONNX_PATH}")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,          # 保存模型参数
        opset_version=OPSET_VERSION,
        do_constant_folding=DO_CONSTANT_FOLDING,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None            # 静态 batch，不设置动态轴
    )
    
    # ========== 4. 验证 ONNX 模型 ==========
    print("验证 ONNX 模型...")
    try:
        onnx_model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型验证通过")
        
        # 打印输入输出信息
        print("\n模型输入:")
        for inp in onnx_model.graph.input:
            print(f"  名称: {inp.name}, 形状: {[dim.dim_value for dim in inp.type.tensor_type.shape.dim]}")
        print("模型输出:")
        for out in onnx_model.graph.output:
            print(f"  名称: {out.name}, 形状: {[dim.dim_value for dim in out.type.tensor_type.shape.dim]}")
        
        # 文件大小
        size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
        print(f"\nONNX 文件大小: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"ONNX 验证失败: {e}")
        return
    
    print("\n导出完成！")
    print(f"可使用以下命令转换为 TensorRT 引擎:")
    print(f"  trtexec --onnx={ONNX_PATH} --saveEngine=model_fp16.engine --fp16")

if __name__ == "__main__":
    export_model_to_onnx()
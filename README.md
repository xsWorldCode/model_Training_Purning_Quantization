# 模型压缩与TensorRT部署实战

本项目演示了从 PyTorch 模型训练、结构化剪枝、微调、ONNX 导出到 TensorRT FP16 推理的完整流程。

## 环境依赖

- Python 3.8+
- PyTorch 2.x
- TensorRT 10.x
- ONNX 1.15+
- OpenCV

## 快速开始

1. 安装依赖：`pip install -r requirements.txt`
2. 将训练好的模型权重放入 `checkpoints/` 目录
3. 执行剪枝：`python pruning/structured_prune.py`
4. 导出 ONNX：`python export/export_onnx.py`
5. TensorRT 推理：`python deploy/infer_tensorrt.py`

## 文件说明

- `models/ResNet.py`: ResNet54 模型定义
- `pruning/`: 非结构化和结构化剪枝脚本
- `eval/`: 剪枝质量评估（余弦相似度、分类准确率）
- `export/`: 导出 ONNX 模型
- `deploy/`: TensorRT 推理脚本

## 实验结果

- 原始模型：71.43% 准确率，269 MB
- 结构化剪枝 50% + 微调：~68% 准确率，56 MB
- 进一步 FP16 量化：~45 MB

## 许可证

MIT
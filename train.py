# train.py - 训练脚本
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.ResNet import ResNet54
import time
import os
import sys

# 添加当前目录到路径，确保可以导入net.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Trainer:
    def __init__(self, data_dir='DataSets/split_data', batch_size=32, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 数据增强
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 创建模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 使用从Resnet54.py导入的resnet54函数
        self.model = ResNet54(num_classes=2).to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()# 定义交叉熵损失函数
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)# 定义AdamW优化器，设置学习率和权重衰减
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )# 定义学习率调度器，根据验证损失调整学习率，设置耐心值和衰减因子
        
    def load_data(self):
        """加载数据集"""
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'test')
        
        if not os.path.exists(train_dir):
            print(f"训练目录不存在: {train_dir}")
            print("请先运行数据划分脚本")
            sys.exit(1)
        
        train_dataset = datasets.ImageFolder(train_dir, transform=self.train_transform)# 加载训练数据集并应用数据增强
        val_dataset = datasets.ImageFolder(val_dir, transform=self.val_transform)# 加载验证数据集并应用预处理
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers, pin_memory=True
        )# 创建训练数据加载器，启用数据洗牌和多线程加载
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers, pin_memory=True
        )# 创建验证数据加载器，不启用数据洗牌
        
        print(f"训练集: {len(train_dataset)} 张图片, {len(train_dataset.classes)} 个类别")
        print(f"验证集: {len(val_dataset)} 张图片")
        print(f"类别: {train_dataset.classes}")
        
        return train_dataset.classes# 返回类别列表，供后续使用
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)# 将输入和标签移动到设备（GPU或CPU）
            
            self.optimizer.zero_grad()# 清空梯度
            outputs = self.model(inputs)# 前向传播
            loss = self.criterion(outputs, targets)# 计算损失
            loss.backward()# 反向传播
            self.optimizer.step()# 更新参数
            
            running_loss += loss.item()# 累积损失
            _, predicted = outputs.max(1)# 获取预测结果
            total += targets.size(0)# 累积总样本数
            correct += predicted.eq(targets).sum().item()# 累积正确预测数
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx+1}/{len(self.train_loader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')# 每10个batch打印一次当前损失和准确率
        
        return running_loss/len(self.train_loader), 100.*correct/total# 返回平均损失和训练准确率
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0# 初始化验证损失、正确预测数和总样本数
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:# 遍历验证数据加载器
                inputs, targets = inputs.to(self.device), targets.to(self.device)# 将输入和标签移动到设备（GPU或CPU）
                outputs = self.model(inputs)# 前向传播
                loss = self.criterion(outputs, targets)# 计算验证损失
                
                val_loss += loss.item()# 累积验证损失
                _, predicted = outputs.max(1)# 获取预测的最大值索引（即预测类别）
                total += targets.size(0)# 累积总样本数
                correct += predicted.eq(targets).sum().item()# 累积正确预测数
        
        return val_loss/len(self.val_loader), 100.*correct/total# 返回平均验证损失和验证准确率
    
    def train(self, num_epochs=50, save_dir='checkpoints'):
        """完整训练流程"""
        os.makedirs(save_dir, exist_ok=True)# 创建保存目录
        
        best_acc = 0.0# 初始化最佳验证准确率
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}# 初始化训练历史记录
        
        print(f"\n🚀 开始训练，共 {num_epochs} 个epoch")
        print(f"💾 模型将保存到: {save_dir}")
        
        for epoch in range(1, num_epochs + 1):# 遍历每个epoch
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)# 训练一个epoch并获取训练损失和准确率
            
            # 验证
            val_loss, val_acc = self.validate()# 验证模型并获取验证损失和准确率
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # 学习率调整
            self.scheduler.step(val_loss)# 根据验证损失调整学习率
            current_lr = self.optimizer.param_groups[0]['lr']# 获取当前学习率
            
            print(f"\n总结 - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'classes': self.classes
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"保存最佳模型，准确率: {val_acc:.2f}%")
            
            # # 定期保存
            # if epoch % 10 == 0:
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': self.model.state_dict(),
            #         'optimizer_state_dict': self.optimizer.state_dict(),
            #     }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
            #     print(f"保存检查点: epoch_{epoch}")
        
        # 保存最终模型
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'classes': self.classes
        }, os.path.join(save_dir, 'model.pth'))
        
        print(f"\n训练完成！最佳验证准确率: {best_acc:.2f}%")
        
        # 绘制训练曲线
        self.plot_training_history(history, save_dir)
        
        return history
    
    def plot_training_history(self, history, save_dir):
        """绘制训练历史图表"""
        try:
            import matplotlib.pyplot as plt
            
            epochs = range(1, len(history['train_loss']) + 1)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # 绘制损失曲线
            axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
            axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
            axes[0].set_xlabel('Epochs')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True)
            
            # 绘制准确率曲线
            axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
            axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_title('Training and Validation Accuracy')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
            print(f"训练历史图表已保存到: {os.path.join(save_dir, 'training_history.png')}")
            
        except ImportError:
            print("未安装matplotlib,跳过图表绘制")
            print("安装命令: pip install matplotlib")

# 使用示例
if __name__ == "__main__":
    # 初始化训练器
    trainer = Trainer(data_dir='DataSets/split', batch_size=4)
    
    # 加载数据
    classes = trainer.load_data()
    trainer.classes = classes
    
    # 开始训练
    history = trainer.train(num_epochs=200)
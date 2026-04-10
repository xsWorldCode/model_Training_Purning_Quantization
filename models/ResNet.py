import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1 卷積：壓縮通道
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 卷積：提取特徵
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 卷積：擴張通道
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # 如果步長不為1或通道不匹配，則對原始輸入進行下採樣
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 殘差連接：先相加
        out = self.relu(out)  # 最後再進行 ReLU 激活
        return out

class ResNet54(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet54, self).__init__()
        self.in_channels = 64
        
        # 第一層：7x7 卷積
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ⚠️ 標準 ResNet-50 塊配置：[3, 4, 6, 3]
        # Layer 1: 64通道, 3個塊
        self.layer1 = self._make_layer(64, 3)
        # Layer 2: 128通道, 4個塊, 步長為2
        self.layer2 = self._make_layer(128, 4, stride=2)
        # Layer 3: 256通道, 6個塊, 步長為2
        self.layer3 = self._make_layer(256, 6, stride=2)
        # Layer 4: 512通道, 3個塊, 步長為2
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 最終分類層：512 * expansion (4) = 2048
        self.fc = nn.Linear(2048, num_classes)

        # 🚀 權重初始化 (Kaiming Normal)
        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        # 當步長不為1或輸入輸出通道不一致時，需要 Downsample 塊對齊維度
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            )

        layers = []
        # 添加第一個 Bottleneck (處理下採樣)
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * Bottleneck.expansion
        
        # 添加剩餘的 Bottleneck (不改變維度)
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial layers
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # ResNet Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    
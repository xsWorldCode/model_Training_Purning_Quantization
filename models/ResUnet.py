import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """殘差塊：包含兩個卷積層和一個跳躍連接"""
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        # 如果輸入輸出通道不一致，使用 1x1 卷積對齊
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_ch)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class OptimizedResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(OptimizedResUNet, self).__init__()
        
        # Encoder (左側：下採樣)
        self.enc1 = ResBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck (底部)
        self.bottleneck = ResBlock(128, 256)
        
        # Decoder (右側：上採樣)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResBlock(256, 128) # 256 = 128(up) + 128(skip)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResBlock(128, 64) # 128 = 64(up) + 64(skip)
        
        # Final Output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        p1 = self.pool1(s1)
        
        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder
        d2 = self.up2(b)
        d2 = torch.cat([d2, s2], dim=1) # 跳躍連接
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, s1], dim=1) # 跳躍連接
        d1 = self.dec1(d1)
        
        return self.final_conv(d1)
#!/usr/bin/env python3
"""
Custom Model Loader for Traffic Sign Classification
Bu dosya PyTorch modelinin tam yapısını state_dict'ten çıkarır.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),  # fc.0 - no bias
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),  # fc.2 - no bias
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return x * y

class BasicBlockSE(nn.Module):
    """ResNet Basic Block with SE Module"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlockSE, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE module
        self.se = SEBlock(out_channels)
        
        # Shortcut connection
        self.shortcut = downsample if downsample else nn.Identity()
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply SE block
        out = self.se(out)
        
        # Add shortcut connection
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class TrafficSignSENet(nn.Module):
    """Custom SENet for Traffic Sign Classification"""
    
    def __init__(self, num_classes=43):
        super(TrafficSignSENet, self).__init__()
        
        # First conv layer (3x3 instead of 7x7)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet layers with SE blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),                    # classifier.0
            nn.Linear(512, 256),                # classifier.1
            nn.ReLU(inplace=True),              # classifier.2
            nn.BatchNorm1d(256),                # classifier.3
            nn.Dropout(0.3),                    # classifier.4
            nn.Linear(256, num_classes)         # classifier.5
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlockSE(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(BasicBlockSE(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x

def load_traffic_sign_model(model_path, device='cpu'):
    """Doğru model yapısı ile PyTorch modelini yükle"""
    
    print(f"🔄 Model yükleniyor: {model_path}")
    
    # Checkpoint yükle
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Model oluştur
    model = TrafficSignSENet(num_classes=43)
    
    try:
        # State dict yükle
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        
        print(f"✅ Model başarıyla yüklendi (Custom SENet)")
        print(f"📊 Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"📈 Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        
        return model
        
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        
        # Debug: State dict anahtarlarını yazdır
        print("\n🔍 State dict anahtarları:")
        for key in sorted(state_dict.keys())[:10]:
            print(f"   {key}: {state_dict[key].shape}")
        
        print("\n🔍 Model beklenen anahtarları:")
        model_keys = set(model.state_dict().keys())
        state_keys = set(state_dict.keys())
        
        missing = model_keys - state_keys
        extra = state_keys - model_keys
        
        if missing:
            print(f"❌ Eksik anahtarlar: {list(missing)[:5]}...")
        if extra:
            print(f"❓ Fazla anahtarlar: {list(extra)[:5]}...")
        
        raise e

if __name__ == "__main__":
    # Test model loading
    model = load_traffic_sign_model("../lib/model.pth")
    print(f"✅ Model test başarılı!")
    print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}") 
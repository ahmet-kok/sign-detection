#!/usr/bin/env python3
"""
Detailed PyTorch Model Analyzer
Bu script model'in detaylarını analiz eder.
"""

import torch
import torch.nn as nn

def analyze_detailed_model(model_path):
    print(f"🔍 Detaylı Model Analizi: {model_path}")
    print("=" * 60)
    
    # Model yükle
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    print(f"📊 Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"📈 Validation Accuracy: {checkpoint.get('val_acc', 'N/A')}")
    
    # Model state dict analizi
    state_dict = checkpoint['model_state_dict']
    print(f"\n🧠 Model Layers ({len(state_dict)} total):")
    
    layer_info = {}
    for key, tensor in state_dict.items():
        layer_name = key.split('.')[0]
        if layer_name not in layer_info:
            layer_info[layer_name] = []
        layer_info[layer_name].append({
            'name': key,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype)
        })
    
    for layer_name, params in layer_info.items():
        print(f"\n📝 {layer_name}:")
        for param in params[:3]:  # İlk 3 parametreyi göster
            print(f"   - {param['name']}: {param['shape']} ({param['dtype']})")
        if len(params) > 3:
            print(f"   ... ve {len(params)-3} parametre daha")
    
    # Input/Output shape tahminini yapalım
    print(f"\n🔮 Model Tahmini Analizi:")
    
    # İlk conv layer'dan input shape tahmini
    first_conv = None
    for key, tensor in state_dict.items():
        if 'conv' in key.lower() and 'weight' in key:
            first_conv = tensor
            break
    
    if first_conv is not None:
        print(f"📥 Tahmini Input Shape: [batch, {first_conv.shape[1]}, H, W]")
        print(f"   - Channels: {first_conv.shape[1]} (RGB=3, Grayscale=1)")
        print(f"   - First Conv Filters: {first_conv.shape[0]}")
    
    # Son layer'dan output shape tahmini
    last_fc = None
    last_fc_name = None
    for key, tensor in state_dict.items():
        if ('fc' in key.lower() or 'linear' in key.lower() or 'classifier' in key.lower()) and 'weight' in key:
            last_fc = tensor
            last_fc_name = key
    
    if last_fc is not None:
        print(f"📤 Output Shape: {last_fc.shape[0]} classes")
        print(f"   - Layer: {last_fc_name}")
        
        # Bu muhtemelen classification task
        if last_fc.shape[0] < 100:
            print(f"💡 Bu bir Classification modeli gibi görünüyor")
            print(f"   - {last_fc.shape[0]} farklı sınıf tahmin ediyor")
        else:
            print(f"💡 Bu büyük output'lu bir model (detection/segmentation olabilir)")
    
    # Model büyüklüğü
    total_params = sum(tensor.numel() for tensor in state_dict.values())
    print(f"\n📊 Model İstatistikleri:")
    print(f"   - Toplam Parametre: {total_params:,}")
    print(f"   - Model Boyutu: ~{total_params * 4 / (1024**2):.1f} MB")
    
    print("\n" + "="*60)
    
    return checkpoint

if __name__ == "__main__":
    checkpoint = analyze_detailed_model("lib/model.pth") 
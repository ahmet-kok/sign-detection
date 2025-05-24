#!/usr/bin/env python3
"""
PyTorch Model Analyzer
Bu script PyTorch model dosyalarını analiz eder ve yapısını gösterir.
"""

import torch
import os
import sys
from pathlib import Path

def analyze_pytorch_model(model_path):
    """PyTorch model dosyasını analiz et"""
    print(f"\n🔍 Model analizi: {model_path}")
    print("=" * 50)
    
    try:
        # Model dosyasını yükle
        if torch.cuda.is_available():
            model = torch.load(model_path)
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))
        
        print(f"✅ Model başarıyla yüklendi")
        print(f"📁 Dosya boyutu: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        # Model tipini kontrol et
        print(f"🏗️  Model tipi: {type(model)}")
        
        if isinstance(model, dict):
            print("\n📋 Model dictionary keys:")
            for key in model.keys():
                print(f"   - {key}: {type(model[key])}")
            
            # Eğer 'model' key'i varsa
            if 'model' in model:
                actual_model = model['model']
                print(f"\n🧠 Actual model type: {type(actual_model)}")
                if hasattr(actual_model, '__class__'):
                    print(f"🏷️  Model class: {actual_model.__class__.__name__}")
            
            # Eğer 'state_dict' varsa
            if 'state_dict' in model:
                print(f"📊 State dict keys count: {len(model['state_dict'])}")
                print("📝 First 5 layer names:")
                for i, key in enumerate(list(model['state_dict'].keys())[:5]):
                    print(f"   {i+1}. {key}")
        
        elif hasattr(model, 'state_dict'):
            # Doğrudan model objesi
            print(f"🏷️  Model class: {model.__class__.__name__}")
            print(f"📊 Parameters count: {sum(p.numel() for p in model.parameters())}")
            
            # Model architecture
            print(f"\n🏗️  Model architecture:")
            print(model)
            
        else:
            print(f"❓ Unknown model structure")
            print(f"   Available attributes: {dir(model)[:10]}...")
        
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"❌ Hata: {e}")

def find_model_files(directory="."):
    """Belirtilen dizinde model dosyalarını bul"""
    extensions = ['.pth', '.pt', '.pkl']
    model_files = []
    
    for ext in extensions:
        model_files.extend(Path(directory).rglob(f"*{ext}"))
    
    return model_files

def main():
    print("🤖 PyTorch Model Analyzer")
    print("=" * 50)
    
    # Komut satırı argümanı varsa kullan
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if os.path.exists(model_path):
            analyze_pytorch_model(model_path)
        else:
            print(f"❌ Dosya bulunamadı: {model_path}")
    else:
        # Mevcut dizinde model dosyalarını ara
        print("📁 Model dosyaları aranıyor...")
        model_files = find_model_files()
        
        if model_files:
            print(f"\n🎯 {len(model_files)} model dosyası bulundu:")
            for i, file_path in enumerate(model_files, 1):
                print(f"{i}. {file_path}")
                
            print(f"\n{'='*50}")
            print("💡 Kullanım:")
            print(f"   python analyze_model.py <model_dosyası_yolu>")
            print(f"\n📖 Örnek:")
            if model_files:
                print(f"   python analyze_model.py {model_files[0]}")
        else:
            print("❌ Model dosyası bulunamadı!")
            print("💡 Model dosyalarınızın (.pth, .pt, .pkl) bulunduğu dizinde çalıştırın")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
FastAPI Backend for Traffic Sign Classification
Bu backend birden fazla .h5 modelini yükler ve model seçimi yapmanıza olanak tanır.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import json
from typing import List, Dict, Any, Optional
import uvicorn

# Import our model selector
from model_selector import load_model_selector, TRAFFIC_SIGN_CLASSES

# FastAPI app
app = FastAPI(title="Traffic Sign Classifier API with Model Selection", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server (default)
        "http://localhost:3002",  # Next.js dev server (alternative port)
        "http://localhost:3001",  # Additional port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model selector
model_selector = None

@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında tüm modelleri yükle"""
    global model_selector
    try:
        model_selector = load_model_selector()
        available_models = model_selector.get_available_models()
        print(f"🚀 Traffic Sign Classifier API with Model Selection başlatıldı!")
        print(f"📊 Available Models: {available_models}")
    except Exception as e:
        print(f"❌ Startup hatası: {e}")
        model_selector = None

@app.get("/")
async def root():
    return {
        "message": "Universal Traffic Sign Classifier API with Multi-Model Support", 
        "status": "running", 
        "version": "2.0.0",
        "features": ["model_selection", "pytorch_pth_support", "tensorflow_h5_support", "region_detection", "parallel_classification"]
    }

@app.get("/health")
async def health_check():
    if not model_selector:
        return {"status": "unhealthy", "model_selector_loaded": False}
    
    model_info = model_selector.get_model_info()
    return {
        "status": "healthy",
        "model_selector_loaded": True,
        "available_models": model_info['available_models'],
        "total_models": model_info['total_models']
    }

@app.get("/models")
async def get_available_models():
    """Mevcut tüm modellerin listesini getir"""
    if not model_selector:
        raise HTTPException(status_code=503, detail="Model selector henüz yüklenmedi")
    
    model_info = model_selector.get_model_info()
    return {
        "success": True,
        "available_models": model_info['available_models'],
        "total_models": model_info['total_models'],
        "model_details": model_info['model_details'],
        "classes": TRAFFIC_SIGN_CLASSES
    }

@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Belirli bir model hakkında detaylı bilgi getir"""
    if not model_selector:
        raise HTTPException(status_code=503, detail="Model selector henüz yüklenmedi")
    
    try:
        model_info = model_selector.get_model_info(model_name)
        return {
            "success": True,
            "model_name": model_name,
            "model_info": model_info
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/classify")
async def classify_traffic_sign(
    file: UploadFile = File(...),
    model_name: str = Form(...)
):
    """Belirtilen modeli kullanarak tek bir trafik işareti resmini sınıflandır"""
    if not model_selector:
        raise HTTPException(status_code=503, detail="Model selector henüz yüklenmedi")
    
    # Model adını kontrol et
    available_models = model_selector.get_available_models()
    if model_name not in available_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{model_name}' bulunamadı. Mevcut modeller: {available_models}"
        )
    
    # Dosya tipini kontrol et
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Sadece resim dosyaları kabul edilir")
    
    try:
        # Resmi oku
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # RGB'ye çevir
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Tahmin yap
        result = model_selector.predict(image, model_name)
        
        return {
            "success": True,
            "filename": file.filename,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İşlem hatası: {e}")

@app.post("/classify-regions")
async def classify_regions(
    file: UploadFile = File(...),
    regions: str = Form(...),
    model_name: str = Form(...)
):
    """Belirtilen modeli kullanarak resimden crop edilmiş bölgeleri sınıflandır"""
    if not model_selector:
        raise HTTPException(status_code=503, detail="Model selector henüz yüklenmedi")
    
    # Model adını kontrol et
    available_models = model_selector.get_available_models()
    if model_name not in available_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{model_name}' bulunamadı. Mevcut modeller: {available_models}"
        )
    
    print(f"Received regions string: {regions}")
    print(f"Using model: {model_name}")
    
    if not regions:
        raise HTTPException(status_code=400, detail="regions parametresi gerekli")
    
    try:
        # Regions JSON'unu parse et
        region_list = json.loads(regions)
        print(f"Parsed regions: {region_list}")
        
        # Ana resmi yükle
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        results = []
        
        for i, region in enumerate(region_list):
            # Bölgeyi crop et
            x, y, w, h = region['x'], region['y'], region['width'], region['height']
            print(f"Processing region {i}: x={x}, y={y}, w={w}, h={h}")
            
            cropped = image.crop((x, y, x + w, y + h))
            
            # RGB'ye çevir
            if cropped.mode != 'RGB':
                cropped = cropped.convert('RGB')
            
            # Tahmin yap
            prediction = model_selector.predict(cropped, model_name)
            print(f"Region {i} classified as: {prediction['class_name']} ({prediction['confidence']:.2f})")
            
            results.append({
                "region_id": i,
                "region": region,
                "prediction": prediction
            })
        
        return {
            "success": True,
            "total_regions": len(region_list),
            "model_used": model_name,
            "results": results
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        raise HTTPException(status_code=400, detail=f"Geçersiz regions JSON formatı: {e}")
    except Exception as e:
        print(f"Region classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Region classification hatası: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
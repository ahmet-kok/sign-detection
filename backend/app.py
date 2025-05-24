#!/usr/bin/env python3
"""
Hugging Face Spaces App for Traffic Sign Classification
Supports both .pth (PyTorch) and .h5 (TensorFlow) models with API endpoints
"""

import gradio as gr
import requests
from PIL import Image
import numpy as np
import io
import json
from typing import List, Dict, Any
import os

# Import our model selector
from model_selector import load_model_selector, TRAFFIC_SIGN_CLASSES

# Global model selector
model_selector = None

def initialize_models():
    """Initialize all models on startup"""
    global model_selector
    try:
        model_selector = load_model_selector("./models")  # Hugging Face expects models in ./models
        available_models = model_selector.get_available_models()
        print(f"🚀 Loaded {len(available_models)} models: {available_models}")
        return f"✅ Successfully loaded {len(available_models)} models"
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return f"❌ Error: {e}"

def get_available_models_list():
    """Get list of available models for dropdown"""
    if not model_selector:
        return ["No models loaded"]
    return model_selector.get_available_models()

def classify_single_image(image: Image.Image, model_name: str):
    """Classify a single image using specified model"""
    if not model_selector:
        return "❌ Models not loaded yet"
    
    if model_name not in model_selector.get_available_models():
        return f"❌ Model '{model_name}' not found"
    
    try:
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Predict
        result = model_selector.predict(image, model_name)
        
        # Format response
        response = f"""
🎯 **Classification Result**

**Model**: {result['model_name']} ({result['model_type']})
**Input Size**: {result['input_size']}

**🏆 Prediction**: {result['class_name']}
**🎲 Confidence**: {result['confidence']:.2%}

**📊 Top 3 Predictions**:
"""
        for i, pred in enumerate(result['top3_predictions'], 1):
            response += f"\n{i}. {pred['class_name']} - {pred['confidence']:.2%}"
        
        return response
        
    except Exception as e:
        return f"❌ Classification error: {e}"

def classify_multiple_models(image: Image.Image, selected_models: List[str]):
    """Classify image with multiple models for comparison"""
    if not model_selector:
        return "❌ Models not loaded yet"
    
    if not selected_models:
        return "❌ Please select at least one model"
    
    try:
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        results = []
        for model_name in selected_models:
            if model_name in model_selector.get_available_models():
                result = model_selector.predict(image, model_name)
                results.append(result)
        
        if not results:
            return "❌ No valid models selected"
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Format response
        response = f"🔍 **Multi-Model Comparison** ({len(results)} models)\n\n"
        
        for i, result in enumerate(results, 1):
            response += f"""
**{i}. {result['model_name']}** ({result['model_type']}, {result['input_size']})
   Prediction: **{result['class_name']}**
   Confidence: **{result['confidence']:.2%}**
"""
        
        # Add agreement analysis
        predictions = [r['class_name'] for r in results]
        unique_predictions = set(predictions)
        agreement = len([p for p in predictions if p == predictions[0]]) / len(predictions)
        
        response += f"\n📈 **Agreement**: {agreement:.0%}"
        if len(unique_predictions) == 1:
            response += " (All models agree!)"
        else:
            response += f" ({len(unique_predictions)} different predictions)"
        
        return response
        
    except Exception as e:
        return f"❌ Classification error: {e}"

# Initialize models on startup
init_status = initialize_models()

# Create Gradio interface
with gr.Blocks(title="🚦 Traffic Sign Classifier", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🚦 AI Traffic Sign Detection & Classification
    
    **Multi-Model Support**: PyTorch (.pth) + TensorFlow (.h5)  
    **99.45% Accuracy** • **43 German Traffic Sign Classes** • **6 Models Available**
    """)
    
    # Model status
    with gr.Row():
        status_text = gr.Textbox(
            value=init_status,
            label="🔧 System Status",
            interactive=False
        )
    
    # Main interface
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="📸 Upload Traffic Sign Image"
            )
            
        with gr.Column(scale=1):
            # Single model classification
            gr.Markdown("### 🎯 Single Model Classification")
            model_dropdown = gr.Dropdown(
                choices=get_available_models_list(),
                value=get_available_models_list()[0] if get_available_models_list()[0] != "No models loaded" else None,
                label="Select Model"
            )
            classify_btn = gr.Button("🔍 Classify", variant="primary")
            single_result = gr.Textbox(
                label="Result",
                lines=10
            )
            
    # Multi-model comparison
    gr.Markdown("### 📊 Multi-Model Comparison")
    with gr.Row():
        with gr.Column():
            models_checklist = gr.CheckboxGroup(
                choices=get_available_models_list(),
                label="Select Models to Compare",
                value=get_available_models_list()[:3] if len(get_available_models_list()) > 3 else get_available_models_list()
            )
            compare_btn = gr.Button("🔄 Compare Models", variant="secondary")
            
        with gr.Column():
            comparison_result = gr.Textbox(
                label="Comparison Results",
                lines=15
            )
    
    # API endpoint info
    gr.Markdown("""
    ### 🔗 API Endpoints
    
    This Space also provides REST API endpoints:
    - `POST /classify` - Single model classification
    - `POST /classify-regions` - Region-based classification
    - `GET /models` - List available models
    
    Use the API URL: `https://your-username-traffic-sign-classifier.hf.space/`
    """)
    
    # Event handlers
    classify_btn.click(
        fn=classify_single_image,
        inputs=[image_input, model_dropdown],
        outputs=single_result
    )
    
    compare_btn.click(
        fn=classify_multiple_models,
        inputs=[image_input, models_checklist],
        outputs=comparison_result
    )

# Add FastAPI endpoints for external API access
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app that will be mounted
api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.get("/models")
async def api_get_models():
    """API endpoint to get available models"""
    if not model_selector:
        return {"error": "Models not loaded"}
    
    model_info = model_selector.get_model_info()
    return {
        "success": True,
        "available_models": model_info['available_models'],
        "total_models": model_info['total_models'],
        "model_details": model_info['model_details'],
        "classes": TRAFFIC_SIGN_CLASSES
    }

@api.post("/classify")
async def api_classify(
    file: UploadFile = File(...),
    model_name: str = Form(...)
):
    """API endpoint for single image classification"""
    if not model_selector:
        return {"error": "Models not loaded"}
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Predict
        result = model_selector.predict(image, model_name)
        
        return {
            "success": True,
            "filename": file.filename,
            "result": result
        }
        
    except Exception as e:
        return {"error": str(e)}

@api.post("/classify-regions")
async def api_classify_regions(
    file: UploadFile = File(...),
    regions: str = Form(...),
    model_name: str = Form(...)
):
    """API endpoint for region-based classification"""
    if not model_selector:
        return {"error": "Models not loaded"}
    
    try:
        # Parse regions
        region_list = json.loads(regions)
        
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        results = []
        for i, region in enumerate(region_list):
            # Crop region
            x, y, w, h = region['x'], region['y'], region['width'], region['height']
            cropped = image.crop((x, y, x + w, y + h))
            
            if cropped.mode != 'RGB':
                cropped = cropped.convert('RGB')
            
            # Predict
            prediction = model_selector.predict(cropped, model_name)
            
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
        
    except Exception as e:
        return {"error": str(e)}

# Mount FastAPI to Gradio
app = gr.mount_gradio_app(api, app, path="/")

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    ) 
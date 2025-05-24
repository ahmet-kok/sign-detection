# 🚦 Traffic Sign Detection & Classification

**Multi-Model AI System** for German Traffic Sign Recognition

## 🎯 Features

- **🔍 Multi-Model Support**: PyTorch (.pth) + TensorFlow (.h5) models
- **🎯 99.45% Accuracy**: State-of-the-art performance
- **📊 43 Classes**: Complete German traffic sign dataset (GTSRB)
- **🤖 Model Comparison**: Compare predictions across multiple models
- **⚡ Real-time API**: FastAPI backend with Gradio interface
- **🌐 Production Ready**: Deployed on Hugging Face Spaces

## 🚀 Quick Start

### **Upload an Image**
1. Click "📸 Upload Traffic Sign Image"
2. Select a model from the dropdown
3. Click "🔍 Classify"

### **Compare Multiple Models**
1. Upload your image
2. Select multiple models in the checkbox list
3. Click "🔄 Compare Models"
4. See which models agree and their confidence levels

## 🔗 API Endpoints

This Space provides REST API endpoints for integration:

### **GET /models**
Get list of available models and their details
```bash
curl https://YOUR_USERNAME-traffic-sign-classifier.hf.space/models
```

### **POST /classify**
Classify a single image
```bash
curl -X POST \
  -F "file=@traffic_sign.jpg" \
  -F "model_name=model1_h5" \
  https://YOUR_USERNAME-traffic-sign-classifier.hf.space/classify
```

### **POST /classify-regions**
Classify specific regions within an image
```bash
curl -X POST \
  -F "file=@image.jpg" \
  -F "regions=[{\"x\":100,\"y\":100,\"width\":50,\"height\":50}]" \
  -F "model_name=model1_h5" \
  https://YOUR_USERNAME-traffic-sign-classifier.hf.space/classify-regions
```

## 🏗️ Architecture

### **Models Supported**
- **PyTorch Models** (.pth): Custom SENet architecture
- **TensorFlow Models** (.h5): Keras sequential models
- **Input Sizes**: Automatically detected (30x30, 32x32, etc.)
- **Preprocessing**: Automatic normalization and resizing

### **Model Selection System**
```python
# Automatically loads all models from ./models/ directory
model_selector = load_model_selector("./models")

# Get available models
models = model_selector.get_available_models()

# Predict with specific model
result = model_selector.predict(image, "model_name")
```

## 📊 Traffic Sign Classes

**43 German Traffic Sign Classes** (GTSRB Dataset):
- Speed limits (20-120 km/h)
- No entry, No overtaking
- Priority road, Yield
- Mandatory directions
- Warning signs
- And more...

## 🔧 Integration

### **Frontend Integration**
```typescript
// Set your API endpoint
const API_BASE = "https://YOUR_USERNAME-traffic-sign-classifier.hf.space";

// Classify image
const response = await fetch(`${API_BASE}/classify`, {
  method: "POST",
  body: formData
});
```

### **Python Integration**
```python
import requests

# Upload and classify
with open("traffic_sign.jpg", "rb") as f:
    response = requests.post(
        "https://YOUR_USERNAME-traffic-sign-classifier.hf.space/classify",
        files={"file": f},
        data={"model_name": "model1_h5"}
    )
result = response.json()
```

## 🎯 Use Cases

- **🚗 Autonomous Vehicles**: Real-time traffic sign recognition
- **📱 Mobile Apps**: Traffic sign identification for drivers
- **🎓 Education**: Learning traffic signs and rules
- **🔬 Research**: Comparing different model architectures
- **🏭 Traffic Management**: Automated sign inventory systems

## 💡 Tips

- **Model Selection**: Different models may perform better on different sign types
- **Image Quality**: Higher resolution images generally give better results
- **Multiple Models**: Use model comparison to increase confidence
- **API Integration**: Use the REST endpoints for production applications

## 🛠️ Built With

- **🤗 Hugging Face Spaces**: Hosting and deployment
- **🎨 Gradio**: Beautiful web interface
- **⚡ FastAPI**: High-performance API
- **🔥 PyTorch**: Deep learning framework
- **🧠 TensorFlow**: Machine learning platform
- **🖼️ PIL/OpenCV**: Image processing

---

**⭐ Star this Space if you find it useful!**

**🔗 Frontend Demo**: Available on Vercel with OpenCV.js integration

**📧 Questions?** Open an issue or discussion! 
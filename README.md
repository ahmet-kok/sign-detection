# 🚦 AI Traffic Sign Detection & Classification

A modern web application that combines **OpenCV.js** computer vision with **PyTorch** deep learning for accurate traffic sign detection and classification.

## 🎯 Features

### 🔍 **Dual-Stage Detection System**
- **Stage 1**: OpenCV.js region detection using computer vision algorithms
- **Stage 2**: PyTorch AI classification with 99.45% accuracy

### 🧠 **AI-Powered Classification**
- **43 Traffic Sign Classes** (German Traffic Sign Recognition Benchmark)
- **Custom SENet Architecture** with Squeeze-and-Excitation blocks
- **Real-time Classification** via FastAPI backend
- **Top-3 Predictions** with confidence scores

### 🔬 **Advanced Image Processing**
- **Automatic Upscaling** for small images
- **Contrast Enhancement** using CLAHE
- **Multi-step Pipeline** visualization
- **Edge Detection** and contour analysis

### 💻 **Modern Tech Stack**
- **Frontend**: Next.js 15 + TypeScript + TailwindCSS
- **Backend**: FastAPI + PyTorch + OpenCV
- **Computer Vision**: OpenCV.js (client-side)
- **AI Model**: Custom SENet (130MB, 11.4M parameters)

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.8+
- 8GB+ RAM (for model loading)

### 1. Install Dependencies

**Frontend:**
```bash
npm install
```

**Backend:**
```bash
cd backend
pip install -r requirements.txt
```

### 2. Add Your Model
Place your PyTorch model file as `lib/model.pth` (130MB)

### 3. Start Servers

**Backend (Terminal 1):**
```bash
cd backend
python main.py
# Server runs on http://localhost:8000
```

**Frontend (Terminal 2):**
```bash
npm run dev
# App runs on http://localhost:3000
```

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Next.js App   │    │   FastAPI API    │    │  PyTorch Model  │
│                 │    │                  │    │                 │
│ • File Upload   │───▶│ • Image Processing│───▶│ • SENet (43cls) │
│ • OpenCV.js     │    │ • Region Cropping │    │ • 99.45% Acc    │
│ • Visualization │◀───│ • Classification  │◀───│ • Top-3 Preds   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔄 Processing Pipeline

### 1. **Image Upload & Preprocessing**
```typescript
// Automatic upscaling for small images
if (maxDimension < 500px) {
  scaleFactor = 500 / maxDimension;
}
```

### 2. **Computer Vision Detection**
```typescript
// OpenCV.js pipeline
Original → Grayscale → Contrast → Blur → Edges → Contours
```

### 3. **AI Classification**
```python
# PyTorch inference
regions → SENet → [43 classes] → Top-3 predictions
```

### 4. **Results Display**
- Detected regions with bounding boxes
- AI classification with confidence scores
- Top-3 alternative predictions
- Downloadable cropped images

## 🎨 UI Components

### **Detection Results Card**
```tsx
{detection.classification ? (
  <div className="bg-green-50 p-3 rounded-lg">
    <h4>🤖 AI Classification</h4>
    <p>{detection.classification.class_name}</p>
    <span>{confidence}% confident</span>
  </div>
) : (
  <div className="bg-yellow-50 p-3 rounded-lg">
    <p>⚠️ AI classification unavailable</p>
  </div>
)}
```

## 🧪 Model Details

### **Architecture: Custom SENet**
- **Base**: ResNet-18 with SE blocks
- **Input**: 32×32 RGB images
- **Output**: 43 traffic sign classes
- **Accuracy**: 99.45% validation
- **Parameters**: 11,398,763

### **Training Dataset**
- **GTSRB**: German Traffic Sign Recognition Benchmark
- **Classes**: 43 different traffic signs
- **Preprocessing**: Resize, normalize, augmentation

### **SE Block Structure**
```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
```

## 📡 API Endpoints

### **Health Check**
```bash
GET http://localhost:8000/health
```

### **Classify Single Image**
```bash
POST http://localhost:8000/classify
Content-Type: multipart/form-data
Body: file (image)
```

### **Classify Regions**
```bash
POST http://localhost:8000/classify-regions
Content-Type: multipart/form-data
Body: 
  - file (image)
  - regions (JSON array of bounding boxes)
```

## 🔧 Configuration

### **Detection Parameters**
```typescript
// lib/detect.ts
const minArea = 500 * (scaleFactor ** 2);
const maxArea = 100000 * (scaleFactor ** 2);
const aspectRatioRange = [0.5, 2.5];
```

### **Model Parameters**
```python
# backend/model_loader.py
transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

## 🚀 Performance

### **Speed Benchmarks**
- **OpenCV Detection**: ~500ms (client-side)
- **AI Classification**: ~200ms per region (server-side)
- **Total Pipeline**: ~1-2 seconds for typical images

### **Accuracy Metrics**
- **Detection Recall**: ~85% (depends on image quality)
- **Classification Accuracy**: 99.45% (on GTSRB test set)
- **End-to-end Accuracy**: ~84% (combined pipeline)

## 🛠️ Development

### **File Structure**
```
sign-detection/
├── app/
│   └── page.tsx              # Main React component
├── lib/
│   ├── detect.ts             # OpenCV.js detection logic
│   └── model.pth             # PyTorch model (130MB)
├── backend/
│   ├── main.py               # FastAPI server
│   ├── model_loader.py       # Custom model architecture
│   └── requirements.txt      # Python dependencies
├── public/
│   └── opencv.js             # OpenCV.js library (9.9MB)
└── package.json              # Node.js dependencies
```

### **Key Technologies**
- **OpenCV.js**: Computer vision in browser
- **PyTorch**: Deep learning inference
- **FastAPI**: High-performance Python API
- **Next.js**: React framework with TypeScript
- **TailwindCSS**: Utility-first styling

## 🔍 Troubleshooting

### **Common Issues**

**1. Model Loading Error**
```bash
# Check model file exists and is correct format
ls -la lib/model.pth
# Should be ~130MB
```

**2. Backend Connection Failed**
```bash
# Check backend is running
curl http://localhost:8000/health
# Should return: {"status": "healthy", "model_loaded": true}
```

**3. OpenCV.js Loading Error**
```bash
# Check opencv.js file exists
ls -la public/opencv.js
# Should be ~9.9MB
```

**4. Memory Issues**
```bash
# Increase Node.js memory limit
export NODE_OPTIONS="--max-old-space-size=8192"
npm run dev
```

## 📈 Future Enhancements

### **Planned Features**
- [ ] **Real-time Video Detection** using webcam
- [ ] **Mobile App** with React Native
- [ ] **Batch Processing** for multiple images
- [ ] **Custom Model Training** interface
- [ ] **Performance Analytics** dashboard

### **Model Improvements**
- [ ] **YOLOv8 Integration** for better detection
- [ ] **Transformer Architecture** for classification
- [ ] **Multi-scale Detection** for various sign sizes
- [ ] **Data Augmentation** for robustness

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

For issues and questions:
- **GitHub Issues**: [Create an issue](https://github.com/your-repo/issues)
- **Documentation**: Check this README
- **Model Questions**: Refer to PyTorch documentation

---

**Built with ❤️ using OpenCV.js + PyTorch + Next.js**

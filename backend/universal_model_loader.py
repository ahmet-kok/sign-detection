#!/usr/bin/env python3
"""
Universal Model Loader for Traffic Sign Classification
Supports both PyTorch (.pth) and TensorFlow/Keras (.h5) models
"""

import os
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Union, Dict, Any, List, Tuple
from PIL import Image
import io

# Import PyTorch model architecture
from model_loader import load_traffic_sign_model

# Traffic sign class names (German Traffic Sign Recognition Benchmark)
TRAFFIC_SIGN_CLASSES = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 
    'No passing', 'No passing for vehicles over 3.5 metric tons', 
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 
    'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 
    'No entry', 'General caution', 'Dangerous curve to the left', 
    'Dangerous curve to the right', 'Double curve', 'Bumpy road', 
    'Slippery road', 'Road narrows on the right', 'Road work', 
    'Traffic signals', 'Pedestrians', 'Children crossing', 
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 
    'End of all speed and passing limits', 'Turn right ahead', 
    'Turn left ahead', 'Ahead only', 'Go straight or right', 
    'Go straight or left', 'Keep right', 'Keep left', 
    'Roundabout mandatory', 'End of no passing', 
    'End of no passing by vehicles over 3.5 metric tons'
]

class UniversalTrafficSignModel:
    """Universal wrapper that can handle both PyTorch and TensorFlow models"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_type = self._detect_model_type()
        self.model = None
        self.device = None
        
        if self.model_type == 'pytorch':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = load_traffic_sign_model(model_path, self.device)
            self.preprocessing = self._pytorch_preprocessing
        elif self.model_type == 'tensorflow':
            self.model = keras.models.load_model(model_path)
            self.preprocessing = self._tensorflow_preprocessing
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
    
    def _detect_model_type(self) -> str:
        """Detect model type based on file extension"""
        ext = os.path.splitext(self.model_path)[1].lower()
        if ext in ['.pth', '.pt']:
            return 'pytorch'
        elif ext in ['.h5', '.keras']:
            return 'tensorflow'
        else:
            raise ValueError(f"Unsupported model extension: {ext}")
    
    def _pytorch_preprocessing(self, image: Image.Image) -> torch.Tensor:
        """PyTorch preprocessing"""
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        tensor = transform(image).unsqueeze(0)
        return tensor.to(self.device)
    
    def _tensorflow_preprocessing(self, image: Image.Image) -> np.ndarray:
        """TensorFlow preprocessing"""
        # Get the actual input size from the model
        input_shape = self.model.input_shape
        if len(input_shape) == 4:  # (batch, height, width, channels)
            height, width = input_shape[1], input_shape[2]
        else:
            # Fallback to standard size
            height, width = 30, 30
        
        # Resize to model's expected input size
        image = image.resize((width, height))
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32)
        
        # Ensure RGB format
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Normalize to [0, 1]
            img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        else:
            raise ValueError("Image must be RGB format")
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Universal prediction method"""
        if self.model_type == 'pytorch':
            return self._pytorch_predict(image)
        else:
            return self._tensorflow_predict(image)
    
    def _pytorch_predict(self, image: Image.Image) -> Dict[str, Any]:
        """PyTorch prediction"""
        try:
            # Preprocess
            image_tensor = self.preprocessing(image)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Top 3 predictions
                top3_prob, top3_indices = torch.topk(probabilities, 3)
                
                results = {
                    'predicted_class': int(predicted.item()),
                    'class_name': TRAFFIC_SIGN_CLASSES[predicted.item()],
                    'confidence': float(confidence.item()),
                    'model_type': 'pytorch',
                    'top3_predictions': [
                        {
                            'class_id': int(top3_indices[0][i].item()),
                            'class_name': TRAFFIC_SIGN_CLASSES[top3_indices[0][i].item()],
                            'confidence': float(top3_prob[0][i].item())
                        }
                        for i in range(3)
                    ]
                }
                
                return results
                
        except Exception as e:
            raise RuntimeError(f"PyTorch prediction error: {e}")
    
    def _tensorflow_predict(self, image: Image.Image) -> Dict[str, Any]:
        """TensorFlow prediction"""
        try:
            # Preprocess
            image_array = self.preprocessing(image)
            
            # Predict
            predictions = self.model.predict(image_array, verbose=0)
            
            # Get results
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            
            # Top 3 predictions
            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            top3_probs = predictions[0][top3_indices]
            
            results = {
                'predicted_class': predicted_class,
                'class_name': TRAFFIC_SIGN_CLASSES[predicted_class],
                'confidence': confidence,
                'model_type': 'tensorflow',
                'top3_predictions': [
                    {
                        'class_id': int(top3_indices[i]),
                        'class_name': TRAFFIC_SIGN_CLASSES[top3_indices[i]],
                        'confidence': float(top3_probs[i])
                    }
                    for i in range(3)
                ]
            }
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"TensorFlow prediction error: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'num_classes': len(TRAFFIC_SIGN_CLASSES)
        }
        
        if self.model_type == 'pytorch':
            info['device'] = str(self.device)
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            info['total_parameters'] = total_params
        else:
            info['input_shape'] = self.model.input_shape
            info['output_shape'] = self.model.output_shape
            info['total_parameters'] = self.model.count_params()
        
        return info

def load_universal_model(model_path: str) -> UniversalTrafficSignModel:
    """Load either PyTorch or TensorFlow model automatically"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return UniversalTrafficSignModel(model_path) 
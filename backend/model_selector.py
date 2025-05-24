#!/usr/bin/env python3
"""
Universal Model Selector for Traffic Sign Classification
Loads all .pth (PyTorch) and .h5 (TensorFlow) models and allows selection by name
"""

import os
import glob
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Union
from PIL import Image

# Import PyTorch model architecture
from model_loader import TrafficSignSENet, load_traffic_sign_model

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

class UniversalModelSelector:
    """Universal model selector that loads both .pth (PyTorch) and .h5 (TensorFlow) models"""
    
    def __init__(self, models_dir: str = "../lib"):
        self.models_dir = models_dir
        self.models = {}
        self.model_info = {}
        self.load_all_models()
    
    def find_model_files(self) -> Dict[str, List[str]]:
        """Find all .pth and .h5 model files in the directory"""
        
        # Find PyTorch models
        pth_pattern = os.path.join(self.models_dir, "*.pth")
        pth_files = glob.glob(pth_pattern)
        
        # Find TensorFlow models
        h5_pattern = os.path.join(self.models_dir, "*.h5")
        h5_files = glob.glob(h5_pattern)
        
        # Also check for .keras files
        keras_pattern = os.path.join(self.models_dir, "*.keras")
        keras_files = glob.glob(keras_pattern)
        
        return {
            'pytorch': sorted(pth_files),
            'tensorflow': sorted(h5_files + keras_files)
        }
    
    def load_pytorch_model(self, model_path: str) -> tuple:
        """Load a PyTorch model"""
        try:
            model_name = os.path.basename(model_path)
            print(f"🔥 Loading PyTorch model: {model_name}...")
            
            # Load the model using the custom loader
            model = load_traffic_sign_model(model_path, device='cpu')
            
            # Get model info
            total_params = sum(p.numel() for p in model.parameters())
            
            model_info = {
                'path': model_path,
                'input_shape': (None, 3, 32, 32),  # PyTorch format (batch, channels, height, width)
                'output_shape': (None, 43),
                'total_parameters': total_params,
                'input_size': (32, 32),  # Default for PyTorch model
                'model_type': 'pytorch'
            }
            
            print(f"  ✅ {model_name}: (3,32,32) → (43,) ({total_params:,} params)")
            
            return model, model_info
            
        except Exception as e:
            print(f"  ❌ Failed to load PyTorch model {model_name}: {e}")
            raise e
    
    def load_tensorflow_model(self, model_path: str) -> tuple:
        """Load a TensorFlow model"""
        try:
            model_name = os.path.basename(model_path)
            print(f"🧠 Loading TensorFlow model: {model_name}...")
            
            # Load the model
            model = keras.models.load_model(model_path)
            
            # Store model info
            input_shape = model.input_shape
            model_info = {
                'path': model_path,
                'input_shape': input_shape,
                'output_shape': model.output_shape,
                'total_parameters': model.count_params(),
                'input_size': (input_shape[1], input_shape[2]) if len(input_shape) == 4 else (30, 30),
                'model_type': 'tensorflow'
            }
            
            print(f"  ✅ {model_name}: {input_shape} → {model.output_shape} ({model.count_params():,} params)")
            
            return model, model_info
            
        except Exception as e:
            print(f"  ❌ Failed to load TensorFlow model {model_name}: {e}")
            raise e
    
    def load_all_models(self):
        """Load all .pth and .h5 models found in the directory"""
        model_files = self.find_model_files()
        
        total_files = len(model_files['pytorch']) + len(model_files['tensorflow'])
        
        if total_files == 0:
            raise FileNotFoundError(f"No .pth or .h5 model files found in {self.models_dir}")
        
        print(f"🔍 Found {total_files} model(s) to load:")
        print(f"   • PyTorch (.pth): {len(model_files['pytorch'])}")
        print(f"   • TensorFlow (.h5/.keras): {len(model_files['tensorflow'])}")
        
        # Load PyTorch models
        for model_path in model_files['pytorch']:
            try:
                model_name = os.path.basename(model_path)
                model, model_info = self.load_pytorch_model(model_path)
                
                self.models[model_name] = model
                self.model_info[model_name] = model_info
                
            except Exception as e:
                print(f"  ❌ Skipping {model_name}: {e}")
        
        # Load TensorFlow models
        for model_path in model_files['tensorflow']:
            try:
                model_name = os.path.basename(model_path)
                model, model_info = self.load_tensorflow_model(model_path)
                
                self.models[model_name] = model
                self.model_info[model_name] = model_info
                
            except Exception as e:
                print(f"  ❌ Skipping {model_name}: {e}")
        
        if not self.models:
            raise RuntimeError("No models could be loaded successfully")
        
        print(f"\n🎉 Successfully loaded {len(self.models)} model(s)!")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.models.keys())
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """Get info for a specific model or all models"""
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
            return self.model_info[model_name]
        else:
            return {
                'available_models': self.get_available_models(),
                'total_models': len(self.models),
                'model_details': self.model_info
            }
    
    def predict_pytorch(self, image: Image.Image, model: torch.nn.Module, model_name: str) -> Dict[str, Any]:
        """Run prediction on a PyTorch model"""
        try:
            # Get the model's expected input size (32x32 for PyTorch model)
            height, width = 32, 32
            
            # Resize image to model's expected size
            resized_image = image.resize((width, height))
            
            # Convert to array and normalize
            img_array = np.array(resized_image, dtype=np.float32) / 255.0
            
            # Convert to PyTorch format: (channels, height, width)
            img_array = img_array.transpose(2, 0, 1)
            
            # Add batch dimension and convert to tensor
            img_tensor = torch.FloatTensor(img_array).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            # Get results
            probs = probabilities[0].numpy()
            predicted_class = int(np.argmax(probs))
            confidence = float(np.max(probs))
            
            # Top 3 predictions
            top3_indices = np.argsort(probs)[-3:][::-1]
            top3_probs = probs[top3_indices]
            
            return {
                'model_name': model_name,
                'predicted_class': predicted_class,
                'class_name': TRAFFIC_SIGN_CLASSES[predicted_class],
                'confidence': confidence,
                'model_type': 'pytorch',
                'input_size': f"{width}x{height}",
                'top3_predictions': [
                    {
                        'class_id': int(top3_indices[i]),
                        'class_name': TRAFFIC_SIGN_CLASSES[top3_indices[i]],
                        'confidence': float(top3_probs[i])
                    }
                    for i in range(3)
                ]
            }
            
        except Exception as e:
            raise RuntimeError(f"PyTorch prediction error with model '{model_name}': {e}")
    
    def predict_tensorflow(self, image: Image.Image, model, model_name: str) -> Dict[str, Any]:
        """Run prediction on a TensorFlow model"""
        try:
            # Get the model's expected input size
            height, width = self.model_info[model_name]['input_size']
            
            # Resize image to model's expected size
            resized_image = image.resize((width, height))
            
            # Convert to array
            img_array = np.array(resized_image, dtype=np.float32)
            
            # Normalize to [0, 1]
            img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)
            
            # Get results
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            
            # Top 3 predictions
            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            top3_probs = predictions[0][top3_indices]
            
            return {
                'model_name': model_name,
                'predicted_class': predicted_class,
                'class_name': TRAFFIC_SIGN_CLASSES[predicted_class],
                'confidence': confidence,
                'model_type': 'tensorflow',
                'input_size': f"{width}x{height}",
                'top3_predictions': [
                    {
                        'class_id': int(top3_indices[i]),
                        'class_name': TRAFFIC_SIGN_CLASSES[top3_indices[i]],
                        'confidence': float(top3_probs[i])
                    }
                    for i in range(3)
                ]
            }
            
        except Exception as e:
            raise RuntimeError(f"TensorFlow prediction error with model '{model_name}': {e}")
    
    def predict(self, image: Image.Image, model_name: str) -> Dict[str, Any]:
        """Run prediction on a specific model (automatically detects model type)"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        model_type = self.model_info[model_name]['model_type']
        
        if model_type == 'pytorch':
            return self.predict_pytorch(image, model, model_name)
        elif model_type == 'tensorflow':
            return self.predict_tensorflow(image, model, model_name)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

def load_model_selector(models_dir: str = "../lib") -> UniversalModelSelector:
    """Load the universal model selector"""
    return UniversalModelSelector(models_dir) 
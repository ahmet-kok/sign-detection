#!/usr/bin/env python3
"""
Test script to demonstrate .h5 model support
This shows how to create a sample .h5 model for testing
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

def create_sample_h5_model():
    """Create a sample traffic sign classification model in .h5 format"""
    print("🔧 Creating sample .h5 model for testing...")
    
    # Create a simple CNN model for traffic sign classification
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(43, activation='softmax')  # 43 traffic sign classes
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create some dummy data for testing
    dummy_data = np.random.random((100, 32, 32, 3))
    dummy_labels = keras.utils.to_categorical(np.random.randint(0, 43, (100,)), 43)
    
    # Train for just 1 epoch to initialize weights
    print("🏋️ Training model for 1 epoch to initialize weights...")
    model.fit(dummy_data, dummy_labels, epochs=1, verbose=0)
    
    # Save the model
    model_path = "../lib/sample_model.h5"
    model.save(model_path)
    print(f"✅ Sample model saved to: {model_path}")
    
    # Print model info
    print("\n📊 Model Summary:")
    model.summary()
    
    return model_path

def test_universal_loader():
    """Test the universal model loader with the sample .h5 model"""
    from universal_model_loader import load_universal_model
    from PIL import Image
    import numpy as np
    
    # Create sample model if it doesn't exist
    model_path = "../lib/sample_model.h5"
    if not os.path.exists(model_path):
        model_path = create_sample_h5_model()
    
    print(f"\n🔍 Testing universal loader with: {model_path}")
    
    try:
        # Load the model
        model = load_universal_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Get model info
        info = model.get_model_info()
        print("\n📋 Model Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Create a dummy image for testing
        dummy_image = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        
        # Test prediction
        print("\n🔮 Testing prediction...")
        result = model.predict(dummy_image)
        
        print("🎯 Prediction Result:")
        print(f"  Predicted Class: {result['predicted_class']}")
        print(f"  Class Name: {result['class_name']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Model Type: {result['model_type']}")
        
        print("\n🥇 Top 3 Predictions:")
        for i, pred in enumerate(result['top3_predictions']):
            print(f"  {i+1}. {pred['class_name']} ({pred['confidence']:.4f})")
        
        print("\n✅ Universal loader test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing universal loader: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚦 Testing .h5 Model Support for Traffic Sign Classification\n")
    
    # Test the universal loader
    test_universal_loader()
    
    print("\n" + "="*60)
    print("📝 How to use your own .h5 model:")
    print("1. Place your .h5 file in the ../lib/ folder")
    print("2. Name it 'model.h5' or use any of these names:")
    print("   - model.pth (PyTorch)")
    print("   - model.pt (PyTorch)")
    print("   - model.h5 (TensorFlow/Keras)")
    print("   - model.keras (TensorFlow/Keras)")
    print("3. The API will automatically detect and load the correct format")
    print("4. Start the backend with: python main.py")
    print("="*60) 
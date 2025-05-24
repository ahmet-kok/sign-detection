#!/usr/bin/env python3
"""
Test script for the classify-regions API endpoint
"""

import requests
import json
from PIL import Image
import io

def test_classify_regions():
    # Test data - your regions format
    regions_data = [{"x": 9, "y": 4, "width": 53, "height": 55}]
    
    # Create a simple test image
    test_image = Image.new('RGB', (100, 100), color='red')
    img_buffer = io.BytesIO()
    test_image.save(img_buffer, format='JPEG')
    img_buffer.seek(0)
    
    # Prepare the request
    files = {
        'file': ('test.jpg', img_buffer, 'image/jpeg')
    }
    
    data = {
        'regions': json.dumps(regions_data)
    }
    
    print("🧪 Testing classify-regions endpoint...")
    print(f"📝 Regions data: {regions_data}")
    print(f"📤 Sending request to http://localhost:8000/classify-regions")
    
    try:
        response = requests.post(
            'http://localhost:8000/classify-regions',
            files=files,
            data=data,
            timeout=10
        )
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"📈 Results: {json.dumps(result, indent=2)}")
        else:
            print("❌ Error!")
            print(f"💥 Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("💡 Make sure the backend is running on http://localhost:8000")

if __name__ == "__main__":
    test_classify_regions() 
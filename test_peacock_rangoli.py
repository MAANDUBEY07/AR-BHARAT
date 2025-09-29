#!/usr/bin/env python3
"""
Test script for the new peacock rangoli test case
Evaluates the RangoliToKolamConverter against the complex peacock design
"""

import requests
import json
import base64
from pathlib import Path

def test_peacock_rangoli_conversion():
    """Test the peacock rangoli conversion accuracy"""
    
    # Load the test image
    image_path = r"c:\Users\MAAN DUBEY\Downloads\d42715cd936f4343674ae53d1456af19.jpg"
    
    if not Path(image_path).exists():
        print(f"❌ Test image not found: {image_path}")
        return
    
    # Read and encode the image
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode()
    
    # Test with enhanced precision (new converter)
    print("🧪 Testing Peacock Rangoli with Enhanced RangoliToKolamConverter...")
    
    payload = {
        'image': f'data:image/jpeg;base64,{image_data}',
        'precision': 'enhanced'  # This should use the new converter
    }
    
    try:
        response = requests.post('http://localhost:5000/convert', 
                               json=payload, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Conversion Successful!")
            print(f"📊 SVG Length: {len(result.get('svg', ''))} characters")
            print(f"🔍 Metadata: {result.get('metadata', {})}")
            
            # Save the result
            with open(r"c:\Users\MAAN DUBEY\Desktop\SIH project\peacock_kolam_result.svg", 'w') as f:
                f.write(result.get('svg', ''))
            
            print("💾 Result saved to peacock_kolam_result.svg")
            
            # Analyze conversion quality indicators
            svg_content = result.get('svg', '')
            metadata = result.get('metadata', {})
            
            print("\n📈 Quality Analysis:")
            print(f"   • SVG complexity: {'High' if len(svg_content) > 10000 else 'Medium' if len(svg_content) > 5000 else 'Low'}")
            print(f"   • Feature detection: {metadata.get('conversion_type', 'Unknown')}")
            print(f"   • Analyzed features: {', '.join(metadata.get('analyzed_features', []))}")
            
            # Check for specific peacock-related elements
            peacock_indicators = []
            if 'blue' in svg_content.lower():
                peacock_indicators.append("Blue elements detected")
            if 'green' in svg_content.lower():
                peacock_indicators.append("Green elements detected")
            if 'circle' in svg_content.lower():
                peacock_indicators.append("Circular patterns detected")
            if len([line for line in svg_content.split('\n') if 'path' in line]) > 20:
                peacock_indicators.append("Complex geometric paths")
            
            if peacock_indicators:
                print(f"   • Peacock conversion indicators: {', '.join(peacock_indicators)}")
            
            return True
            
        else:
            print(f"❌ Conversion failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def compare_with_traditional_generator():
    """Compare with the old traditional generator approach"""
    
    image_path = r"c:\Users\MAAN DUBEY\Downloads\d42715cd936f4343674ae53d1456af19.jpg"
    
    if not Path(image_path).exists():
        print(f"❌ Test image not found: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode()
    
    print("\n🔄 Testing with Standard precision (for comparison)...")
    
    payload = {
        'image': f'data:image/jpeg;base64,{image_data}',
        'precision': 'standard'
    }
    
    try:
        response = requests.post('http://localhost:5000/convert', 
                               json=payload, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Standard conversion successful!")
            print(f"📊 SVG Length: {len(result.get('svg', ''))} characters")
            print(f"🔍 Metadata: {result.get('metadata', {})}")
            
            # Save comparison result
            with open(r"c:\Users\MAAN DUBEY\Desktop\SIH project\peacock_kolam_standard.svg", 'w') as f:
                f.write(result.get('svg', ''))
            
            return True
        else:
            print(f"❌ Standard conversion failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Standard request failed: {e}")
        return False

if __name__ == "__main__":
    print("🦚 Peacock Rangoli to Kolam Conversion Test")
    print("=" * 50)
    
    # Test the enhanced converter
    enhanced_success = test_peacock_rangoli_conversion()
    
    # Test standard for comparison
    standard_success = compare_with_traditional_generator()
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Enhanced Converter: {'✅ PASS' if enhanced_success else '❌ FAIL'}")
    print(f"   Standard Converter: {'✅ PASS' if standard_success else '❌ FAIL'}")
    
    if enhanced_success and standard_success:
        print("\n🎯 Both converters working. Check SVG outputs for quality comparison!")
    elif enhanced_success:
        print("\n🎯 Enhanced converter working successfully!")
    else:
        print("\n⚠️ Issues detected. Check server logs.")
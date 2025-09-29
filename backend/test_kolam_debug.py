#!/usr/bin/env python3
"""
Test script to debug the SVG coordinate validation issues in kolam_from_image.py
"""

import traceback
import os
import sys

# Ensure the backend directory is in the path
sys.path.insert(0, os.path.dirname(__file__))

from kolam.kolam_from_image import kolam_from_image_py

def test_kolam_generation():
    """Test the kolam generation with various test images"""
    
    # Create a simple test image first
    import cv2
    import numpy as np
    
    # Create a simple 200x200 test image with a circle
    test_img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(test_img, (100, 100), 50, (255, 255, 255), 2)
    
    test_path = os.path.join(os.path.dirname(__file__), 'test_simple_circle.jpg')
    cv2.imwrite(test_path, test_img)
    
    print("Testing kolam generation with simple circle image...")
    
    try:
        # Test with different spacing values
        for spacing in [15, 20, 25, 30]:
            print(f"\nTesting with spacing={spacing}")
            svg_result = kolam_from_image_py(test_path, spacing=spacing)
            
            if svg_result:
                print(f"✅ Success with spacing={spacing}")
                print(f"   SVG length: {len(svg_result)} characters")
                
                # Check for common coordinate validation issues
                if "'nan'" in svg_result.lower():
                    print("   ⚠️  Found NaN values in SVG")
                if "'inf'" in svg_result.lower():
                    print("   ⚠️  Found Infinity values in SVG")
                if "cx='" in svg_result and "cy='" in svg_result:
                    print("   ✅ Circle coordinates found in SVG")
                    
            else:
                print(f"❌ No SVG generated for spacing={spacing}")
                
    except Exception as e:
        print(f"❌ Error during kolam generation: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
    
    # Clean up
    if os.path.exists(test_path):
        os.remove(test_path)

if __name__ == "__main__":
    test_kolam_generation()
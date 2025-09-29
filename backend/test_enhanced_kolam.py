#!/usr/bin/env python3
"""Test the enhanced kolam generator"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kolam.enhanced_polar_kolam_generator import generate_enhanced_kolam_from_image

def test_enhanced_generator():
    """Test the enhanced kolam generator with sample image"""
    
    # Test with a sample image - we'll use the test input image
    test_image_path = "test_input.jpg"
    
    if not os.path.exists(test_image_path):
        print("Creating a simple test image...")
        # Create a simple circular pattern for testing
        import cv2
        import numpy as np
        
        # Create a simple circular/floral pattern
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        center = (150, 150)
        
        # Draw concentric circles in different colors
        cv2.circle(img, center, 120, (0, 100, 255), -1)  # Orange outer
        cv2.circle(img, center, 90, (0, 255, 100), -1)   # Green middle
        cv2.circle(img, center, 60, (255, 100, 0), -1)   # Blue inner
        cv2.circle(img, center, 30, (0, 255, 255), -1)   # Yellow center
        
        # Add some petal-like shapes
        for i in range(8):
            angle = i * np.pi / 4
            x = int(center[0] + 80 * np.cos(angle))
            y = int(center[1] + 80 * np.sin(angle))
            cv2.circle(img, (x, y), 25, (255, 255, 0), -1)
        
        cv2.imwrite(test_image_path, img)
        print(f"Created test image: {test_image_path}")
    
    print("Testing enhanced kolam generator...")
    
    try:
        # Generate kolam with different spacing values
        spacings = [20, 25, 30]
        
        for spacing in spacings:
            print(f"\nGenerating kolam with spacing={spacing}")
            svg_content = generate_enhanced_kolam_from_image(test_image_path, spacing)
            
            # Save the output
            output_filename = f"enhanced_kolam_test_spacing_{spacing}.svg"
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            
            print(f"Generated kolam saved as: {output_filename}")
            print(f"SVG content length: {len(svg_content)} characters")
            
            # Basic validation
            if '<svg' in svg_content and '</svg>' in svg_content:
                print("✓ Valid SVG structure")
            else:
                print("✗ Invalid SVG structure")
        
        print("\n" + "="*50)
        print("Enhanced kolam generator test completed!")
        print("Check the generated SVG files to see the results.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_generator()
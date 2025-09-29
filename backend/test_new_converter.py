#!/usr/bin/env python3
"""Test the new rangoli to kolam converter"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from kolam.rangoli_to_kolam_converter import convert_rangoli_to_kolam

def test_converter():
    # Test with our peacock rangoli image
    img_path = r"c:\Users\MAAN DUBEY\Downloads\d42715cd936f4343674ae53d1456af19.jpg"
    
    print(f"Testing converter with image: {img_path}")
    print(f"Image exists: {os.path.exists(img_path)}")
    
    try:
        svg_output = convert_rangoli_to_kolam(img_path)
        
        print(f"SVG generated successfully!")
        print(f"SVG length: {len(svg_output)} characters")
        print(f"First 200 characters:")
        print(svg_output[:200])
        
        # Check if it has the right background color (light instead of dark)
        if '#f8f8f8' in svg_output:
            print("✅ Correct light background detected")
        else:
            print("❌ Light background not found")
        
        # Check for geometric bird elements
        if 'fill="#4169e1"' in svg_output:  # Blue bird body
            print("✅ Blue bird elements detected")
        else:
            print("❌ Blue bird elements not found")
            
        # Check for petal elements
        if 'stroke="#228b22"' in svg_output:  # Green petals
            print("✅ Green petal elements detected")
        else:
            print("❌ Green petal elements not found")
        
        # Save output for inspection
        with open('converter_test_output.svg', 'w') as f:
            f.write(svg_output)
        print("✅ Output saved to converter_test_output.svg")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_converter()
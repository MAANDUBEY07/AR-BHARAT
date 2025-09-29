#!/usr/bin/env python3

import sys
import traceback
from kolam.kolam_from_image import kolam_from_image_py

def test_kolam_generation():
    try:
        print("Testing kolam generation...")
        img_path = "test_input.jpg"
        print(f"Processing image: {img_path}")
        
        result = kolam_from_image_py(img_path, spacing=15)
        print("SUCCESS: Kolam generated successfully!")
        print(f"SVG length: {len(result)} characters")
        
        # Save SVG for inspection
        with open("test_output.svg", "w") as f:
            f.write(result)
        print("SVG saved to test_output.svg")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kolam_generation()
    sys.exit(0 if success else 1)
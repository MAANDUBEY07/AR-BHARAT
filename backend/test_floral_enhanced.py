#!/usr/bin/env python3
"""Test enhanced kolam with floral image"""

from kolam.enhanced_polar_kolam_generator import generate_enhanced_kolam_from_image

def test_floral():
    print("Testing enhanced kolam with floral image...")
    
    try:
        svg_content = generate_enhanced_kolam_from_image('floral_kolam_test.jpg', spacing=25)
        
        with open('enhanced_floral_kolam.svg', 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        print(f"Generated enhanced floral kolam: enhanced_floral_kolam.svg")
        print(f"SVG content length: {len(svg_content)} characters")
        
        # Check if it contains expected elements
        if '<circle' in svg_content:
            circle_count = svg_content.count('<circle')
            print(f"✓ Contains {circle_count} circles (dots)")
        
        if '<path' in svg_content:
            path_count = svg_content.count('<path')
            print(f"✓ Contains {path_count} paths (connections)")
        
        print("Enhanced floral kolam generation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_floral()
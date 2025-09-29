#!/usr/bin/env python3
"""
Create test rangoli image for web testing
"""

import cv2
import numpy as np

def create_test_rangoli():
    """Create a beautiful test rangoli for web testing"""
    print("Creating test rangoli image...")
    
    # Create a 400x400 canvas
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    center = (200, 200)
    
    # Create a beautiful geometric rangoli
    colors = [
        (255, 100, 150),  # Pink
        (100, 255, 150),  # Green
        (150, 100, 255),  # Purple
        (255, 200, 100),  # Orange
        (100, 200, 255),  # Blue
    ]
    
    # Create concentric patterns
    for ring in range(5):
        radius = 30 + ring * 25
        n_petals = 8 + ring * 2
        color = colors[ring % len(colors)]
        
        # Draw petals
        for i in range(n_petals):
            angle = 2 * np.pi * i / n_petals
            
            # Petal center
            px = int(center[0] + radius * np.cos(angle))
            py = int(center[1] + radius * np.sin(angle))
            
            # Draw petal as filled circle
            cv2.circle(img, (px, py), 8, color, -1)
            
            # Add petal details
            for j in range(3):
                detail_angle = angle + (j - 1) * 0.2
                detail_radius = radius * 0.8
                dx = int(center[0] + detail_radius * np.cos(detail_angle))
                dy = int(center[1] + detail_radius * np.sin(detail_angle))
                cv2.circle(img, (dx, dy), 3, color, -1)
    
    # Add central flower
    cv2.circle(img, center, 15, (255, 255, 255), -1)
    cv2.circle(img, center, 12, (255, 200, 50), -1)
    
    # Add connecting curves
    for i in range(16):
        angle1 = 2 * np.pi * i / 16
        angle2 = 2 * np.pi * (i + 2) / 16
        
        r1, r2 = 80, 140
        x1 = int(center[0] + r1 * np.cos(angle1))
        y1 = int(center[1] + r1 * np.sin(angle1))
        x2 = int(center[0] + r2 * np.cos(angle2))
        y2 = int(center[1] + r2 * np.sin(angle2))
        
        # Curved connection
        cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 2)
    
    # Save the test image
    output_path = "test_rangoli_for_web.jpg"
    cv2.imwrite(output_path, img)
    print(f"Test rangoli created: {output_path}")
    return output_path

if __name__ == "__main__":
    create_test_rangoli()
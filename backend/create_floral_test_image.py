#!/usr/bin/env python3
"""Create a floral test image similar to the screenshot"""

import cv2
import numpy as np
import math

def create_floral_kolam_test():
    # Create a larger image similar to the original
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    center = (200, 200)
    
    # Fill background with warm color
    img[:] = (40, 60, 80)  # Dark background
    
    # Create concentric floral pattern
    # Outer ring - Red petals
    for i in range(12):
        angle = i * 2 * np.pi / 12
        petal_center = (
            int(center[0] + 140 * np.cos(angle)),
            int(center[1] + 140 * np.sin(angle))
        )
        
        # Create petal shape with ellipse
        axes = (25, 15)
        ellipse_angle = math.degrees(angle)
        cv2.ellipse(img, petal_center, axes, ellipse_angle, 0, 360, (0, 100, 200), -1)  # Red petals
        cv2.ellipse(img, petal_center, (20, 12), ellipse_angle, 0, 360, (0, 150, 255), -1)  # Lighter red
    
    # Middle ring - Yellow/Orange petals
    for i in range(8):
        angle = i * 2 * np.pi / 8 + np.pi/8
        petal_center = (
            int(center[0] + 100 * np.cos(angle)),
            int(center[1] + 100 * np.sin(angle))
        )
        
        axes = (30, 20)
        ellipse_angle = math.degrees(angle)
        cv2.ellipse(img, petal_center, axes, ellipse_angle, 0, 360, (0, 200, 255), -1)  # Orange
        cv2.ellipse(img, petal_center, (25, 16), ellipse_angle, 0, 360, (50, 255, 255), -1)  # Yellow
    
    # Inner ring - Green leaves
    for i in range(6):
        angle = i * 2 * np.pi / 6
        petal_center = (
            int(center[0] + 60 * np.cos(angle)),
            int(center[1] + 60 * np.sin(angle))
        )
        
        axes = (20, 35)
        ellipse_angle = math.degrees(angle)
        cv2.ellipse(img, petal_center, axes, ellipse_angle, 0, 360, (100, 200, 0), -1)  # Green
        cv2.ellipse(img, petal_center, (16, 30), ellipse_angle, 0, 360, (150, 255, 50), -1)  # Light green
    
    # Center circle - Golden
    cv2.circle(img, center, 35, (0, 215, 255), -1)  # Golden center
    cv2.circle(img, center, 25, (50, 255, 255), -1)  # Bright golden
    
    # Add some texture and details
    # Add small dots around
    for i in range(24):
        angle = i * 2 * np.pi / 24
        dot_pos = (
            int(center[0] + 180 * np.cos(angle)),
            int(center[1] + 180 * np.sin(angle))
        )
        cv2.circle(img, dot_pos, 8, (100, 255, 255), -1)  # Yellow dots
        cv2.circle(img, dot_pos, 5, (0, 200, 255), -1)    # Orange center
    
    # Save the test image
    cv2.imwrite('floral_kolam_test.jpg', img)
    print("Created floral kolam test image: floral_kolam_test.jpg")
    
    return 'floral_kolam_test.jpg'

if __name__ == "__main__":
    create_floral_kolam_test()
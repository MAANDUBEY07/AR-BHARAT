#!/usr/bin/env python3
"""Create a simple test image for kolam generation"""

import cv2
import numpy as np

# Create a simple test image with some geometric patterns
img = np.zeros((400, 400, 3), dtype=np.uint8)

# Add some simple shapes to test kolam generation
cv2.circle(img, (200, 200), 50, (255, 255, 255), 2)
cv2.circle(img, (150, 150), 30, (255, 255, 255), 2)
cv2.circle(img, (250, 150), 30, (255, 255, 255), 2)
cv2.circle(img, (150, 250), 30, (255, 255, 255), 2)
cv2.circle(img, (250, 250), 30, (255, 255, 255), 2)

# Add some connecting lines
cv2.line(img, (150, 150), (250, 250), (255, 255, 255), 2)
cv2.line(img, (250, 150), (150, 250), (255, 255, 255), 2)

# Save test image
cv2.imwrite('test_kolam.jpg', img)
print("Test image created: test_kolam.jpg")
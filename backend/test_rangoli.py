import cv2
import numpy as np

# Create a more complex test image that might trigger the coordinate error
img = np.zeros((300, 300, 3), dtype=np.uint8)

# Add some complex shapes that might cause coordinate issues
cv2.circle(img, (100, 100), 40, (255, 255, 255), 2)
cv2.circle(img, (200, 100), 40, (255, 255, 255), 2)
cv2.circle(img, (150, 200), 40, (255, 255, 255), 2)
cv2.rectangle(img, (50, 50), (250, 250), (255, 255, 255), 2)

# Add some lines that might create edge cases
cv2.line(img, (0, 0), (300, 300), (255, 255, 255), 2)
cv2.line(img, (300, 0), (0, 300), (255, 255, 255), 2)

cv2.imwrite("test_rangoli.jpg", img)
print("Test image created: test_rangoli.jpg")
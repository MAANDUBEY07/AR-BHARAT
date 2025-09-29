from PIL import Image, ImageDraw
import os

# Create a simple test pattern for kolam generation
def create_test_pattern():
    # Create a 200x200 white canvas
    img = Image.new('RGB', (200, 200), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple diamond pattern with dots
    # This mimics a basic kolam structure
    
    # Draw dots in a grid
    dot_size = 3
    for x in range(20, 180, 20):
        for y in range(20, 180, 20):
            draw.ellipse([x-dot_size, y-dot_size, x+dot_size, y+dot_size], fill='black')
    
    # Draw connecting lines to form a pattern
    # Central diamond
    draw.line([(100, 40), (140, 100), (100, 160), (60, 100), (100, 40)], fill='blue', width=2)
    
    # Corner elements
    draw.line([(40, 40), (80, 60), (60, 80), (40, 40)], fill='red', width=2)
    draw.line([(160, 40), (120, 60), (140, 80), (160, 40)], fill='red', width=2)
    draw.line([(40, 160), (80, 140), (60, 120), (40, 160)], fill='red', width=2)
    draw.line([(160, 160), (120, 140), (140, 120), (160, 160)], fill='red', width=2)
    
    # Save the test image
    img.save('c:\\Users\\MAAN DUBEY\\Desktop\\SIH project\\test_kolam_pattern.png')
    print("Test kolam pattern created at: c:\\Users\\MAAN DUBEY\\Desktop\\SIH project\\test_kolam_pattern.png")

if __name__ == "__main__":
    create_test_pattern()
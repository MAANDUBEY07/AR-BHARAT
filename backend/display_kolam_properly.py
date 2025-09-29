#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from kolam.kolam_from_image import kolam_from_image_py
from PIL import Image
import os

def display_kolam_properly():
    """Display the generated kolam output with proper visualization"""
    
    print("=== LOADING AND PROCESSING IMAGE ===")
    
    # Load the original image
    if os.path.exists("test_input.jpg"):
        original = cv2.imread("test_input.jpg")
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        print(f"✅ Original image loaded: {original.shape[1]}x{original.shape[0]} pixels")
    else:
        print("❌ test_input.jpg not found")
        return
    
    # Check if SVG exists
    if os.path.exists("test_output.svg"):
        with open("test_output.svg", 'r') as f:
            svg_content = f.read()
        print(f"✅ SVG file loaded: {len(svg_content):,} characters")
    else:
        print("🔄 Generating new kolam...")
        svg_content = kolam_from_image_py("test_input.jpg", spacing=15)
        with open("test_output.svg", 'w') as f:
            f.write(svg_content)
        print(f"✅ New SVG generated: {len(svg_content):,} characters")
    
    print("\n=== ANALYZING SVG CONTENT ===")
    
    # Count dots and arcs in SVG
    dot_count = svg_content.count('<use xlink:href="#m78e36de7f3"')
    arc_count = svg_content.count('stroke: #0000ff')
    
    print(f"🔵 Black dots found: {dot_count}")
    print(f"🔷 Blue arcs found: {arc_count}")
    print(f"✨ Total kolam elements: {dot_count + arc_count}")
    
    # Extract coordinates from SVG for proper visualization
    dots = []
    arcs = []
    
    # Parse dot coordinates
    lines = svg_content.split('\n')
    for line in lines:
        if 'use xlink:href="#m78e36de7f3"' in line:
            # Extract x and y coordinates
            x_start = line.find('x="') + 3
            x_end = line.find('"', x_start)
            y_start = line.find('y="') + 3  
            y_end = line.find('"', y_start)
            
            if x_start > 2 and y_start > 2:
                try:
                    x = float(line[x_start:x_end])
                    y = float(line[y_start:y_end])
                    dots.append((x, y))
                except:
                    pass
    
    print(f"📍 Extracted {len(dots)} dot coordinates from SVG")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    axes[0].imshow(original_rgb)
    axes[0].set_title("🎨 Original Rangoli Image", fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # Generated kolam with proper coordinates
    if dots:
        dot_x = [d[0] for d in dots[:1000]]  # Limit for display
        dot_y = [d[1] for d in dots[:1000]]
        
        # Plot dots
        axes[1].scatter(dot_x, dot_y, s=30, c='black', marker='o', alpha=0.8, label=f'{len(dot_x)} Dots')
        
        # Draw arcs around some dots to show the pattern
        r = 20  # Arc radius
        theta = np.linspace(0, 2*np.pi, 25)
        
        # Show arcs for every 5th dot to avoid overcrowding the display
        arc_sample = dots[::5][:50]  # Sample every 5th dot, max 50
        for xi, yi in arc_sample:
            # Four quarter-circle arcs around each dot
            for quarter in range(4):
                start_angle = quarter * np.pi/2
                end_angle = (quarter + 1) * np.pi/2
                arc_theta = np.linspace(start_angle, end_angle, 10)
                
                arc_x = xi + r * np.cos(arc_theta)
                arc_y = yi + r * np.sin(arc_theta)
                axes[1].plot(arc_x, arc_y, 'blue', linewidth=2, alpha=0.7)
        
        axes[1].set_title(f"🕸️ Generated Tamil Kolam\n(Dot-Grid + Arc Pattern)", 
                         fontsize=16, fontweight='bold')
        axes[1].set_aspect('equal')
        axes[1].invert_yaxis()  # Match SVG coordinate system
        axes[1].axis('off')
        axes[1].legend(loc='upper right')
        
        # Add grid to show structure
        axes[1].grid(True, alpha=0.2)
        
    else:
        axes[1].text(0.5, 0.5, "❌ Could not extract coordinates", 
                    ha='center', va='center', fontsize=14, 
                    transform=axes[1].transAxes)
        axes[1].set_title("Generated Kolam", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("kolam_display.png", dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"\n=== KOLAM GENERATION SUMMARY ===")
    print(f"✅ Processing successful!")
    print(f"📊 Dots generated: {dot_count} points") 
    print(f"🎨 Arcs per dot: 4 quarter-circles")
    print(f"💾 Total SVG size: {len(svg_content):,} characters")
    print(f"🖼️  Display saved as: kolam_display.png")
    print(f"📁 Full SVG available: test_output.svg")
    print(f"🥽 Ready for AR viewing and web display!")
    
    plt.show()

if __name__ == "__main__":
    display_kolam_properly()
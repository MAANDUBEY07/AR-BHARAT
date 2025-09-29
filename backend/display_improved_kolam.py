#!/usr/bin/env python3
"""
Display improved kolam patterns with authentic Tamil geometric designs
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from kolam.kolam_from_image import kolam_from_image_py
from PIL import Image
import os
import xml.etree.ElementTree as ET

def display_improved_kolam():
    """Display the new improved kolam output with geometric patterns"""
    
    print("🎨 === IMPROVED KOLAM GENERATION ===")
    
    # Load the original image
    if os.path.exists("test_input.jpg"):
        original = cv2.imread("test_input.jpg")
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        print(f"✅ Original image loaded: {original.shape[1]}x{original.shape[0]} pixels")
    else:
        print("❌ test_input.jpg not found")
        return
    
    print("🔄 Generating improved kolam with geometric patterns...")
    
    # Generate new kolam with improved algorithm
    try:
        svg_content = kolam_from_image_py("test_input.jpg", spacing=25)
        
        # Save the new SVG
        with open("improved_kolam_output.svg", 'w') as f:
            f.write(svg_content)
        
        print(f"✅ New geometric kolam generated: {len(svg_content):,} characters")
        
    except Exception as e:
        print(f"❌ Error generating kolam: {str(e)}")
        return
    
    print("\n📊 === ANALYZING IMPROVED KOLAM ===")
    
    # Analyze SVG content
    line_count = svg_content.count('<path d=')
    circle_count = svg_content.count('<circle')
    dot_count = svg_content.count('markersize')
    
    print(f"🔵 Dots (grid points): {circle_count}")
    print(f"🌀 Geometric patterns: {line_count}")
    print(f"✨ Total kolam elements: {circle_count + line_count}")
    print(f"🎯 Pattern complexity: {'High' if line_count > 50 else 'Medium' if line_count > 20 else 'Low'}")
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image
    axes[0].imshow(original_rgb)
    axes[0].set_title("🎨 Original Rangoli Image", fontsize=18, fontweight='bold', pad=20)
    axes[0].axis('off')
    
    # Load and display the generated SVG
    try:
        # Parse SVG to extract drawing commands for matplotlib
        display_svg_with_matplotlib(axes[1], svg_content)
        axes[1].set_title(f"🕸️ Improved Tamil Kolam\n(Geometric Patterns + Grid Structure)", 
                         fontsize=18, fontweight='bold', pad=20)
        axes[1].axis('off')
        axes[1].set_aspect('equal')
        
    except Exception as e:
        print(f"⚠️ Could not parse SVG for display: {str(e)}")
        axes[1].text(0.5, 0.5, f"✅ Kolam Generated!\n🎯 {line_count} geometric patterns\n📁 Saved as improved_kolam_output.svg", 
                    ha='center', va='center', fontsize=16, 
                    transform=axes[1].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1].set_title("🕸️ Improved Tamil Kolam", fontsize=18, fontweight='bold', pad=20)
    
    # Add comparison legend
    fig.suptitle("Rangoli-to-Kolam Conversion: Authentic Tamil Geometric Patterns", 
                fontsize=22, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig("improved_kolam_comparison.png", dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"\n🎉 === KOLAM GENERATION COMPLETE ===")
    print(f"✅ Improved kolam successfully generated!")
    print(f"📈 Pattern enhancement: Structured geometric designs")
    print(f"🎨 Visual style: Traditional Tamil kolam aesthetics")
    print(f"💾 Files saved:")
    print(f"   • improved_kolam_output.svg - Vector kolam pattern")
    print(f"   • improved_kolam_comparison.png - Visual comparison")
    print(f"🥽 Ready for AR viewing and web display!")
    
    # Show comparison
    plt.show()

def display_svg_with_matplotlib(ax, svg_content):
    """Parse SVG and display with matplotlib"""
    try:
        # Parse circles (dots)
        circles = []
        lines = svg_content.split('\n')
        
        for line in lines:
            if '<circle' in line:
                # Extract circle parameters
                cx_start = line.find('cx="') + 4
                cx_end = line.find('"', cx_start)
                cy_start = line.find('cy="') + 4
                cy_end = line.find('"', cy_start)
                r_start = line.find('r="') + 3
                r_end = line.find('"', r_start)
                
                if cx_start > 3 and cy_start > 3 and r_start > 2:
                    try:
                        cx = float(line[cx_start:cx_end])
                        cy = float(line[cy_start:cy_end])
                        r = float(line[r_start:r_end])
                        circles.append((cx, cy, r))
                    except:
                        pass
        
        # Draw circles
        if circles:
            for cx, cy, r in circles[:100]:  # Limit for display
                ax.plot(cx, cy, 'ko', markersize=8, alpha=0.9)
        
        # Parse and draw paths (geometric patterns)
        path_count = 0
        for line in lines:
            if '<path d=' in line:
                # Extract path data
                d_start = line.find('d="') + 3
                d_end = line.find('"', d_start)
                
                if d_start > 2:
                    try:
                        path_data = line[d_start:d_end]
                        x_coords, y_coords = parse_path_data(path_data)
                        
                        if len(x_coords) > 1:
                            ax.plot(x_coords, y_coords, 'b-', linewidth=2.5, alpha=0.7)
                            path_count += 1
                            
                            if path_count >= 100:  # Limit for display performance
                                break
                    except:
                        pass
        
        print(f"📊 Displayed: {len(circles)} dots and {path_count} geometric patterns")
        
        # Set appropriate limits
        if circles:
            x_coords = [c[0] for c in circles]
            y_coords = [c[1] for c in circles]
            
            if x_coords and y_coords:
                margin = 20
                ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
                ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
        
    except Exception as e:
        print(f"Error parsing SVG: {str(e)}")
        ax.text(0.5, 0.5, "SVG Generated Successfully\nView improved_kolam_output.svg", 
               ha='center', va='center', transform=ax.transAxes)

def parse_path_data(path_data):
    """Parse SVG path data to extract coordinates"""
    x_coords = []
    y_coords = []
    
    # Simple parsing for M and L commands
    parts = path_data.replace('M', ' M ').replace('L', ' L ').split()
    
    i = 0
    while i < len(parts):
        if parts[i] == 'M' or parts[i] == 'L':
            if i + 1 < len(parts):
                coord = parts[i + 1]
                if ',' in coord:
                    try:
                        x, y = coord.split(',')
                        x_coords.append(float(x))
                        y_coords.append(float(y))
                    except:
                        pass
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    return x_coords, y_coords

if __name__ == "__main__":
    display_improved_kolam()
#!/usr/bin/env python3
"""
Authentic Traditional Kolam Generator
Focuses on creating genuine kolam patterns with proper dot grids and traditional connections
"""

import cv2
import numpy as np
import svgwrite
import math
from typing import Tuple, List, Dict, Any
from sklearn.cluster import KMeans

class AuthenticKolamGenerator:
    def __init__(self):
        self.traditional_patterns = {
            'simple': self._generate_simple_kolam,
            'floral': self._generate_floral_kolam,
            'geometric': self._generate_geometric_kolam,
            'complex': self._generate_complex_kolam
        }
    
    def generate_authentic_kolam(self, img_path: str, spacing: int = 30) -> str:
        """Generate authentic traditional kolam patterns"""
        img = cv2.imread(img_path)
        if img is None:
            return self._generate_default_kolam(spacing)
        
        # Analyze image to choose appropriate traditional pattern
        pattern_type = self._analyze_for_traditional_pattern(img)
        print(f"Selected traditional pattern: {pattern_type}")
        
        # Generate the appropriate kolam type
        return self.traditional_patterns[pattern_type](img, spacing)
    
    def _analyze_for_traditional_pattern(self, img: np.ndarray) -> str:
        """Analyze image to select appropriate traditional kolam pattern"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate complexity based on edges and features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)
        
        # Detect circular shapes (common in traditional kolams)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 30)
        has_circles = circles is not None and len(circles[0]) > 2
        
        # Color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        bright_colors = np.mean(hsv[:,:,2] > 180)  # High value (brightness)
        
        print(f"Analysis - Edge density: {edge_density:.3f}, Circles: {has_circles}, Bright: {bright_colors:.3f}")
        
        if edge_density > 0.15 and has_circles:
            return 'complex'
        elif bright_colors > 0.3 or has_circles:
            return 'floral'
        elif edge_density > 0.08:
            return 'geometric'
        else:
            return 'simple'
    
    def _generate_simple_kolam(self, img: np.ndarray, spacing: int) -> str:
        """Generate simple traditional kolam with basic dot grid"""
        height, width = img.shape[:2]
        
        # Resize for consistency
        max_size = 400
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            width = int(width * scale)
            height = int(height * scale)
            img = cv2.resize(img, (width, height))
        
        # Create SVG
        dwg = svgwrite.Drawing(size=(f'{width}px', f'{height}px'), 
                              viewBox=f'0 0 {width} {height}')
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='#0a0a0a'))
        
        # Extract colors
        colors = self._extract_traditional_colors(img)
        
        # Create traditional dot grid (odd numbers for symmetry)
        rows = ((height // spacing) // 2) * 2 + 1  # Ensure odd number
        cols = ((width // spacing) // 2) * 2 + 1   # Ensure odd number
        
        # Center the grid
        start_x = (width - (cols - 1) * spacing) // 2
        start_y = (height - (rows - 1) * spacing) // 2
        
        print(f"Creating {rows}x{cols} dot grid, spacing={spacing}")
        
        # Generate dots and store positions
        dots = []
        for row in range(rows):
            for col in range(cols):
                x = start_x + col * spacing
                y = start_y + row * spacing
                
                # Add dot
                dwg.add(dwg.circle(
                    center=(x, y), r=4,
                    fill=colors['dot'],
                    opacity=0.9
                ))
                
                dots.append((x, y, row, col))
        
        # Connect dots with traditional patterns
        self._add_traditional_connections(dwg, dots, colors, spacing, 'simple')
        
        return dwg.tostring()
    
    def _generate_floral_kolam(self, img: np.ndarray, spacing: int) -> str:
        """Generate floral kolam with petal-like connections"""
        height, width = img.shape[:2]
        
        # Resize for consistency
        max_size = 400
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            width = int(width * scale)
            height = int(height * scale)
            img = cv2.resize(img, (width, height))
        
        # Create SVG
        dwg = svgwrite.Drawing(size=(f'{width}px', f'{height}px'), 
                              viewBox=f'0 0 {width} {height}')
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='#0a0a0a'))
        
        # Extract colors
        colors = self._extract_traditional_colors(img)
        
        # Create flower-based dot arrangement
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) // 3
        
        dots = []
        
        # Central dot
        dwg.add(dwg.circle(center=(center_x, center_y), r=5, fill=colors['dot'], opacity=0.9))
        dots.append((center_x, center_y, 'center'))
        
        # Create petal arrangements
        n_petals = 8
        for petal in range(n_petals):
            angle = (2 * np.pi * petal) / n_petals
            
            # Multiple dots per petal
            for r_idx, radius in enumerate([max_radius * 0.3, max_radius * 0.6, max_radius * 0.9]):
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                
                # Add main petal dot
                dot_size = 5 if r_idx == 0 else 4
                dwg.add(dwg.circle(center=(x, y), r=dot_size, fill=colors['dot'], opacity=0.9))
                dots.append((x, y, f'petal_{petal}_{r_idx}'))
                
                # Add side dots for fuller petals
                if r_idx > 0:
                    offset_angle = 0.3  # radians
                    for side in [-1, 1]:
                        side_angle = angle + side * offset_angle
                        side_x = center_x + radius * 0.7 * np.cos(side_angle)
                        side_y = center_y + radius * 0.7 * np.sin(side_angle)
                        
                        dwg.add(dwg.circle(center=(side_x, side_y), r=3, fill=colors['accent'], opacity=0.8))
                        dots.append((side_x, side_y, f'side_{petal}_{r_idx}_{side}'))
        
        # Add floral connections
        self._add_floral_connections(dwg, dots, colors)
        
        return dwg.tostring()
    
    def _generate_geometric_kolam(self, img: np.ndarray, spacing: int) -> str:
        """Generate geometric kolam with traditional patterns"""
        height, width = img.shape[:2]
        
        # Resize for consistency
        max_size = 400
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            width = int(width * scale)
            height = int(height * scale)
            img = cv2.resize(img, (width, height))
        
        # Create SVG
        dwg = svgwrite.Drawing(size=(f'{width}px', f'{height}px'), 
                              viewBox=f'0 0 {width} {height}')
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='#0a0a0a'))
        
        # Extract colors
        colors = self._extract_traditional_colors(img)
        
        # Create geometric dot pattern
        center_x, center_y = width // 2, height // 2
        
        dots = []
        
        # Create concentric squares/diamonds
        for level in range(1, 5):
            size = level * spacing
            
            # Create square/diamond arrangement
            n_points = 4 * level * 2  # More points for outer levels
            for i in range(n_points):
                angle = (2 * np.pi * i) / n_points
                x = center_x + size * np.cos(angle)
                y = center_y + size * np.sin(angle)
                
                # Keep within bounds
                if 10 <= x <= width-10 and 10 <= y <= height-10:
                    dot_size = 5 - level if level <= 3 else 2
                    dwg.add(dwg.circle(center=(x, y), r=dot_size, fill=colors['dot'], opacity=0.9))
                    dots.append((x, y, level, i))
        
        # Central dot
        dwg.add(dwg.circle(center=(center_x, center_y), r=6, fill=colors['accent'], opacity=1.0))
        dots.append((center_x, center_y, 0, 0))
        
        # Add geometric connections
        self._add_geometric_connections(dwg, dots, colors)
        
        return dwg.tostring()
    
    def _generate_complex_kolam(self, img: np.ndarray, spacing: int) -> str:
        """Generate complex traditional kolam"""
        height, width = img.shape[:2]
        
        # Resize for consistency
        max_size = 400
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            width = int(width * scale)
            height = int(height * scale)
            img = cv2.resize(img, (width, height))
        
        # Create SVG
        dwg = svgwrite.Drawing(size=(f'{width}px', f'{height}px'), 
                              viewBox=f'0 0 {width} {height}')
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='#0a0a0a'))
        
        # Extract colors
        colors = self._extract_traditional_colors(img)
        
        # Create complex multi-layer kolam
        center_x, center_y = width // 2, height // 2
        
        dots = []
        
        # Layer 1: Central mandala
        for ring in range(1, 4):
            n_dots = ring * 6
            radius = ring * spacing * 0.8
            
            for i in range(n_dots):
                angle = (2 * np.pi * i) / n_dots
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                
                dwg.add(dwg.circle(center=(x, y), r=4, fill=colors['dot'], opacity=0.9))
                dots.append((x, y, f'mandala_{ring}_{i}'))
        
        # Layer 2: Corner patterns
        corner_offset = min(width, height) * 0.3
        corners = [
            (center_x - corner_offset, center_y - corner_offset),
            (center_x + corner_offset, center_y - corner_offset),
            (center_x + corner_offset, center_y + corner_offset),
            (center_x - corner_offset, center_y + corner_offset)
        ]
        
        for corner_idx, (cx, cy) in enumerate(corners):
            if 20 <= cx <= width-20 and 20 <= cy <= height-20:
                # Small flower pattern at each corner
                for i in range(6):
                    angle = (2 * np.pi * i) / 6
                    x = cx + spacing * 0.5 * np.cos(angle)
                    y = cy + spacing * 0.5 * np.sin(angle)
                    
                    if 10 <= x <= width-10 and 10 <= y <= height-10:
                        dwg.add(dwg.circle(center=(x, y), r=3, fill=colors['accent'], opacity=0.8))
                        dots.append((x, y, f'corner_{corner_idx}_{i}'))
        
        # Central dot
        dwg.add(dwg.circle(center=(center_x, center_y), r=6, fill=colors['primary'], opacity=1.0))
        dots.append((center_x, center_y, 'center'))
        
        # Add complex connections
        self._add_complex_connections(dwg, dots, colors)
        
        return dwg.tostring()
    
    def _generate_default_kolam(self, spacing: int) -> str:
        """Generate a beautiful default kolam when no image is provided"""
        width, height = 400, 400
        
        dwg = svgwrite.Drawing(size=(f'{width}px', f'{height}px'), 
                              viewBox=f'0 0 {width} {height}')
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='#0a0a0a'))
        
        # Traditional colors
        colors = {
            'dot': 'white',
            'primary': '#E24A90',
            'secondary': '#90E24A',
            'accent': '#4AA9E2'
        }
        
        # Create traditional 13x13 dot grid
        rows = cols = 13
        start_x = start_y = (width - (cols - 1) * spacing) // 2
        
        dots = []
        for row in range(rows):
            for col in range(cols):
                x = start_x + col * spacing
                y = start_y + row * spacing
                
                dwg.add(dwg.circle(center=(x, y), r=4, fill=colors['dot'], opacity=0.9))
                dots.append((x, y, row, col))
        
        # Add traditional connections
        self._add_traditional_connections(dwg, dots, colors, spacing, 'default')
        
        return dwg.tostring()
    
    def _extract_traditional_colors(self, img: np.ndarray) -> Dict[str, str]:
        """Extract colors suitable for traditional kolam"""
        # Reshape image for color clustering
        data = img.reshape((-1, 3))
        
        # Use K-means to find dominant colors
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(data)
        colors_bgr = kmeans.cluster_centers_.astype(int)
        
        # Convert to hex
        def bgr_to_hex(bgr):
            return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"
        
        hex_colors = [bgr_to_hex(color) for color in colors_bgr]
        
        return {
            'dot': 'white',  # Always white for traditional kolam
            'primary': hex_colors[0] if hex_colors[0] != '#000000' else '#E24A90',
            'secondary': hex_colors[1] if hex_colors[1] != '#000000' else '#90E24A',
            'accent': hex_colors[2] if hex_colors[2] != '#000000' else '#4AA9E2'
        }
    
    def _add_traditional_connections(self, dwg: svgwrite.Drawing, dots: List, colors: Dict, spacing: int, pattern_type: str):
        """Add traditional kolam connections between dots"""
        if pattern_type == 'simple' or pattern_type == 'default':
            # Simple traditional connections
            for i, (x1, y1, row1, col1) in enumerate(dots):
                for j, (x2, y2, row2, col2) in enumerate(dots[i+1:], i+1):
                    # Connect adjacent dots
                    if abs(row1 - row2) + abs(col1 - col2) == 1:
                        dwg.add(dwg.line(start=(x1, y1), end=(x2, y2), 
                                       stroke=colors['primary'], stroke_width=2, opacity=0.7))
                    
                    # Connect diagonal neighbors for traditional pattern
                    elif abs(row1 - row2) == 1 and abs(col1 - col2) == 1:
                        # Curved diagonal connection
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        ctrl_x = mid_x + 5 * (1 if col2 > col1 else -1)
                        ctrl_y = mid_y + 5 * (1 if row2 > row1 else -1)
                        
                        path_data = f"M {x1} {y1} Q {ctrl_x} {ctrl_y} {x2} {y2}"
                        dwg.add(dwg.path(d=path_data, stroke=colors['secondary'], 
                                       stroke_width=1.5, fill='none', opacity=0.6))
    
    def _add_floral_connections(self, dwg: svgwrite.Drawing, dots: List, colors: Dict):
        """Add floral connections for flower-like kolam patterns"""
        center_dots = [dot for dot in dots if 'center' in str(dot[2])]
        petal_dots = [dot for dot in dots if 'petal' in str(dot[2])]
        
        if center_dots:
            cx, cy = center_dots[0][0], center_dots[0][1]
            
            # Connect center to all petal dots
            for px, py, petal_info in petal_dots:
                # Curved connection from center to petal
                mid_x, mid_y = (cx + px) / 2, (cy + py) / 2
                # Add slight curve
                offset = 10
                ctrl_x = mid_x + offset * np.cos(np.arctan2(py - cy, px - cx) + np.pi/2)
                ctrl_y = mid_y + offset * np.sin(np.arctan2(py - cy, px - cx) + np.pi/2)
                
                path_data = f"M {cx} {cy} Q {ctrl_x} {ctrl_y} {px} {py}"
                dwg.add(dwg.path(d=path_data, stroke=colors['primary'], 
                               stroke_width=2, fill='none', opacity=0.8))
        
        # Connect petals in circular fashion
        petal_groups = {}
        for px, py, petal_info in petal_dots:
            if isinstance(petal_info, str) and 'petal_' in petal_info:
                parts = petal_info.split('_')
                if len(parts) >= 3:
                    petal_num = parts[1]
                    if petal_num not in petal_groups:
                        petal_groups[petal_num] = []
                    petal_groups[petal_num].append((px, py))
        
        # Connect adjacent petals
        for petal_num, petal_points in petal_groups.items():
            for i in range(len(petal_points) - 1):
                x1, y1 = petal_points[i]
                x2, y2 = petal_points[i + 1]
                
                dwg.add(dwg.line(start=(x1, y1), end=(x2, y2), 
                               stroke=colors['secondary'], stroke_width=1.5, opacity=0.7))
    
    def _add_geometric_connections(self, dwg: svgwrite.Drawing, dots: List, colors: Dict):
        """Add geometric connections for geometric kolam patterns"""
        # Group dots by level
        level_groups = {}
        for x, y, level, index in dots:
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append((x, y, index))
        
        # Connect dots within same level (circular)
        for level, level_dots in level_groups.items():
            if level == 0:  # Skip center dot
                continue
            
            level_dots.sort(key=lambda d: d[2])  # Sort by index
            
            for i in range(len(level_dots)):
                x1, y1, _ = level_dots[i]
                x2, y2, _ = level_dots[(i + 1) % len(level_dots)]
                
                # Curved connection for more organic look
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Calculate curve control point
                dx, dy = x2 - x1, y2 - y1
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    perp_x = -dy / length * 8
                    perp_y = dx / length * 8
                    ctrl_x = mid_x + perp_x
                    ctrl_y = mid_y + perp_y
                    
                    path_data = f"M {x1} {y1} Q {ctrl_x} {ctrl_y} {x2} {y2}"
                    dwg.add(dwg.path(d=path_data, stroke=colors['primary'], 
                                   stroke_width=2, fill='none', opacity=0.8))
        
        # Connect between levels (radial)
        center = level_groups.get(0, [(200, 200, 0)])[0]  # Default center if not found
        cx, cy = center[0], center[1]
        
        for level in range(1, 4):
            if level in level_groups:
                for x, y, _ in level_groups[level][::2]:  # Every other dot for less crowding
                    dwg.add(dwg.line(start=(cx, cy), end=(x, y), 
                                   stroke=colors['secondary'], stroke_width=1.5, opacity=0.6))
    
    def _add_complex_connections(self, dwg: svgwrite.Drawing, dots: List, colors: Dict):
        """Add complex connections for elaborate kolam patterns"""
        # Group by pattern type
        mandala_dots = [dot for dot in dots if isinstance(dot[2], str) and 'mandala' in dot[2]]
        corner_dots = [dot for dot in dots if isinstance(dot[2], str) and 'corner' in dot[2]]
        center_dots = [dot for dot in dots if isinstance(dot[2], str) and 'center' in dot[2]]
        
        if center_dots:
            cx, cy = center_dots[0][0], center_dots[0][1]
            
            # Connect center to mandala dots with varied patterns
            for i, (mx, my, _) in enumerate(mandala_dots):
                if i % 3 == 0:  # Not all connections to avoid overcrowding
                    # Spiral-like connection
                    angle = math.atan2(my - cy, mx - cx)
                    mid_dist = math.sqrt((mx - cx)**2 + (my - cy)**2) / 2
                    
                    # Create spiral control points
                    spiral_angle = angle + math.pi / 4
                    ctrl1_x = cx + mid_dist * 0.5 * math.cos(spiral_angle)
                    ctrl1_y = cy + mid_dist * 0.5 * math.sin(spiral_angle)
                    
                    spiral_angle2 = angle - math.pi / 4
                    ctrl2_x = cx + mid_dist * 1.5 * math.cos(spiral_angle2)
                    ctrl2_y = cy + mid_dist * 1.5 * math.sin(spiral_angle2)
                    
                    path_data = f"M {cx} {cy} Q {ctrl1_x} {ctrl1_y} {ctrl2_x} {ctrl2_y} T {mx} {my}"
                    dwg.add(dwg.path(d=path_data, stroke=colors['primary'], 
                                   stroke_width=1.8, fill='none', opacity=0.7))
        
        # Connect corner patterns to nearest mandala dots
        for corner_x, corner_y, corner_info in corner_dots:
            # Find nearest mandala dots
            distances = []
            for mx, my, _ in mandala_dots:
                dist = math.sqrt((corner_x - mx)**2 + (corner_y - my)**2)
                distances.append((dist, mx, my))
            
            # Connect to 2 nearest mandala dots
            distances.sort()
            for i in range(min(2, len(distances))):
                _, mx, my = distances[i]
                
                # Curved connection
                mid_x, mid_y = (corner_x + mx) / 2, (corner_y + my) / 2
                offset = 15
                ctrl_x = mid_x + offset * math.cos(math.atan2(my - corner_y, mx - corner_x) + math.pi/3)
                ctrl_y = mid_y + offset * math.sin(math.atan2(my - corner_y, mx - corner_x) + math.pi/3)
                
                path_data = f"M {corner_x} {corner_y} Q {ctrl_x} {ctrl_y} {mx} {my}"
                dwg.add(dwg.path(d=path_data, stroke=colors['accent'], 
                               stroke_width=1.5, fill='none', opacity=0.6))


def generate_authentic_kolam_from_image(img_path: str, spacing: int = 30) -> str:
    """Main function to generate authentic traditional kolam from image"""
    try:
        generator = AuthenticKolamGenerator()
        return generator.generate_authentic_kolam(img_path, spacing)
    except Exception as e:
        print(f"Error generating authentic kolam: {e}")
        # Return default kolam if anything fails
        generator = AuthenticKolamGenerator()
        return generator._generate_default_kolam(spacing)
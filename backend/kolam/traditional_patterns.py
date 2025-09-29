#!/usr/bin/env python3
"""
Traditional Tamil Kolam Pattern Library
Contains authentic kolam patterns with proper dot-grid layouts and curved connections.
"""

import math
import numpy as np
from typing import List, Tuple, Dict


class TraditionalKolamPatterns:
    """Library of traditional Tamil kolam patterns"""
    
    @staticmethod
    def generate_pulli_kolam(dots_h: int = 5, dots_v: int = 5) -> Dict:
        """
        Generate traditional pulli (dot) kolam with connecting curves.
        
        Args:
            dots_h: Number of dots horizontally
            dots_v: Number of dots vertically
            
        Returns:
            Dictionary with dot positions and connecting curves
        """
        dots = []
        curves = []
        
        # Create dot grid
        for i in range(dots_v):
            for j in range(dots_h):
                dots.append({'x': j * 40 + 20, 'y': i * 40 + 20})
        
        # Add traditional connecting curves between dots
        # Horizontal curves
        for i in range(dots_v):
            for j in range(dots_h - 1):
                start_x = j * 40 + 20 + 8  # dot radius offset
                start_y = i * 40 + 20
                end_x = (j + 1) * 40 + 20 - 8
                end_y = i * 40 + 20
                
                # Create curved path between dots
                control_y = start_y + (-1) ** i * 15  # Alternate curve direction
                curve = {
                    'path': f"M {start_x} {start_y} Q {(start_x + end_x) / 2} {control_y} {end_x} {end_y}",
                    'stroke': '#2458ff',
                    'stroke_width': 3
                }
                curves.append(curve)
        
        # Vertical curves  
        for i in range(dots_v - 1):
            for j in range(dots_h):
                start_x = j * 40 + 20
                start_y = i * 40 + 20 + 8
                end_x = j * 40 + 20
                end_y = (i + 1) * 40 + 20 - 8
                
                # Create curved path between dots
                control_x = start_x + (-1) ** j * 15  # Alternate curve direction
                curve = {
                    'path': f"M {start_x} {start_y} Q {control_x} {(start_y + end_y) / 2} {end_x} {end_y}",
                    'stroke': '#2458ff',
                    'stroke_width': 3
                }
                curves.append(curve)
        
        return {
            'dots': dots,
            'curves': curves,
            'width': dots_h * 40,
            'height': dots_v * 40,
            'type': 'pulli_kolam'
        }
    
    @staticmethod
    def generate_sikku_kolam(size: int = 5) -> Dict:
        """
        Generate traditional sikku (interlaced) kolam pattern.
        
        Args:
            size: Size of the pattern grid
            
        Returns:
            Dictionary with pattern data
        """
        dots = []
        curves = []
        width = size * 60
        height = size * 60
        
        center_x = width // 2
        center_y = height // 2
        
        # Create central dot
        dots.append({'x': center_x, 'y': center_y})
        
        # Create interlaced pattern radiating from center
        for ring in range(1, size // 2 + 1):
            radius = ring * 30
            
            # Number of points on this ring
            num_points = ring * 8
            
            ring_dots = []
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                ring_dots.append({'x': x, 'y': y})
                dots.append({'x': x, 'y': y})
            
            # Connect dots in this ring with interlaced curves
            for i in range(num_points):
                next_i = (i + 2) % num_points  # Skip one dot for interlacing
                
                start_x = ring_dots[i]['x']
                start_y = ring_dots[i]['y']
                end_x = ring_dots[next_i]['x']
                end_y = ring_dots[next_i]['y']
                
                # Create curved interlaced connection
                mid_angle = math.atan2((start_y + end_y) / 2 - center_y, 
                                     (start_x + end_x) / 2 - center_x)
                control_radius = radius + (-1) ** i * 10
                control_x = center_x + control_radius * math.cos(mid_angle)
                control_y = center_y + control_radius * math.sin(mid_angle)
                
                curve = {
                    'path': f"M {start_x:.1f} {start_y:.1f} Q {control_x:.1f} {control_y:.1f} {end_x:.1f} {end_y:.1f}",
                    'stroke': '#E24A90',
                    'stroke_width': 2
                }
                curves.append(curve)
        
        return {
            'dots': dots,
            'curves': curves,
            'width': width,
            'height': height,
            'type': 'sikku_kolam'
        }
    
    @staticmethod
    def generate_kambi_kolam(loops: int = 3) -> Dict:
        """
        Generate traditional kambi (rope/loop) kolam pattern.
        
        Args:
            loops: Number of interlocking loops
            
        Returns:
            Dictionary with pattern data
        """
        dots = []
        curves = []
        
        width = loops * 80 + 40
        height = 120
        
        for i in range(loops):
            # Center position for this loop
            center_x = (i + 1) * 80
            center_y = height // 2
            
            # Add central dot
            dots.append({'x': center_x, 'y': center_y})
            
            # Create interlocking loop
            loop_radius = 25
            
            # Main loop curve
            for j in range(4):
                angle_start = j * math.pi / 2
                angle_end = (j + 1) * math.pi / 2
                
                start_x = center_x + loop_radius * math.cos(angle_start)
                start_y = center_y + loop_radius * math.sin(angle_start)
                end_x = center_x + loop_radius * math.cos(angle_end)
                end_y = center_y + loop_radius * math.sin(angle_end)
                
                # Create quarter circle arc
                sweep_flag = 1 if j < 2 else 0
                arc_path = f"M {start_x:.1f} {start_y:.1f} A {loop_radius} {loop_radius} 0 0 {sweep_flag} {end_x:.1f} {end_y:.1f}"
                
                curve = {
                    'path': arc_path,
                    'stroke': '#4A90E2',
                    'stroke_width': 4,
                    'fill': 'none'
                }
                curves.append(curve)
            
            # Connect to next loop if not the last
            if i < loops - 1:
                next_center_x = (i + 2) * 80
                
                # Connecting curve between loops
                connect_path = f"M {center_x + loop_radius} {center_y} Q {center_x + 40} {center_y - 15} {next_center_x - loop_radius} {center_y}"
                
                curve = {
                    'path': connect_path,
                    'stroke': '#90E24A',
                    'stroke_width': 3,
                    'fill': 'none'
                }
                curves.append(curve)
        
        return {
            'dots': dots,
            'curves': curves,
            'width': width,
            'height': height,
            'type': 'kambi_kolam'
        }
    
    @staticmethod
    def generate_neli_kolam(size: int = 4) -> Dict:
        """
        Generate traditional neli (twisted/braided) kolam pattern.
        
        Args:
            size: Complexity size of the pattern
            
        Returns:
            Dictionary with pattern data
        """
        dots = []
        curves = []
        
        width = size * 50 + 50
        height = size * 50 + 50
        
        # Create braided pattern
        center_x = width // 2
        center_y = height // 2
        
        # Central dot
        dots.append({'x': center_x, 'y': center_y})
        
        # Create nested twisted loops
        for layer in range(1, size + 1):
            radius = layer * 20
            
            # Number of twists for this layer
            num_twists = layer * 2
            
            for twist in range(num_twists):
                angle_offset = 2 * math.pi * twist / num_twists
                
                # Create twisted loop
                t_values = np.linspace(0, 2 * math.pi, 20)
                path_points = []
                
                for t in t_values:
                    # Parametric equations for twisted curve
                    r = radius * (1 + 0.3 * math.sin(3 * t + angle_offset))
                    x = center_x + r * math.cos(t + angle_offset)
                    y = center_y + r * math.sin(t + angle_offset)
                    path_points.append((x, y))
                
                # Create smooth path through points
                if path_points:
                    path_data = f"M {path_points[0][0]:.1f} {path_points[0][1]:.1f}"
                    
                    for i in range(1, len(path_points), 2):
                        if i + 1 < len(path_points):
                            # Quadratic bezier curve through multiple points
                            cp_x = path_points[i][0]
                            cp_y = path_points[i][1]
                            end_x = path_points[i + 1][0]
                            end_y = path_points[i + 1][1]
                            
                            path_data += f" Q {cp_x:.1f} {cp_y:.1f} {end_x:.1f} {end_y:.1f}"
                    
                    path_data += " Z"  # Close the loop
                    
                    curve = {
                        'path': path_data,
                        'stroke': f'hsl({layer * 60 + twist * 20}, 70%, 50%)',
                        'stroke_width': 2,
                        'fill': 'none'
                    }
                    curves.append(curve)
        
        return {
            'dots': dots,
            'curves': curves,
            'width': width,
            'height': height,
            'type': 'neli_kolam'
        }
    
    @staticmethod
    def generate_traditional_motif(motif_type: str = 'lotus') -> Dict:
        """
        Generate traditional kolam motifs.
        
        Args:
            motif_type: Type of motif ('lotus', 'peacock', 'lamp', 'flower')
            
        Returns:
            Dictionary with pattern data
        """
        dots = []
        curves = []
        width = 200
        height = 200
        center_x = width // 2
        center_y = height // 2
        
        if motif_type == 'lotus':
            # Central dot
            dots.append({'x': center_x, 'y': center_y})
            
            # Lotus petals
            num_petals = 8
            petal_length = 60
            
            for i in range(num_petals):
                angle = 2 * math.pi * i / num_petals
                
                # Petal shape using bezier curves
                tip_x = center_x + petal_length * math.cos(angle)
                tip_y = center_y + petal_length * math.sin(angle)
                
                # Control points for petal shape
                ctrl1_angle = angle - 0.3
                ctrl2_angle = angle + 0.3
                ctrl_radius = petal_length * 0.7
                
                ctrl1_x = center_x + ctrl_radius * math.cos(ctrl1_angle)
                ctrl1_y = center_y + ctrl_radius * math.sin(ctrl1_angle)
                ctrl2_x = center_x + ctrl_radius * math.cos(ctrl2_angle)
                ctrl2_y = center_y + ctrl_radius * math.sin(ctrl2_angle)
                
                # Create petal curve
                petal_path = f"M {center_x} {center_y} Q {ctrl1_x:.1f} {ctrl1_y:.1f} {tip_x:.1f} {tip_y:.1f} Q {ctrl2_x:.1f} {ctrl2_y:.1f} {center_x} {center_y}"
                
                curve = {
                    'path': petal_path,
                    'stroke': '#E24A90',
                    'stroke_width': 2,
                    'fill': 'rgba(226, 74, 144, 0.1)'
                }
                curves.append(curve)
            
            # Central circle
            center_circle = {
                'path': f"M {center_x + 15} {center_y} A 15 15 0 1 0 {center_x - 15} {center_y} A 15 15 0 1 0 {center_x + 15} {center_y}",
                'stroke': '#4A90E2',
                'stroke_width': 3,
                'fill': 'none'
            }
            curves.append(center_circle)
        
        elif motif_type == 'peacock':
            # Simplified peacock pattern
            # Body
            body_path = f"M {center_x} {center_y + 30} Q {center_x + 20} {center_y} {center_x} {center_y - 30}"
            curves.append({
                'path': body_path,
                'stroke': '#4A90E2',
                'stroke_width': 4,
                'fill': 'none'
            })
            
            # Tail feathers
            for i in range(5):
                angle = -math.pi/3 + i * math.pi/6
                feather_x = center_x - 50 * math.cos(angle)
                feather_y = center_y - 50 * math.sin(angle)
                
                feather_path = f"M {center_x} {center_y} Q {feather_x:.1f} {feather_y:.1f} {feather_x + 20 * math.cos(angle + math.pi/2):.1f} {feather_y + 20 * math.sin(angle + math.pi/2):.1f}"
                
                curves.append({
                    'path': feather_path,
                    'stroke': f'hsl({i * 60}, 70%, 50%)',
                    'stroke_width': 2,
                    'fill': 'none'
                })
        
        return {
            'dots': dots,
            'curves': curves,
            'width': width,
            'height': height,
            'type': f'{motif_type}_motif'
        }


def generate_pattern_svg(pattern_data: Dict, background_color: str = '#0a0a0a') -> str:
    """
    Convert pattern data to SVG string.
    
    Args:
        pattern_data: Pattern data from one of the generator functions
        background_color: Background color for the SVG
        
    Returns:
        SVG string
    """
    import svgwrite
    
    width = pattern_data['width']
    height = pattern_data['height']
    
    dwg = svgwrite.Drawing(size=(f'{width}px', f'{height}px'), 
                          viewBox=f'0 0 {width} {height}')
    
    # Add background
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill=background_color))
    
    # Add curves first (so dots appear on top)
    for curve in pattern_data['curves']:
        path = dwg.path(d=curve['path'])
        path['stroke'] = curve['stroke']
        path['stroke-width'] = curve['stroke_width']
        path['fill'] = curve.get('fill', 'none')
        if 'opacity' in curve:
            path['opacity'] = curve['opacity']
        dwg.add(path)
    
    # Add dots
    for dot in pattern_data['dots']:
        circle = dwg.circle(center=(dot['x'], dot['y']), r=4, 
                          fill='white', opacity=0.9)
        dwg.add(circle)
    
    return dwg.tostring()


if __name__ == "__main__":
    # Test different pattern types
    patterns = [
        ('pulli', TraditionalKolamPatterns.generate_pulli_kolam(5, 4)),
        ('sikku', TraditionalKolamPatterns.generate_sikku_kolam(4)),
        ('kambi', TraditionalKolamPatterns.generate_kambi_kolam(3)),
        ('neli', TraditionalKolamPatterns.generate_neli_kolam(3)),
        ('lotus', TraditionalKolamPatterns.generate_traditional_motif('lotus'))
    ]
    
    for name, pattern in patterns:
        svg_content = generate_pattern_svg(pattern)
        filename = f"traditional_{name}_kolam.svg"
        
        with open(filename, 'w') as f:
            f.write(svg_content)
        
        print(f"✅ Generated: {filename}")
#!/usr/bin/env python3
"""
Traditional Flowing Kolam Generator
Creates authentic Tamil kolam patterns with flowing curves, interlaced designs,
and traditional decorative elements matching classic kolam aesthetics.
"""

import cv2
import numpy as np
import svgwrite
import math
import random
from typing import Tuple, List, Dict, Any

class TraditionalFlowingKolamGenerator:
    def __init__(self):
        self.traditional_colors = {
            'background': '#0a0a0a',     # Dark background
            'primary': '#ffffff',        # White base lines  
            'accent': '#dc143c',         # Red accents
            'secondary': '#ff6347',      # Orange highlights
            'dots': '#ffffff',           # White dots
            'fill': '#dc143c'            # Solid red fill (SVGWrite doesn't support rgba in some contexts)
        }
        
        self.pattern_templates = [
            self._generate_lotus_mandala,
            self._generate_interlaced_diamond,
            self._generate_petal_flow,
            self._generate_celtic_knot_style,
            self._generate_concentric_loops,
            self._generate_floral_vine,
            self._generate_geometric_flower,
            self._generate_radial_burst
        ]

    def generate_traditional_kolam(self, img_path: str = None, pattern_size: int = 400) -> str:
        """Generate a traditional flowing kolam pattern"""
        # Create SVG drawing
        dwg = svgwrite.Drawing(size=(f'{pattern_size}px', f'{pattern_size}px'), 
                              viewBox=f'0 0 {pattern_size} {pattern_size}')
        
        # Dark background
        dwg.add(dwg.rect(insert=(0, 0), size=(pattern_size, pattern_size), 
                        fill=self.traditional_colors['background']))
        
        # Choose random pattern template
        pattern_generator = random.choice(self.pattern_templates)
        pattern_generator(dwg, pattern_size)
        
        return dwg.tostring()
    
    def _generate_lotus_mandala(self, dwg: svgwrite.Drawing, size: int):
        """Generate lotus mandala-style kolam"""
        center_x = center_y = size // 2
        
        # Central dot
        dwg.add(dwg.circle(center=(center_x, center_y), r=4,
                          fill=self.traditional_colors['dots']))
        
        # Inner petals
        petal_count = 8
        inner_radius = size * 0.15
        
        for i in range(petal_count):
            angle = (2 * math.pi * i) / petal_count
            
            # Petal shape using cubic curves
            x1 = center_x + inner_radius * math.cos(angle - math.pi/8)
            y1 = center_y + inner_radius * math.sin(angle - math.pi/8)
            x2 = center_x + inner_radius * 1.8 * math.cos(angle)
            y2 = center_y + inner_radius * 1.8 * math.sin(angle)
            x3 = center_x + inner_radius * math.cos(angle + math.pi/8)
            y3 = center_y + inner_radius * math.sin(angle + math.pi/8)
            
            # Create petal path
            petal_path = dwg.path(
                d=f"M {center_x} {center_y} "
                  f"Q {x1} {y1} {x2} {y2} "
                  f"Q {x3} {y3} {center_x} {center_y}",
                stroke=self.traditional_colors['primary'],
                stroke_width=2,
                fill=self.traditional_colors['fill']
            )
            dwg.add(petal_path)
        
        # Outer ring patterns
        outer_radius = size * 0.35
        for i in range(12):
            angle = (2 * math.pi * i) / 12
            x = center_x + outer_radius * math.cos(angle)
            y = center_y + outer_radius * math.sin(angle)
            
            # Decorative loops
            loop_size = 15
            loop_path = dwg.path(
                d=f"M {x-loop_size} {y} "
                  f"Q {x} {y-loop_size} {x+loop_size} {y} "
                  f"Q {x} {y+loop_size} {x-loop_size} {y} Z",
                stroke=self.traditional_colors['primary'],
                stroke_width=2,
                fill='none'
            )
            dwg.add(loop_path)
            
            # Connecting dots
            dwg.add(dwg.circle(center=(x, y), r=3,
                              fill=self.traditional_colors['dots']))
    
    def _generate_interlaced_diamond(self, dwg: svgwrite.Drawing, size: int):
        """Generate interlaced diamond pattern"""
        center_x = center_y = size // 2
        
        # Create diamond grid
        grid_spacing = size // 8
        
        for row in range(-3, 4):
            for col in range(-3, 4):
                if (row + col) % 2 == 0:  # Checkerboard pattern
                    x = center_x + col * grid_spacing
                    y = center_y + row * grid_spacing
                    
                    # Diamond shape with flowing curves
                    diamond_size = grid_spacing * 0.4
                    
                    # Curved diamond using bezier curves
                    diamond_path = dwg.path(
                        d=f"M {x} {y-diamond_size} "
                          f"C {x+diamond_size*0.7} {y-diamond_size*0.3} "
                          f"{x+diamond_size*0.7} {y+diamond_size*0.3} "
                          f"{x} {y+diamond_size} "
                          f"C {x-diamond_size*0.7} {y+diamond_size*0.3} "
                          f"{x-diamond_size*0.7} {y-diamond_size*0.3} "
                          f"{x} {y-diamond_size} Z",
                        stroke=self.traditional_colors['primary'],
                        stroke_width=2.5,
                        fill='none'
                    )
                    dwg.add(diamond_path)
                    
                    # Central dot
                    dwg.add(dwg.circle(center=(x, y), r=2,
                                      fill=self.traditional_colors['dots']))
        
        # Connecting interlaced lines
        for i in range(8):
            angle = (2 * math.pi * i) / 8
            start_radius = size * 0.1
            end_radius = size * 0.4
            
            x1 = center_x + start_radius * math.cos(angle)
            y1 = center_y + start_radius * math.sin(angle)
            x2 = center_x + end_radius * math.cos(angle)
            y2 = center_y + end_radius * math.sin(angle)
            
            # Curved connection
            mid_x = center_x + (start_radius + end_radius) / 2 * math.cos(angle + math.pi/16)
            mid_y = center_y + (start_radius + end_radius) / 2 * math.sin(angle + math.pi/16)
            
            connection_path = dwg.path(
                d=f"M {x1} {y1} Q {mid_x} {mid_y} {x2} {y2}",
                stroke=self.traditional_colors['accent'],
                stroke_width=2,
                fill='none'
            )
            dwg.add(connection_path)
    
    def _generate_petal_flow(self, dwg: svgwrite.Drawing, size: int):
        """Generate flowing petal design"""
        center_x = center_y = size // 2
        
        # Multiple concentric petal rings
        rings = [0.15, 0.25, 0.35]
        petal_counts = [6, 10, 14]
        
        for ring_idx, (radius_factor, petal_count) in enumerate(zip(rings, petal_counts)):
            radius = size * radius_factor
            
            for i in range(petal_count):
                angle = (2 * math.pi * i) / petal_count
                
                # Petal position
                px = center_x + radius * math.cos(angle)
                py = center_y + radius * math.sin(angle)
                
                # Petal size varies by ring
                petal_size = 15 + ring_idx * 5
                
                # Create flowing petal shape
                petal_angle = angle + math.pi/2
                
                # Control points for petal curve
                cp1_x = px + petal_size * 0.8 * math.cos(petal_angle - math.pi/3)
                cp1_y = py + petal_size * 0.8 * math.sin(petal_angle - math.pi/3)
                cp2_x = px + petal_size * math.cos(petal_angle)
                cp2_y = py + petal_size * math.sin(petal_angle)
                cp3_x = px + petal_size * 0.8 * math.cos(petal_angle + math.pi/3)
                cp3_y = py + petal_size * 0.8 * math.sin(petal_angle + math.pi/3)
                
                petal_path = dwg.path(
                    d=f"M {px} {py} "
                      f"C {cp1_x} {cp1_y} {cp2_x} {cp2_y} {cp3_x} {cp3_y} "
                      f"Q {px + petal_size*0.3*math.cos(angle)} {py + petal_size*0.3*math.sin(angle)} {px} {py}",
                    stroke=self.traditional_colors['primary'],
                    stroke_width=2,
                    fill=self.traditional_colors['fill'] if ring_idx % 2 == 0 else 'none'
                )
                dwg.add(petal_path)
                
                # Dot at petal base
                dwg.add(dwg.circle(center=(px, py), r=2,
                                  fill=self.traditional_colors['dots']))
    
    def _generate_celtic_knot_style(self, dwg: svgwrite.Drawing, size: int):
        """Generate Celtic knot-style interlaced pattern"""
        center_x = center_y = size // 2
        
        # Grid for knot intersections
        grid_size = 5
        cell_size = size // (grid_size + 1)
        
        # Create interlaced paths
        paths = []
        
        # Horizontal paths
        for row in range(1, grid_size):
            y = center_y - (grid_size // 2 - row) * cell_size
            
            path_points = []
            for col in range(grid_size + 1):
                x = center_x - (grid_size // 2 - col) * cell_size
                
                # Add curve variation
                curve_offset = 15 * math.sin(col * math.pi / 3)
                path_points.append((x, y + curve_offset))
            
            # Create smooth curved path
            if len(path_points) >= 2:
                path_d = f"M {path_points[0][0]} {path_points[0][1]}"
                
                for i in range(1, len(path_points)):
                    if i < len(path_points) - 1:
                        # Smooth curve through points
                        cx = (path_points[i][0] + path_points[i+1][0]) / 2
                        cy = (path_points[i][1] + path_points[i+1][1]) / 2
                        path_d += f" Q {path_points[i][0]} {path_points[i][1]} {cx} {cy}"
                    else:
                        path_d += f" Q {path_points[i][0]} {path_points[i][1]} {path_points[i][0]} {path_points[i][1]}"
                
                dwg.add(dwg.path(d=path_d,
                               stroke=self.traditional_colors['primary'],
                               stroke_width=3,
                               fill='none'))
        
        # Vertical paths
        for col in range(1, grid_size):
            x = center_x - (grid_size // 2 - col) * cell_size
            
            path_points = []
            for row in range(grid_size + 1):
                y = center_y - (grid_size // 2 - row) * cell_size
                
                # Add curve variation
                curve_offset = 15 * math.cos(row * math.pi / 3)
                path_points.append((x + curve_offset, y))
            
            # Create smooth curved path
            if len(path_points) >= 2:
                path_d = f"M {path_points[0][0]} {path_points[0][1]}"
                
                for i in range(1, len(path_points)):
                    if i < len(path_points) - 1:
                        # Smooth curve through points
                        cx = (path_points[i][0] + path_points[i+1][0]) / 2
                        cy = (path_points[i][1] + path_points[i+1][1]) / 2
                        path_d += f" Q {path_points[i][0]} {path_points[i][1]} {cx} {cy}"
                    else:
                        path_d += f" Q {path_points[i][0]} {path_points[i][1]} {path_points[i][0]} {path_points[i][1]}"
                
                dwg.add(dwg.path(d=path_d,
                               stroke=self.traditional_colors['accent'],
                               stroke_width=3,
                               fill='none'))
        
        # Add intersection dots
        for row in range(1, grid_size):
            for col in range(1, grid_size):
                x = center_x - (grid_size // 2 - col) * cell_size
                y = center_y - (grid_size // 2 - row) * cell_size
                
                dwg.add(dwg.circle(center=(x, y), r=4,
                                  fill=self.traditional_colors['dots']))
    
    def _generate_concentric_loops(self, dwg: svgwrite.Drawing, size: int):
        """Generate concentric loop pattern"""
        center_x = center_y = size // 2
        
        # Multiple concentric rings of loops
        ring_count = 4
        
        for ring in range(1, ring_count + 1):
            radius = size * 0.08 * ring
            loop_count = 6 + ring * 2
            
            for i in range(loop_count):
                angle = (2 * math.pi * i) / loop_count
                
                # Loop center
                lx = center_x + radius * math.cos(angle)
                ly = center_y + radius * math.sin(angle)
                
                # Loop size
                loop_radius = 12 - ring * 2
                
                # Create loop with flourish
                loop_path = dwg.path(
                    d=f"M {lx - loop_radius} {ly} "
                      f"Q {lx} {ly - loop_radius * 1.5} {lx + loop_radius} {ly} "
                      f"Q {lx} {ly + loop_radius * 1.5} {lx - loop_radius} {ly} "
                      f"M {lx} {ly - loop_radius * 0.7} "
                      f"Q {lx + loop_radius * 0.5} {ly - loop_radius * 0.2} {lx} {ly + loop_radius * 0.7}",
                    stroke=self.traditional_colors['primary'],
                    stroke_width=2.5,
                    fill='none'
                )
                dwg.add(loop_path)
                
                # Center dot
                dwg.add(dwg.circle(center=(lx, ly), r=2,
                                  fill=self.traditional_colors['dots']))
                
                # Connect to next ring
                if ring < ring_count:
                    next_radius = size * 0.08 * (ring + 1)
                    next_angle = angle + (math.pi / (loop_count * 2))  # Offset for interlacing
                    next_x = center_x + next_radius * math.cos(next_angle)
                    next_y = center_y + next_radius * math.sin(next_angle)
                    
                    # Curved connection
                    mid_x = (lx + next_x) / 2 + 10 * math.cos(angle + math.pi/2)
                    mid_y = (ly + next_y) / 2 + 10 * math.sin(angle + math.pi/2)
                    
                    connect_path = dwg.path(
                        d=f"M {lx} {ly} Q {mid_x} {mid_y} {next_x} {next_y}",
                        stroke=self.traditional_colors['accent'],
                        stroke_width=1.5,
                        fill='none',
                        opacity=0.7
                    )
                    dwg.add(connect_path)
    
    def _generate_floral_vine(self, dwg: svgwrite.Drawing, size: int):
        """Generate flowing floral vine pattern"""
        center_x = center_y = size // 2
        
        # Main vine paths
        vine_count = 6
        
        for vine_idx in range(vine_count):
            angle_offset = (2 * math.pi * vine_idx) / vine_count
            
            # Vine path points
            path_points = []
            flower_positions = []
            
            for t in range(0, 100, 8):
                # Spiral vine path
                radius = (size * 0.4) * (t / 100)
                angle = angle_offset + (t / 100) * 4 * math.pi
                
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                
                path_points.append((x, y))
                
                # Add flowers at intervals
                if t % 24 == 0 and t > 0:
                    flower_positions.append((x, y, angle))
            
            # Draw vine
            if len(path_points) >= 2:
                vine_d = f"M {path_points[0][0]} {path_points[0][1]}"
                
                for i in range(1, len(path_points), 2):
                    if i + 1 < len(path_points):
                        vine_d += f" Q {path_points[i][0]} {path_points[i][1]} {path_points[i+1][0]} {path_points[i+1][1]}"
                
                dwg.add(dwg.path(d=vine_d,
                               stroke=self.traditional_colors['primary'],
                               stroke_width=2,
                               fill='none'))
            
            # Add flowers
            for fx, fy, fangle in flower_positions:
                # Simple flower with petals
                petal_count = 5
                petal_size = 8
                
                for p in range(petal_count):
                    petal_angle = fangle + (2 * math.pi * p) / petal_count
                    
                    px1 = fx + petal_size * math.cos(petal_angle - math.pi/5)
                    py1 = fy + petal_size * math.sin(petal_angle - math.pi/5)
                    px2 = fx + petal_size * 1.5 * math.cos(petal_angle)
                    py2 = fy + petal_size * 1.5 * math.sin(petal_angle)
                    px3 = fx + petal_size * math.cos(petal_angle + math.pi/5)
                    py3 = fy + petal_size * math.sin(petal_angle + math.pi/5)
                    
                    petal = dwg.path(
                        d=f"M {fx} {fy} Q {px1} {py1} {px2} {py2} Q {px3} {py3} {fx} {fy}",
                        stroke=self.traditional_colors['accent'],
                        stroke_width=1.5,
                        fill=self.traditional_colors['fill']
                    )
                    dwg.add(petal)
                
                # Flower center
                dwg.add(dwg.circle(center=(fx, fy), r=3,
                                  fill=self.traditional_colors['dots']))
    
    def _generate_geometric_flower(self, dwg: svgwrite.Drawing, size: int):
        """Generate geometric flower pattern"""
        center_x = center_y = size // 2
        
        # Central geometric flower
        layers = 3
        base_petals = 8
        
        for layer in range(layers):
            petal_count = base_petals + layer * 4
            radius = size * (0.1 + layer * 0.08)
            
            for i in range(petal_count):
                angle = (2 * math.pi * i) / petal_count
                
                # Petal points
                px = center_x + radius * math.cos(angle)
                py = center_y + radius * math.sin(angle)
                
                # Geometric petal shape
                petal_size = 15 - layer * 3
                
                # Diamond-shaped petal
                p1_x = px + petal_size * math.cos(angle - math.pi/6)
                p1_y = py + petal_size * math.sin(angle - math.pi/6)
                p2_x = px + petal_size * 1.5 * math.cos(angle)
                p2_y = py + petal_size * 1.5 * math.sin(angle)
                p3_x = px + petal_size * math.cos(angle + math.pi/6)
                p3_y = py + petal_size * math.sin(angle + math.pi/6)
                
                petal_path = dwg.path(
                    d=f"M {px} {py} L {p1_x} {p1_y} L {p2_x} {p2_y} L {p3_x} {p3_y} Z",
                    stroke=self.traditional_colors['primary'],
                    stroke_width=2,
                    fill=self.traditional_colors['fill'] if layer % 2 == 0 else 'none'
                )
                dwg.add(petal_path)
                
                # Connection lines to center
                if layer == 0:
                    connect_line = dwg.line(
                        start=(center_x, center_y),
                        end=(px, py),
                        stroke=self.traditional_colors['accent'],
                        stroke_width=1,
                        opacity=0.5
                    )
                    dwg.add(connect_line)
        
        # Central dot
        dwg.add(dwg.circle(center=(center_x, center_y), r=5,
                          fill=self.traditional_colors['dots']))
        
        # Corner decorative elements
        corner_size = size * 0.15
        corners = [
            (corner_size, corner_size),
            (size - corner_size, corner_size),
            (size - corner_size, size - corner_size),
            (corner_size, size - corner_size)
        ]
        
        for cx, cy in corners:
            # Small decorative swirl
            swirl_path = dwg.path(
                d=f"M {cx-20} {cy} "
                  f"Q {cx-10} {cy-10} {cx} {cy} "
                  f"Q {cx+10} {cy+10} {cx+20} {cy} "
                  f"Q {cx+10} {cy-10} {cx} {cy-20}",
                stroke=self.traditional_colors['accent'],
                stroke_width=2,
                fill='none'
            )
            dwg.add(swirl_path)
    
    def _generate_radial_burst(self, dwg: svgwrite.Drawing, size: int):
        """Generate radial burst pattern"""
        center_x = center_y = size // 2
        
        # Central burst with rays
        ray_count = 16
        inner_radius = size * 0.05
        outer_radius = size * 0.4
        
        for i in range(ray_count):
            angle = (2 * math.pi * i) / ray_count
            
            # Ray endpoints
            inner_x = center_x + inner_radius * math.cos(angle)
            inner_y = center_y + inner_radius * math.sin(angle)
            outer_x = center_x + outer_radius * math.cos(angle)
            outer_y = center_y + outer_radius * math.sin(angle)
            
            # Curved ray
            mid_radius = (inner_radius + outer_radius) / 2
            curve_angle = angle + math.pi/32  # Slight curve
            mid_x = center_x + mid_radius * math.cos(curve_angle)
            mid_y = center_y + mid_radius * math.sin(curve_angle)
            
            ray_path = dwg.path(
                d=f"M {inner_x} {inner_y} Q {mid_x} {mid_y} {outer_x} {outer_y}",
                stroke=self.traditional_colors['primary'],
                stroke_width=2,
                fill='none'
            )
            dwg.add(ray_path)
            
            # Decorative elements at ray ends
            if i % 2 == 0:
                # Alternating decorative ends
                end_size = 8
                end_circle = dwg.circle(
                    center=(outer_x, outer_y),
                    r=end_size,
                    stroke=self.traditional_colors['accent'],
                    stroke_width=2,
                    fill=self.traditional_colors['fill']
                )
                dwg.add(end_circle)
            else:
                # Diamond end
                end_size = 6
                diamond_path = dwg.path(
                    d=f"M {outer_x} {outer_y-end_size} "
                      f"L {outer_x+end_size} {outer_y} "
                      f"L {outer_x} {outer_y+end_size} "
                      f"L {outer_x-end_size} {outer_y} Z",
                    stroke=self.traditional_colors['accent'],
                    stroke_width=2,
                    fill=self.traditional_colors['fill']
                )
                dwg.add(diamond_path)
        
        # Central decorative element
        center_rings = 3
        for ring in range(center_rings):
            ring_radius = inner_radius * (ring + 1) / center_rings
            
            dwg.add(dwg.circle(
                center=(center_x, center_y),
                r=ring_radius,
                stroke=self.traditional_colors['dots'],
                stroke_width=2,
                fill='none'
            ))


def generate_traditional_flowing_kolam_from_image(img_path: str, pattern_size: int = 400) -> str:
    """Generate a traditional flowing kolam pattern"""
    generator = TraditionalFlowingKolamGenerator()
    return generator.generate_traditional_kolam(img_path, pattern_size)


# Test the generator
if __name__ == "__main__":
    generator = TraditionalFlowingKolamGenerator()
    svg_output = generator.generate_traditional_kolam(pattern_size=400)
    
    # Save test output
    with open("traditional_flowing_kolam_test.svg", "w") as f:
        f.write(svg_output)
    
    print("Traditional flowing kolam generated successfully!")
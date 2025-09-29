#!/usr/bin/env python3
"""
Authentic Traditional Tamil Kolam Generator
Creates proper Tamil kolam patterns with traditional dot grids and connecting curves.
Focuses on producing visually accurate and culturally authentic kolam designs.
"""

import cv2
import numpy as np
import svgwrite
import math
import random
from typing import Tuple, List, Dict, Any

class AuthenticTraditionalKolamGenerator:
    def __init__(self):
        self.grid_patterns = {
            'pulli_kolam': self._generate_pulli_kolam,
            'sikku_kolam': self._generate_sikku_kolam,  
            'kambi_kolam': self._generate_kambi_kolam,
            'neli_kolam': self._generate_neli_kolam,
            'mandala_kolam': self._generate_mandala_kolam
        }
        
        # Traditional Tamil kolam colors
        self.traditional_colors = {
            'white': '#ffffff',      # Rice flour - most traditional
            'yellow': '#ffd700',     # Turmeric powder
            'red': '#dc143c',        # Kumkum/Sindoor
            'orange': '#ff6347',     # Marigold petals
            'blue': '#4169e1',       # Blue chalk
            'green': '#228b22',      # Green rangoli powder
            'purple': '#8a2be2'      # Purple rangoli powder
        }

    def analyze_input_for_pattern_selection(self, img_path: str) -> Dict[str, Any]:
        """Analyze input image to select appropriate kolam pattern"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Resize for analysis
        height, width = img.shape[:2]
        max_dimension = 400
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Analyze image characteristics
        analysis = {
            'has_circular_elements': self._detect_circles(gray),
            'has_geometric_patterns': self._detect_geometric_patterns(gray),
            'has_organic_shapes': self._detect_organic_shapes(gray),
            'color_complexity': self._analyze_color_complexity(img),
            'symmetry_level': self._detect_symmetry(gray)
        }
        
        # Select pattern based on analysis
        if analysis['has_circular_elements'] and analysis['symmetry_level'] > 0.6:
            pattern_type = 'mandala_kolam'
        elif analysis['has_geometric_patterns'] and analysis['symmetry_level'] > 0.4:
            pattern_type = 'sikku_kolam'
        elif analysis['has_organic_shapes']:
            pattern_type = 'kambi_kolam'  
        elif analysis['color_complexity'] > 0.5:
            pattern_type = 'neli_kolam'
        else:
            pattern_type = 'pulli_kolam'  # Default traditional pattern
            
        analysis['selected_pattern'] = pattern_type
        return analysis

    def _detect_circles(self, gray: np.ndarray) -> bool:
        """Detect circular elements in image"""
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=100)
        return circles is not None

    def _detect_geometric_patterns(self, gray: np.ndarray) -> bool:
        """Detect geometric patterns like lines and rectangles"""
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        return lines is not None and len(lines) > 10

    def _detect_organic_shapes(self, gray: np.ndarray) -> bool:
        """Detect organic, flowing shapes"""
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        organic_count = 0
        for contour in contours:
            if len(contour) > 20:
                # Calculate contour smoothness
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Organic shapes have many points and high curvature
                if len(approx) > 8:
                    organic_count += 1
        
        return organic_count > 3

    def _analyze_color_complexity(self, img: np.ndarray) -> float:
        """Analyze color complexity of image"""
        # Convert to HSV and analyze color distribution
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        
        # Calculate histogram
        hist = cv2.calcHist([h_channel], [0], None, [180], [0, 180])
        
        # Normalize and calculate entropy (color complexity)
        hist = hist.flatten()
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]  # Remove zeros
        
        entropy = -np.sum(hist * np.log2(hist))
        return entropy / 8.0  # Normalize to 0-1 range

    def _detect_symmetry(self, gray: np.ndarray) -> float:
        """Detect symmetry level in image"""
        h, w = gray.shape
        
        # Check horizontal symmetry
        left_half = gray[:, :w//2]
        right_half = cv2.flip(gray[:, w//2:], 1)
        
        # Resize to match if needed
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # Calculate correlation
        correlation = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)
        h_symmetry = correlation[0, 0] if correlation.size > 0 else 0
        
        # Check vertical symmetry
        top_half = gray[:h//2, :]
        bottom_half = cv2.flip(gray[h//2:, :], 0)
        
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_height, :]
        bottom_half = bottom_half[:min_height, :]
        
        correlation = cv2.matchTemplate(top_half, bottom_half, cv2.TM_CCOEFF_NORMED)
        v_symmetry = correlation[0, 0] if correlation.size > 0 else 0
        
        return max(abs(h_symmetry), abs(v_symmetry))

    def generate_authentic_kolam(self, img_path: str, spacing: int = 30) -> str:
        """Generate authentic Tamil kolam pattern"""
        analysis = self.analyze_input_for_pattern_selection(img_path)
        pattern_type = analysis['selected_pattern']
        
        print(f"Generating {pattern_type} based on image analysis")
        
        # Generate the selected pattern
        generator_func = self.grid_patterns[pattern_type]
        return generator_func(spacing, analysis)

    def _generate_pulli_kolam(self, spacing: int, analysis: Dict) -> str:
        """Generate traditional Pulli (dot-based) Kolam - most common type"""
        canvas_size = 500
        dwg = svgwrite.Drawing(size=(f'{canvas_size}px', f'{canvas_size}px'), 
                              viewBox=f'0 0 {canvas_size} {canvas_size}')
        dwg.add(dwg.rect(insert=(0, 0), size=(canvas_size, canvas_size), fill='#0a0a0a'))
        
        # Create dot grid (pulli pattern)
        grid_size = 9  # Traditional 9x9 or similar
        center_x = center_y = canvas_size // 2
        start_x = center_x - (grid_size * spacing) // 2
        start_y = center_y - (grid_size * spacing) // 2
        
        # Store dot positions for connection
        dots = []
        
        # Create dots (pulli)
        for row in range(grid_size):
            dot_row = []
            for col in range(grid_size):
                x = start_x + col * spacing
                y = start_y + row * spacing
                
                # Add dot
                dwg.add(dwg.circle(
                    center=(x, y), r=4,
                    fill=self.traditional_colors['white'],
                    opacity=1.0
                ))
                dot_row.append((x, y))
            dots.append(dot_row)
        
        # Connect dots with traditional curves
        self._add_pulli_connections(dwg, dots, spacing)
        
        return dwg.tostring()

    def _add_pulli_connections(self, dwg: svgwrite.Drawing, dots: List[List[Tuple]], spacing: int):
        """Add traditional pulli kolam connections between dots"""
        rows, cols = len(dots), len(dots[0])
        
        # Traditional pulli patterns - connect around dots to form loops
        for row in range(rows - 1):
            for col in range(cols - 1):
                # Get four corner dots
                top_left = dots[row][col]
                top_right = dots[row][col + 1]
                bottom_left = dots[row + 1][col]
                bottom_right = dots[row + 1][col + 1]
                
                # Create flowing curves around dots
                self._create_pulli_loop(dwg, top_left, top_right, bottom_right, bottom_left, spacing)
        
        # Add decorative outer connections
        self._add_outer_decorative_curves(dwg, dots, spacing)

    def _create_pulli_loop(self, dwg: svgwrite.Drawing, tl: Tuple, tr: Tuple, br: Tuple, bl: Tuple, spacing: int):
        """Create traditional loop around four dots"""
        tl_x, tl_y = tl
        tr_x, tr_y = tr
        br_x, br_y = br
        bl_x, bl_y = bl
        
        # Calculate curve control points (traditional kolam curves flow around dots)
        offset = spacing * 0.3
        
        # Top curve
        path_data = f"M {tl_x + offset:.1f} {tl_y:.1f} "
        path_data += f"Q {(tl_x + tr_x) / 2:.1f} {tl_y - offset:.1f} {tr_x - offset:.1f} {tr_y:.1f} "
        
        # Right curve
        path_data += f"Q {tr_x:.1f} {(tr_y + br_y) / 2:.1f} {br_x:.1f} {br_y - offset:.1f} "
        
        # Bottom curve
        path_data += f"Q {(br_x + bl_x) / 2:.1f} {br_y + offset:.1f} {bl_x + offset:.1f} {bl_y:.1f} "
        
        # Left curve (close the loop)
        path_data += f"Q {bl_x:.1f} {(bl_y + tl_y) / 2:.1f} {tl_x + offset:.1f} {tl_y:.1f}"
        
        dwg.add(dwg.path(d=path_data, stroke=self.traditional_colors['white'], 
                       stroke_width=2.5, fill='none', opacity=0.9))

    def _add_outer_decorative_curves(self, dwg: svgwrite.Drawing, dots: List[List[Tuple]], spacing: int):
        """Add decorative curves around the outer border"""
        rows, cols = len(dots), len(dots[0])
        
        # Add corner decorations
        corners = [
            dots[0][0],           # Top-left
            dots[0][cols-1],      # Top-right  
            dots[rows-1][cols-1], # Bottom-right
            dots[rows-1][0]       # Bottom-left
        ]
        
        for corner in corners:
            self._add_corner_decoration(dwg, corner, spacing)

    def _add_corner_decoration(self, dwg: svgwrite.Drawing, center: Tuple, spacing: int):
        """Add traditional corner decorations"""
        cx, cy = center
        radius = spacing * 0.8
        
        # Create petal-like decorations at corners
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            angle_rad = math.radians(angle)
            end_x = cx + radius * math.cos(angle_rad)
            end_y = cy + radius * math.sin(angle_rad)
            
            # Create curved petal
            mid_angle = math.radians(angle + 22.5)
            mid_x = cx + radius * 1.3 * math.cos(mid_angle)
            mid_y = cy + radius * 1.3 * math.sin(mid_angle)
            
            path_data = f"M {cx:.1f} {cy:.1f} Q {mid_x:.1f} {mid_y:.1f} {end_x:.1f} {end_y:.1f}"
            dwg.add(dwg.path(d=path_data, stroke=self.traditional_colors['yellow'], 
                           stroke_width=1.5, fill='none', opacity=0.7))

    def _generate_sikku_kolam(self, spacing: int, analysis: Dict) -> str:
        """Generate Sikku (line-based) Kolam with interwoven patterns"""
        canvas_size = 500
        dwg = svgwrite.Drawing(size=(f'{canvas_size}px', f'{canvas_size}px'), 
                              viewBox=f'0 0 {canvas_size} {canvas_size}')
        dwg.add(dwg.rect(insert=(0, 0), size=(canvas_size, canvas_size), fill='#0a0a0a'))
        
        center = canvas_size // 2
        
        # Create interwoven line patterns characteristic of sikku kolam
        self._create_interwoven_lines(dwg, center, spacing)
        self._create_geometric_border(dwg, center, canvas_size, spacing)
        
        return dwg.tostring()

    def _create_interwoven_lines(self, dwg: svgwrite.Drawing, center: int, spacing: int):
        """Create interwoven line patterns"""
        # Create crossing diagonal lines
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = center + i * spacing
                y = center + j * spacing
                
                # Create X-pattern at each intersection
                self._create_x_pattern(dwg, x, y, spacing * 0.4)

    def _create_x_pattern(self, dwg: svgwrite.Drawing, cx: int, cy: int, size: float):
        """Create X-pattern at intersection"""
        # Diagonal lines forming X
        path1 = f"M {cx - size:.1f} {cy - size:.1f} L {cx + size:.1f} {cy + size:.1f}"
        path2 = f"M {cx - size:.1f} {cy + size:.1f} L {cx + size:.1f} {cy - size:.1f}"
        
        dwg.add(dwg.path(d=path1, stroke=self.traditional_colors['white'], 
                       stroke_width=2, opacity=0.8))
        dwg.add(dwg.path(d=path2, stroke=self.traditional_colors['white'], 
                       stroke_width=2, opacity=0.8))

    def _create_geometric_border(self, dwg: svgwrite.Drawing, center: int, canvas_size: int, spacing: int):
        """Create geometric border pattern"""
        border_offset = 50
        
        # Create diamond border
        points = [
            (center, border_offset),                           # Top
            (canvas_size - border_offset, center),             # Right
            (center, canvas_size - border_offset),             # Bottom  
            (border_offset, center)                            # Left
        ]
        
        # Connect points with curved lines
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]
            
            # Create curved connection
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            
            # Add curvature toward center
            curve_factor = 0.3
            ctrl_x = mid_x + (center - mid_x) * curve_factor
            ctrl_y = mid_y + (center - mid_y) * curve_factor
            
            path_data = f"M {start[0]:.1f} {start[1]:.1f} Q {ctrl_x:.1f} {ctrl_y:.1f} {end[0]:.1f} {end[1]:.1f}"
            dwg.add(dwg.path(d=path_data, stroke=self.traditional_colors['orange'], 
                           stroke_width=3, fill='none', opacity=0.9))

    def _generate_kambi_kolam(self, spacing: int, analysis: Dict) -> str:
        """Generate Kambi (loop-based) Kolam with flowing curves"""
        canvas_size = 500
        dwg = svgwrite.Drawing(size=(f'{canvas_size}px', f'{canvas_size}px'), 
                              viewBox=f'0 0 {canvas_size} {canvas_size}')
        dwg.add(dwg.rect(insert=(0, 0), size=(canvas_size, canvas_size), fill='#0a0a0a'))
        
        center = canvas_size // 2
        
        # Create flowing loop patterns
        self._create_flowing_loops(dwg, center, spacing)
        self._create_spiral_elements(dwg, center, spacing)
        
        return dwg.tostring()

    def _create_flowing_loops(self, dwg: svgwrite.Drawing, center: int, spacing: int):
        """Create flowing loop patterns characteristic of kambi kolam"""
        # Create multiple concentric loops with variations
        for ring in range(1, 5):
            radius = ring * spacing * 1.5
            
            # Create wavy circular path
            path_data = "M "
            n_points = 16
            
            for i in range(n_points + 1):
                angle = (2 * math.pi * i) / n_points
                wave_offset = math.sin(angle * 3) * spacing * 0.3  # Create waves
                
                x = center + (radius + wave_offset) * math.cos(angle)
                y = center + (radius + wave_offset) * math.sin(angle)
                
                if i == 0:
                    path_data += f"{x:.1f} {y:.1f} "
                else:
                    path_data += f"L {x:.1f} {y:.1f} "
            
            path_data += "Z"  # Close the loop
            
            color = self.traditional_colors['red'] if ring % 2 == 0 else self.traditional_colors['blue']
            dwg.add(dwg.path(d=path_data, stroke=color, stroke_width=2.5, 
                           fill='none', opacity=0.8))

    def _create_spiral_elements(self, dwg: svgwrite.Drawing, center: int, spacing: int):
        """Add spiral decorative elements"""
        # Create four spirals in quadrants
        offsets = [(-spacing*2, -spacing*2), (spacing*2, -spacing*2), 
                  (spacing*2, spacing*2), (-spacing*2, spacing*2)]
        
        for offset_x, offset_y in offsets:
            spiral_center_x = center + offset_x
            spiral_center_y = center + offset_y
            
            self._create_spiral(dwg, spiral_center_x, spiral_center_y, spacing * 0.8)

    def _create_spiral(self, dwg: svgwrite.Drawing, cx: int, cy: int, max_radius: float):
        """Create spiral pattern"""
        path_data = f"M {cx:.1f} {cy:.1f} "
        
        n_turns = 3
        n_points = 50
        
        for i in range(n_points):
            t = i / n_points
            angle = t * n_turns * 2 * math.pi
            radius = t * max_radius
            
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            
            if i == 0:
                path_data = f"M {x:.1f} {y:.1f} "
            else:
                path_data += f"L {x:.1f} {y:.1f} "
        
        dwg.add(dwg.path(d=path_data, stroke=self.traditional_colors['green'], 
                       stroke_width=2, fill='none', opacity=0.7))

    def _generate_neli_kolam(self, spacing: int, analysis: Dict) -> str:
        """Generate Neli (square-based) Kolam with geometric patterns"""
        canvas_size = 500
        dwg = svgwrite.Drawing(size=(f'{canvas_size}px', f'{canvas_size}px'), 
                              viewBox=f'0 0 {canvas_size} {canvas_size}')
        dwg.add(dwg.rect(insert=(0, 0), size=(canvas_size, canvas_size), fill='#0a0a0a'))
        
        center = canvas_size // 2
        
        # Create nested square patterns
        self._create_nested_squares(dwg, center, spacing)
        self._create_corner_motifs(dwg, center, canvas_size, spacing)
        
        return dwg.tostring()

    def _create_nested_squares(self, dwg: svgwrite.Drawing, center: int, spacing: int):
        """Create nested square patterns with connecting elements"""
        for level in range(1, 6):
            size = level * spacing * 1.2
            
            # Square corners
            half_size = size / 2
            corners = [
                (center - half_size, center - half_size),  # Top-left
                (center + half_size, center - half_size),  # Top-right
                (center + half_size, center + half_size),  # Bottom-right
                (center - half_size, center + half_size)   # Bottom-left
            ]
            
            # Create rounded square
            path_data = f"M {corners[0][0] + 10:.1f} {corners[0][1]:.1f} "
            
            for i in range(len(corners)):
                current = corners[i]
                next_corner = corners[(i + 1) % len(corners)]
                
                # Create rounded corners
                corner_radius = 15
                path_data += f"L {next_corner[0] - corner_radius:.1f} {next_corner[1]:.1f} "
                path_data += f"Q {next_corner[0]:.1f} {next_corner[1]:.1f} {next_corner[0]:.1f} {next_corner[1] + corner_radius:.1f} "
            
            path_data += "Z"
            
            color = self.traditional_colors['purple'] if level % 2 == 0 else self.traditional_colors['yellow']
            dwg.add(dwg.path(d=path_data, stroke=color, stroke_width=2, 
                           fill='none', opacity=0.8))

    def _create_corner_motifs(self, dwg: svgwrite.Drawing, center: int, canvas_size: int, spacing: int):
        """Create decorative motifs at corners"""
        corner_positions = [
            (spacing, spacing),
            (canvas_size - spacing, spacing),
            (canvas_size - spacing, canvas_size - spacing),
            (spacing, canvas_size - spacing)
        ]
        
        for cx, cy in corner_positions:
            # Create flower-like motif
            for petal in range(6):
                angle = (2 * math.pi * petal) / 6
                petal_length = spacing * 0.6
                
                end_x = cx + petal_length * math.cos(angle)
                end_y = cy + petal_length * math.sin(angle)
                
                # Create petal curve
                mid_angle = angle + math.pi / 12
                mid_x = cx + petal_length * 0.7 * math.cos(mid_angle)
                mid_y = cy + petal_length * 0.7 * math.sin(mid_angle)
                
                path_data = f"M {cx:.1f} {cy:.1f} Q {mid_x:.1f} {mid_y:.1f} {end_x:.1f} {end_y:.1f}"
                dwg.add(dwg.path(d=path_data, stroke=self.traditional_colors['red'], 
                               stroke_width=2, fill='none', opacity=0.7))

    def _generate_mandala_kolam(self, spacing: int, analysis: Dict) -> str:
        """Generate Mandala-style Kolam with radial symmetry"""
        canvas_size = 500
        dwg = svgwrite.Drawing(size=(f'{canvas_size}px', f'{canvas_size}px'), 
                              viewBox=f'0 0 {canvas_size} {canvas_size}')
        dwg.add(dwg.rect(insert=(0, 0), size=(canvas_size, canvas_size), fill='#0a0a0a'))
        
        center = canvas_size // 2
        
        # Create mandala with radial patterns
        self._create_central_mandala(dwg, center, spacing)
        self._create_radial_petals(dwg, center, spacing)
        self._create_outer_mandala_ring(dwg, center, spacing)
        
        return dwg.tostring()

    def _create_central_mandala(self, dwg: svgwrite.Drawing, center: int, spacing: int):
        """Create central mandala pattern"""
        # Central circle
        dwg.add(dwg.circle(center=(center, center), r=spacing//2,
                         stroke=self.traditional_colors['white'], stroke_width=3,
                         fill='none', opacity=1.0))
        
        # Inner ring of dots
        for i in range(8):
            angle = (2 * math.pi * i) / 8
            x = center + (spacing * 0.8) * math.cos(angle)
            y = center + (spacing * 0.8) * math.sin(angle)
            
            dwg.add(dwg.circle(center=(x, y), r=3,
                             fill=self.traditional_colors['yellow'], opacity=1.0))

    def _create_radial_petals(self, dwg: svgwrite.Drawing, center: int, spacing: int):
        """Create radial petal patterns"""
        n_petals = 12
        petal_radius = spacing * 1.5
        
        for i in range(n_petals):
            angle = (2 * math.pi * i) / n_petals
            
            # Petal tip
            tip_x = center + petal_radius * math.cos(angle)
            tip_y = center + petal_radius * math.sin(angle)
            
            # Petal sides
            side_angle1 = angle - math.pi / 24
            side_angle2 = angle + math.pi / 24
            
            side1_x = center + petal_radius * 0.3 * math.cos(side_angle1)
            side1_y = center + petal_radius * 0.3 * math.sin(side_angle1)
            
            side2_x = center + petal_radius * 0.3 * math.cos(side_angle2)
            side2_y = center + petal_radius * 0.3 * math.sin(side_angle2)
            
            # Create petal path
            path_data = f"M {side1_x:.1f} {side1_y:.1f} Q {tip_x:.1f} {tip_y:.1f} {side2_x:.1f} {side2_y:.1f}"
            
            color = self.traditional_colors['blue'] if i % 2 == 0 else self.traditional_colors['red']
            dwg.add(dwg.path(d=path_data, stroke=color, stroke_width=2.5, 
                           fill='none', opacity=0.8))

    def _create_outer_mandala_ring(self, dwg: svgwrite.Drawing, center: int, spacing: int):
        """Create outer decorative ring"""
        outer_radius = spacing * 3
        
        # Create decorative outer circle with wave pattern
        path_data = "M "
        n_points = 24
        
        for i in range(n_points + 1):
            angle = (2 * math.pi * i) / n_points
            wave_amplitude = spacing * 0.2 * math.sin(angle * 6)  # 6 waves around circle
            
            radius = outer_radius + wave_amplitude
            x = center + radius * math.cos(angle)
            y = center + radius * math.sin(angle)
            
            if i == 0:
                path_data += f"{x:.1f} {y:.1f} "
            else:
                path_data += f"L {x:.1f} {y:.1f} "
        
        path_data += "Z"
        
        dwg.add(dwg.path(d=path_data, stroke=self.traditional_colors['orange'], 
                       stroke_width=3, fill='none', opacity=0.9))


def generate_authentic_traditional_kolam_from_image(img_path: str, spacing: int = 30) -> str:
    """Main function to generate authentic traditional kolam from image"""
    try:
        generator = AuthenticTraditionalKolamGenerator()
        return generator.generate_authentic_kolam(img_path, spacing)
    except Exception as e:
        print(f"Error generating authentic traditional kolam: {e}")
        raise
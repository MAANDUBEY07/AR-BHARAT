#!/usr/bin/env python3
"""
Improved Authentic Tamil Kolam Pattern Generator
Creates varied, authentic kolam patterns that actually analyze input images
and generate different outputs based on image content.
"""

import cv2
import numpy as np
import svgwrite
import math
from typing import Tuple, List, Dict, Any


class ImprovedKolamGenerator:
    def __init__(self):
        self.dot_patterns = self._initialize_dot_patterns()
        
    def _initialize_dot_patterns(self):
        """Initialize traditional kolam dot connection patterns"""
        return {
            # Simple patterns - single loops and curves
            'simple_loop': self._create_simple_loop,
            'horizontal_arc': self._create_horizontal_arc,
            'vertical_arc': self._create_vertical_arc,
            'diagonal_curve': self._create_diagonal_curve,
            
            # Medium patterns - interconnected curves  
            'figure_eight': self._create_figure_eight,
            'double_loop': self._create_double_loop,
            'crossing_arcs': self._create_crossing_arcs,
            'spiral_loop': self._create_spiral_loop,
            
            # Complex patterns - traditional kolam motifs
            'petal_pattern': self._create_petal_pattern,
            'mandala_basic': self._create_mandala_basic,
            'wave_connection': self._create_wave_connection,
            'diamond_loop': self._create_diamond_loop
        }

    def analyze_image_regions(self, img_path: str) -> Dict[str, Any]:
        """Analyze image to extract structure for varied kolam generation"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
            
        # Resize for analysis
        height, width = img.shape[:2]
        max_dimension = 400
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
            height, width = new_height, new_width

        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Multiple edge detection approaches
        edges_canny = cv2.Canny(gray, 50, 150)
        edges_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        edges_sobel = np.uint8(np.absolute(edges_sobel))
        
        # Analyze texture and patterns
        texture_analysis = self._analyze_texture(gray)
        
        # Color analysis
        dominant_colors = self._analyze_colors(img)
        
        # Structural analysis
        contours, _ = cv2.findContours(edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return {
            'image_shape': (height, width),
            'gray': gray,
            'edges_canny': edges_canny,
            'edges_sobel': edges_sobel,
            'texture_info': texture_analysis,
            'dominant_colors': dominant_colors,
            'contours': contours,
            'complexity_score': self._calculate_complexity(edges_canny, contours)
        }

    def _analyze_texture(self, gray_img: np.ndarray) -> Dict[str, float]:
        """Analyze texture properties of the image"""
        # Calculate texture measures
        laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()
        
        # Local binary pattern approximation
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        texture_resp = cv2.filter2D(gray_img, cv2.CV_32F, kernel)
        texture_variance = np.var(texture_resp)
        
        # Gradient analysis
        grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return {
            'laplacian_variance': float(laplacian_var),
            'texture_variance': float(texture_variance),
            'gradient_mean': float(np.mean(gradient_magnitude)),
            'gradient_std': float(np.std(gradient_magnitude))
        }

    def _analyze_colors(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Calculate color histograms
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Find dominant hue
        dominant_hue = int(np.argmax(hist_h))
        
        # Color diversity score
        color_diversity = len(np.unique(img.reshape(-1, img.shape[-1]), axis=0))
        
        return {
            'dominant_hue': dominant_hue,
            'color_diversity': color_diversity,
            'saturation_mean': float(np.mean(hsv[:, :, 1])),
            'value_mean': float(np.mean(hsv[:, :, 2]))
        }

    def _calculate_complexity(self, edges: np.ndarray, contours: List) -> float:
        """Calculate overall image complexity score"""
        edge_density = np.sum(edges > 0) / edges.size
        num_contours = len(contours)
        
        # Contour complexity
        contour_complexity = 0
        if contours:
            total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
            contour_complexity = total_perimeter / (edges.shape[0] * edges.shape[1])
        
        return edge_density * 0.4 + (num_contours / 100) * 0.3 + contour_complexity * 0.3

    def generate_adaptive_kolam(self, img_path: str, grid_spacing: int = 20) -> str:
        """Generate kolam that adapts to image content"""
        
        # Analyze the image
        analysis = self.analyze_image_regions(img_path)
        height, width = analysis['image_shape']
        
        # Create grid based on image structure
        grid_points = self._create_adaptive_grid(analysis, grid_spacing)
        
        # Generate SVG
        dwg = svgwrite.Drawing(size=(f'{width}px', f'{height}px'), 
                              viewBox=f'0 0 {width} {height}')
        
        # Add dark background for contrast
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='#0a0a0a'))
        
        # Colors based on image analysis
        colors = self._select_kolam_colors(analysis['dominant_colors'])
        
        # Generate patterns for each grid point
        for grid_point in grid_points:
            self._add_adaptive_pattern(dwg, grid_point, analysis, colors, grid_spacing)
        
        return dwg.tostring()

    def _create_adaptive_grid(self, analysis: Dict, spacing: int) -> List[Dict]:
        """Create grid points that adapt to image structure"""
        height, width = analysis['image_shape']
        edges = analysis['edges_canny']
        
        grid_points = []
        
        # Create base grid
        y_positions = np.arange(spacing, height - spacing, spacing)
        x_positions = np.arange(spacing, width - spacing, spacing)
        
        for y in y_positions:
            for x in x_positions:
                # Analyze local region
                local_region = self._get_local_region(edges, int(x), int(y), spacing // 2)
                
                if local_region is not None:
                    edge_density = np.mean(local_region) / 255.0
                    
                    # Only place dots where there's some structure
                    if edge_density > 0.05:
                        
                        # Calculate pattern complexity based on local features
                        pattern_complexity = min(1.0, edge_density * 2)
                        
                        # Analyze local texture
                        gray_region = self._get_local_region(
                            analysis['gray'], int(x), int(y), spacing // 2
                        )
                        
                        texture_score = 0
                        if gray_region is not None:
                            texture_score = np.std(gray_region) / 255.0
                        
                        grid_point = {
                            'x': float(x),
                            'y': float(y),
                            'edge_density': edge_density,
                            'pattern_complexity': pattern_complexity,
                            'texture_score': texture_score,
                            'local_features': self._extract_local_features(local_region)
                        }
                        
                        grid_points.append(grid_point)
        
        return grid_points

    def _get_local_region(self, img: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
        """Extract local region around a point"""
        h, w = img.shape[:2]
        x1 = max(0, x - radius)
        x2 = min(w, x + radius)
        y1 = max(0, y - radius)  
        y2 = min(h, y + radius)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        return img[y1:y2, x1:x2]

    def _extract_local_features(self, region: np.ndarray) -> Dict:
        """Extract features from local region"""
        if region is None or region.size == 0:
            return {'has_edges': False, 'direction': 'none', 'density': 0.0}
        
        # Find dominant edge direction
        grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
        
        mag_x = np.mean(np.abs(grad_x))
        mag_y = np.mean(np.abs(grad_y))
        
        if mag_x > mag_y * 1.5:
            direction = 'vertical'
        elif mag_y > mag_x * 1.5:
            direction = 'horizontal'
        else:
            direction = 'mixed'
        
        return {
            'has_edges': np.max(region) > 100,
            'direction': direction,
            'density': float(np.mean(region) / 255.0)
        }

    def _select_kolam_colors(self, color_info: Dict) -> Dict[str, str]:
        """Select appropriate colors based on image analysis"""
        # Traditional kolam colors with variations
        base_colors = {
            'primary': '#4A90E2',    # Blue
            'secondary': '#E24A90',  # Pink
            'accent': '#90E24A',     # Green
            'dot': 'white',
            'connection': '#7B68EE'   # Medium slate blue
        }
        
        # Adjust based on dominant hue
        hue = color_info.get('dominant_hue', 120)
        saturation = color_info.get('saturation_mean', 128) / 255.0
        
        if hue < 30 or hue > 150:  # Red-ish tones
            base_colors['primary'] = '#E25D4A'  # Red-orange
        elif hue < 60:  # Orange-yellow tones  
            base_colors['primary'] = '#E2A04A'  # Orange
        elif hue < 90:  # Yellow-green tones
            base_colors['primary'] = '#A0E24A'  # Yellow-green
        
        # Adjust saturation
        if saturation < 0.3:  # Low saturation - use more muted colors
            base_colors['primary'] = '#6B8CAE'
            base_colors['secondary'] = '#AE6B8C'
        
        return base_colors

    def _add_adaptive_pattern(self, dwg: svgwrite.Drawing, grid_point: Dict, 
                             analysis: Dict, colors: Dict, spacing: int):
        """Add pattern that adapts to local image features"""
        x, y = grid_point['x'], grid_point['y']
        complexity = grid_point['pattern_complexity']
        features = grid_point['local_features']
        
        # Add dot first
        dwg.add(dwg.circle(
            center=(x, y), r=3, 
            fill=colors['dot'], 
            opacity=0.9
        ))
        
        # Select pattern based on complexity and features
        if complexity > 0.7:
            pattern_type = 'complex'
        elif complexity > 0.4:
            pattern_type = 'medium'
        else:
            pattern_type = 'simple'
        
        # Generate connecting curves based on pattern type
        if pattern_type == 'complex':
            self._add_complex_pattern(dwg, x, y, spacing, colors, features)
        elif pattern_type == 'medium':
            self._add_medium_pattern(dwg, x, y, spacing, colors, features)
        else:
            self._add_simple_pattern(dwg, x, y, spacing, colors, features)

    def _add_simple_pattern(self, dwg: svgwrite.Drawing, x: float, y: float, 
                           spacing: int, colors: Dict, features: Dict):
        """Add simple kolam patterns"""
        radius = spacing * 0.3
        
        # Simple circular arc
        path_data = f"M {x-radius} {y} Q {x} {y-radius} {x+radius} {y}"
        dwg.add(dwg.path(d=path_data, stroke=colors['primary'], 
                        stroke_width=2, fill='none', opacity=0.8))
        
        # Connecting arc below
        path_data2 = f"M {x-radius} {y} Q {x} {y+radius} {x+radius} {y}"
        dwg.add(dwg.path(d=path_data2, stroke=colors['primary'], 
                        stroke_width=2, fill='none', opacity=0.8))

    def _add_medium_pattern(self, dwg: svgwrite.Drawing, x: float, y: float, 
                           spacing: int, colors: Dict, features: Dict):
        """Add medium complexity kolam patterns"""
        radius = spacing * 0.35
        
        # Figure-8 style pattern
        # Upper curve
        path_data = f"M {x-radius} {y-radius*0.5} Q {x} {y-radius} {x+radius} {y-radius*0.5} Q {x} {y} {x-radius} {y-radius*0.5}"
        dwg.add(dwg.path(d=path_data, stroke=colors['secondary'], 
                        stroke_width=2.5, fill='none', opacity=0.8))
        
        # Lower curve  
        path_data2 = f"M {x-radius} {y+radius*0.5} Q {x} {y+radius} {x+radius} {y+radius*0.5} Q {x} {y} {x-radius} {y+radius*0.5}"
        dwg.add(dwg.path(d=path_data2, stroke=colors['secondary'], 
                        stroke_width=2.5, fill='none', opacity=0.8))
        
        # Central connecting loop
        small_r = radius * 0.4
        path_data3 = f"M {x-small_r} {y} Q {x} {y-small_r} {x+small_r} {y} Q {x} {y+small_r} {x-small_r} {y}"
        dwg.add(dwg.path(d=path_data3, stroke=colors['accent'], 
                        stroke_width=1.5, fill='none', opacity=0.7))

    def _add_complex_pattern(self, dwg: svgwrite.Drawing, x: float, y: float, 
                           spacing: int, colors: Dict, features: Dict):
        """Add complex traditional kolam patterns"""
        radius = spacing * 0.4
        
        # Multiple interconnected loops
        # Outer pattern
        for i in range(4):
            angle = i * math.pi / 2
            offset_x = radius * 0.7 * math.cos(angle)
            offset_y = radius * 0.7 * math.sin(angle)
            
            center_x = x + offset_x
            center_y = y + offset_y
            
            # Petal-like curves
            path_data = f"""M {center_x-radius*0.3} {center_y} 
                           Q {center_x} {center_y-radius*0.5} {center_x+radius*0.3} {center_y}
                           Q {center_x} {center_y+radius*0.5} {center_x-radius*0.3} {center_y}"""
            
            dwg.add(dwg.path(d=path_data, stroke=colors['connection'], 
                            stroke_width=2, fill='none', opacity=0.7))
        
        # Central mandala pattern
        points = 8
        for i in range(points):
            angle1 = 2 * math.pi * i / points
            angle2 = 2 * math.pi * (i + 1) / points
            
            x1 = x + radius * 0.4 * math.cos(angle1)
            y1 = y + radius * 0.4 * math.sin(angle1)
            x2 = x + radius * 0.4 * math.cos(angle2)
            y2 = y + radius * 0.4 * math.sin(angle2)
            
            # Curved connection between points
            mid_x = x + radius * 0.2 * math.cos((angle1 + angle2) / 2)
            mid_y = y + radius * 0.2 * math.sin((angle1 + angle2) / 2)
            
            path_data = f"M {x1} {y1} Q {mid_x} {mid_y} {x2} {y2}"
            dwg.add(dwg.path(d=path_data, stroke=colors['accent'], 
                            stroke_width=1.5, fill='none', opacity=0.8))

    # Placeholder methods for dot patterns (can be expanded)
    def _create_simple_loop(self, x, y, size): return [], []
    def _create_horizontal_arc(self, x, y, size): return [], []
    def _create_vertical_arc(self, x, y, size): return [], []
    def _create_diagonal_curve(self, x, y, size): return [], []
    def _create_figure_eight(self, x, y, size): return [], []
    def _create_double_loop(self, x, y, size): return [], []
    def _create_crossing_arcs(self, x, y, size): return [], []
    def _create_spiral_loop(self, x, y, size): return [], []
    def _create_petal_pattern(self, x, y, size): return [], []
    def _create_mandala_basic(self, x, y, size): return [], []
    def _create_wave_connection(self, x, y, size): return [], []
    def _create_diamond_loop(self, x, y, size): return [], []


def generate_improved_kolam_from_image(img_path: str, spacing: int = 20) -> str:
    """Main function to generate improved kolam from image"""
    generator = ImprovedKolamGenerator()
    return generator.generate_adaptive_kolam(img_path, spacing)


if __name__ == "__main__":
    # Test the improved generator
    import sys
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        svg_content = generate_improved_kolam_from_image(img_path)
        
        output_file = "improved_kolam_output.svg"
        with open(output_file, 'w') as f:
            f.write(svg_content)
        
        print(f"✅ Improved kolam generated: {output_file}")
    else:
        print("Usage: python improved_authentic_kolam.py <image_path>")
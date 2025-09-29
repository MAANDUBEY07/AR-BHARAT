"""
Authentic Dot Grid Kolam Generator
Implements traditional Tamil kolam with:
1. Pulli (dot grid) foundation
2. Continuous curved lines
3. Loop and spiral patterns
4. Authentic kolam vocabulary
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw
import math
import random
from typing import List, Tuple, Dict, Any
import xml.etree.ElementTree as ET
from xml.dom import minidom

class AuthenticDotGridKolamGenerator:
    """Generate authentic kolam patterns with proper dot grid foundation and curved lines"""
    
    def __init__(self):
        self.dot_spacing = 40  # Standard spacing between dots
        self.line_width = 3
        self.canvas_size = (800, 800)
        self.dot_radius = 2
        self.curve_smoothness = 16  # Points per curve segment
        
    def analyze_input_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze input image to determine kolam pattern type and characteristics"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Detect symmetry type
        symmetry = self.detect_symmetry(gray)
        
        # Determine grid size based on image complexity
        complexity = self.calculate_complexity(gray)
        if complexity > 0.7:
            grid_size = (15, 15)  # Complex pattern
        elif complexity > 0.4:
            grid_size = (11, 11)  # Medium pattern  
        else:
            grid_size = (7, 7)    # Simple pattern
            
        # Detect central motifs
        has_central_motif = self.detect_central_motif(gray)
        
        # Determine pattern type based on analysis
        if has_central_motif and symmetry >= 4:
            pattern_type = "mandala_kolam"
        elif symmetry >= 2:
            pattern_type = "symmetric_kolam"
        else:
            pattern_type = "basic_kolam"
            
        return {
            'pattern_type': pattern_type,
            'grid_size': grid_size,
            'symmetry': symmetry,
            'complexity': complexity,
            'has_central_motif': has_central_motif,
            'image_size': (width, height)
        }
    
    def detect_symmetry(self, gray_image: np.ndarray) -> int:
        """Detect rotational symmetry order"""
        h, w = gray_image.shape
        center = (w // 2, h // 2)
        
        # Test for 2-fold, 4-fold, 6-fold, 8-fold symmetry
        symmetries = []
        for fold in [2, 4, 6, 8]:
            angle = 360 / fold
            similarity = 0
            
            for i in range(fold):
                rotated = self.rotate_image(gray_image, i * angle, center)
                similarity += cv2.matchTemplate(gray_image, rotated, cv2.TM_CCOEFF_NORMED)[0][0]
            
            symmetries.append((fold, similarity / fold))
        
        # Return the symmetry order with highest similarity
        best_symmetry = max(symmetries, key=lambda x: x[1])
        return best_symmetry[0] if best_symmetry[1] > 0.3 else 1
    
    def rotate_image(self, image: np.ndarray, angle: float, center: Tuple[int, int]) -> np.ndarray:
        """Rotate image around center point"""
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    def calculate_complexity(self, gray_image: np.ndarray) -> float:
        """Calculate pattern complexity based on edge density"""
        edges = cv2.Canny(gray_image, 50, 150)
        return np.sum(edges > 0) / edges.size
    
    def detect_central_motif(self, gray_image: np.ndarray) -> bool:
        """Detect if there's a central motif in the image"""
        h, w = gray_image.shape
        center_region = gray_image[h//4:3*h//4, w//4:3*w//4]
        edge_regions = [
            gray_image[:h//4, :],          # Top
            gray_image[3*h//4:, :],        # Bottom  
            gray_image[:, :w//4],          # Left
            gray_image[:, 3*w//4:]         # Right
        ]
        
        center_activity = np.std(center_region)
        edge_activity = np.mean([np.std(region) for region in edge_regions])
        
        return center_activity > edge_activity * 1.2
    
    def generate_dot_grid(self, grid_size: Tuple[int, int], canvas_size: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Generate traditional pulli (dot) grid"""
        rows, cols = grid_size
        width, height = canvas_size
        
        # Calculate spacing to center the grid
        x_spacing = width / (cols + 1)
        y_spacing = height / (rows + 1)
        
        dots = []
        for i in range(rows):
            for j in range(cols):
                x = int((j + 1) * x_spacing)
                y = int((i + 1) * y_spacing)
                dots.append((x, y))
        
        return dots
    
    def generate_kolam_curves(self, dots: List[Tuple[int, int]], pattern_type: str, symmetry: int) -> List[Dict[str, Any]]:
        """Generate authentic kolam curves that flow around dots"""
        curves = []
        
        if pattern_type == "mandala_kolam":
            curves.extend(self.generate_mandala_pattern(dots, symmetry))
        elif pattern_type == "symmetric_kolam":
            curves.extend(self.generate_symmetric_pattern(dots, symmetry))
        else:
            curves.extend(self.generate_basic_pattern(dots))
            
        return curves
    
    def generate_mandala_pattern(self, dots: List[Tuple[int, int]], symmetry: int) -> List[Dict[str, Any]]:
        """Generate mandala-style kolam with central focus"""
        curves = []
        
        # Find center dot
        if not dots:
            return curves
            
        center_x = sum(x for x, y in dots) / len(dots)
        center_y = sum(y for x, y in dots) / len(dots)
        center = (center_x, center_y)
        
        # Sort dots by distance from center
        dots_with_distance = [(dot, math.sqrt((dot[0] - center_x)**2 + (dot[1] - center_y)**2)) for dot in dots]
        dots_by_distance = sorted(dots_with_distance, key=lambda x: x[1])
        
        # Create concentric loops
        rings = self.group_dots_into_rings(dots_by_distance, 3)
        
        for ring_idx, ring_dots in enumerate(rings):
            if len(ring_dots) < 3:
                continue
                
            # Create flowing curves around ring
            ring_curves = self.create_ring_curves(ring_dots, center, symmetry)
            curves.extend(ring_curves)
            
            # Add decorative loops at key points
            if ring_idx > 0:  # Skip innermost ring
                decorative_loops = self.create_decorative_loops(ring_dots, center)
                curves.extend(decorative_loops)
        
        return curves
    
    def generate_symmetric_pattern(self, dots: List[Tuple[int, int]], symmetry: int) -> List[Dict[str, Any]]:
        """Generate symmetric kolam pattern"""
        curves = []
        
        if not dots:
            return curves
            
        # Create base pattern in one sector
        sector_angle = 360 / symmetry
        base_curves = self.create_base_sector_pattern(dots, sector_angle)
        
        # Replicate across all sectors
        center_x = sum(x for x, y in dots) / len(dots)
        center_y = sum(y for x, y in dots) / len(dots)
        center = (center_x, center_y)
        
        for i in range(symmetry):
            angle = i * sector_angle
            rotated_curves = self.rotate_curves(base_curves, angle, center)
            curves.extend(rotated_curves)
            
        return curves
    
    def generate_basic_pattern(self, dots: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Generate basic kolam pattern with simple curves"""
        curves = []
        
        if len(dots) < 4:
            return curves
            
        # Create simple flowing pattern
        connected_dots = self.create_dot_connections(dots)
        
        for connection in connected_dots:
            curve = self.create_smooth_curve_between_dots(connection)
            curves.append(curve)
            
        return curves
    
    def group_dots_into_rings(self, dots_with_distance: List[Tuple[Tuple[int, int], float]], num_rings: int) -> List[List[Tuple[int, int]]]:
        """Group dots into concentric rings"""
        if not dots_with_distance:
            return []
            
        total_dots = len(dots_with_distance)
        dots_per_ring = max(1, total_dots // num_rings)
        
        rings = []
        for i in range(0, total_dots, dots_per_ring):
            ring_dots = [dot for dot, dist in dots_with_distance[i:i+dots_per_ring]]
            if ring_dots:
                rings.append(ring_dots)
        
        return rings
    
    def create_ring_curves(self, ring_dots: List[Tuple[int, int]], center: Tuple[float, float], symmetry: int) -> List[Dict[str, Any]]:
        """Create flowing curves around a ring of dots"""
        curves = []
        
        if len(ring_dots) < 3:
            return curves
        
        # Sort dots by angle from center
        center_x, center_y = center
        dots_with_angle = []
        for x, y in ring_dots:
            angle = math.atan2(y - center_y, x - center_x)
            dots_with_angle.append(((x, y), angle))
        
        dots_with_angle.sort(key=lambda x: x[1])
        sorted_dots = [dot for dot, angle in dots_with_angle]
        
        # Create smooth curve connecting dots in ring
        curve_points = []
        for i in range(len(sorted_dots)):
            current_dot = sorted_dots[i]
            next_dot = sorted_dots[(i + 1) % len(sorted_dots)]
            
            # Create curved path around dots (not through them)
            curve_segment = self.create_curved_path_around_dots(current_dot, next_dot, center)
            curve_points.extend(curve_segment)
        
        if curve_points:
            curves.append({
                'type': 'closed_curve',
                'points': curve_points,
                'style': 'flowing'
            })
        
        return curves
    
    def create_decorative_loops(self, ring_dots: List[Tuple[int, int]], center: Tuple[float, float]) -> List[Dict[str, Any]]:
        """Create decorative loops at key points"""
        curves = []
        center_x, center_y = center
        
        # Create loops at every few dots
        for i in range(0, len(ring_dots), max(1, len(ring_dots) // 4)):
            dot_x, dot_y = ring_dots[i]
            
            # Create petal-like loop extending outward from center
            angle_to_center = math.atan2(dot_y - center_y, dot_x - center_x)
            
            loop_points = self.create_petal_loop(dot_x, dot_y, angle_to_center)
            curves.append({
                'type': 'decorative_loop',
                'points': loop_points,
                'style': 'petal'
            })
        
        return curves
    
    def create_base_sector_pattern(self, dots: List[Tuple[int, int]], sector_angle: float) -> List[Dict[str, Any]]:
        """Create pattern for one sector that will be replicated"""
        curves = []
        
        # Filter dots to one sector for base pattern
        center_x = sum(x for x, y in dots) / len(dots)
        center_y = sum(y for x, y in dots) / len(dots)
        
        sector_dots = []
        for x, y in dots:
            angle = math.degrees(math.atan2(y - center_y, x - center_x))
            if angle < 0:
                angle += 360
            if angle <= sector_angle:
                sector_dots.append((x, y))
        
        if len(sector_dots) >= 2:
            # Create simple connecting curves in sector
            connections = self.create_dot_connections(sector_dots)
            for connection in connections:
                curve = self.create_smooth_curve_between_dots(connection)
                curves.append(curve)
        
        return curves
    
    def create_dot_connections(self, dots: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """Create logical connections between dots for curve paths"""
        connections = []
        
        if len(dots) < 2:
            return connections
        
        # Create connections based on proximity and grid logic
        for i, dot1 in enumerate(dots):
            for j, dot2 in enumerate(dots[i+1:], i+1):
                distance = math.sqrt((dot1[0] - dot2[0])**2 + (dot1[1] - dot2[1])**2)
                if distance < self.dot_spacing * 2.5:  # Connect nearby dots
                    connections.append([dot1, dot2])
        
        return connections
    
    def create_smooth_curve_between_dots(self, dot_connection: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Create smooth curve between connected dots"""
        if len(dot_connection) != 2:
            return {'type': 'curve', 'points': [], 'style': 'smooth'}
        
        start_dot, end_dot = dot_connection
        
        # Generate smooth curve points
        curve_points = self.generate_bezier_curve(start_dot, end_dot)
        
        return {
            'type': 'curve',
            'points': curve_points,
            'style': 'smooth'
        }
    
    def create_curved_path_around_dots(self, dot1: Tuple[int, int], dot2: Tuple[int, int], center: Tuple[float, float]) -> List[Tuple[int, int]]:
        """Create curved path that flows around dots rather than through them"""
        x1, y1 = dot1
        x2, y2 = dot2
        cx, cy = center
        
        # Calculate control points for curve that avoids dots
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Offset curve away from center
        offset_distance = self.dot_spacing * 0.7
        angle_to_center = math.atan2(mid_y - cy, mid_x - cx)
        
        control_x = mid_x + offset_distance * math.cos(angle_to_center)
        control_y = mid_y + offset_distance * math.sin(angle_to_center)
        
        # Generate smooth curve
        return self.generate_quadratic_bezier_curve((x1, y1), (control_x, control_y), (x2, y2))
    
    def create_petal_loop(self, center_x: int, center_y: int, angle: float) -> List[Tuple[int, int]]:
        """Create petal-like decorative loop"""
        points = []
        loop_radius = self.dot_spacing * 0.6
        
        # Create petal shape with multiple curve points
        for i in range(self.curve_smoothness):
            t = i / (self.curve_smoothness - 1)
            petal_angle = angle + (t - 0.5) * math.pi / 3  # ±30 degrees
            
            # Varying radius for petal shape
            radius = loop_radius * (0.3 + 0.7 * math.sin(t * math.pi))
            
            x = center_x + radius * math.cos(petal_angle)
            y = center_y + radius * math.sin(petal_angle)
            points.append((int(x), int(y)))
        
        return points
    
    def generate_bezier_curve(self, start: Tuple[int, int], end: Tuple[int, int], num_points: int = None) -> List[Tuple[int, int]]:
        """Generate smooth Bezier curve between two points"""
        if num_points is None:
            num_points = self.curve_smoothness
            
        x1, y1 = start
        x2, y2 = end
        
        # Calculate control points for natural curve
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        control_offset = distance * 0.3
        
        angle = math.atan2(y2 - y1, x2 - x1)
        perpendicular_angle = angle + math.pi / 2
        
        # Control points offset perpendicular to line
        cx1 = x1 + control_offset * math.cos(perpendicular_angle)
        cy1 = y1 + control_offset * math.sin(perpendicular_angle)
        cx2 = x2 + control_offset * math.cos(perpendicular_angle)
        cy2 = y2 + control_offset * math.sin(perpendicular_angle)
        
        # Generate curve points
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # Cubic Bezier curve
            x = (1-t)**3 * x1 + 3*(1-t)**2*t * cx1 + 3*(1-t)*t**2 * cx2 + t**3 * x2
            y = (1-t)**3 * y1 + 3*(1-t)**2*t * cy1 + 3*(1-t)*t**2 * cy2 + t**3 * y2
            
            points.append((int(x), int(y)))
        
        return points
    
    def generate_quadratic_bezier_curve(self, start: Tuple[int, int], control: Tuple[float, float], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Generate quadratic Bezier curve"""
        points = []
        x1, y1 = start
        cx, cy = control
        x2, y2 = end
        
        for i in range(self.curve_smoothness):
            t = i / (self.curve_smoothness - 1)
            
            # Quadratic Bezier formula
            x = (1-t)**2 * x1 + 2*(1-t)*t * cx + t**2 * x2
            y = (1-t)**2 * y1 + 2*(1-t)*t * cy + t**2 * y2
            
            points.append((int(x), int(y)))
        
        return points
    
    def rotate_curves(self, curves: List[Dict[str, Any]], angle_degrees: float, center: Tuple[float, float]) -> List[Dict[str, Any]]:
        """Rotate curve patterns around center point"""
        rotated_curves = []
        angle_rad = math.radians(angle_degrees)
        cx, cy = center
        
        for curve in curves:
            rotated_curve = curve.copy()
            rotated_points = []
            
            for x, y in curve['points']:
                # Translate to origin
                tx = x - cx
                ty = y - cy
                
                # Rotate
                rx = tx * math.cos(angle_rad) - ty * math.sin(angle_rad)
                ry = tx * math.sin(angle_rad) + ty * math.cos(angle_rad)
                
                # Translate back
                final_x = rx + cx
                final_y = ry + cy
                
                rotated_points.append((int(final_x), int(final_y)))
            
            rotated_curve['points'] = rotated_points
            rotated_curves.append(rotated_curve)
        
        return rotated_curves
    
    def generate_kolam_svg(self, image_path: str) -> str:
        """Generate complete authentic kolam SVG from input image"""
        # Analyze input image
        analysis = self.analyze_input_image(image_path)
        
        # Generate dot grid
        dots = self.generate_dot_grid(analysis['grid_size'], self.canvas_size)
        
        # Generate authentic kolam curves
        curves = self.generate_kolam_curves(dots, analysis['pattern_type'], analysis['symmetry'])
        
        # Create SVG
        svg_content = self.create_svg_from_dots_and_curves(dots, curves, analysis)
        
        return svg_content
    
    def create_svg_from_dots_and_curves(self, dots: List[Tuple[int, int]], curves: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
        """Create SVG representation of authentic kolam"""
        width, height = self.canvas_size
        
        # Create SVG root
        svg = ET.Element('svg')
        svg.set('width', str(width))
        svg.set('height', str(height))
        svg.set('viewBox', f'0 0 {width} {height}')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        
        # Add background
        bg = ET.SubElement(svg, 'rect')
        bg.set('width', str(width))
        bg.set('height', str(height))
        bg.set('fill', '#f8f8f8')  # Traditional light background
        
        # Add title
        title = ET.SubElement(svg, 'title')
        title.text = f"Authentic {analysis['pattern_type'].replace('_', ' ').title()}"
        
        # Add metadata
        desc = ET.SubElement(svg, 'desc')
        desc.text = f"Generated using authentic dot grid system. Grid: {analysis['grid_size']}, Symmetry: {analysis['symmetry']}"
        
        # Draw dots (pulli)
        dots_group = ET.SubElement(svg, 'g')
        dots_group.set('id', 'pulli-dots')
        
        for x, y in dots:
            dot = ET.SubElement(dots_group, 'circle')
            dot.set('cx', str(x))
            dot.set('cy', str(y))
            dot.set('r', str(self.dot_radius))
            dot.set('fill', '#666666')
            dot.set('opacity', '0.3')
        
        # Draw curves
        curves_group = ET.SubElement(svg, 'g')
        curves_group.set('id', 'kolam-curves')
        curves_group.set('fill', 'none')
        curves_group.set('stroke', '#2458ff')
        curves_group.set('stroke-width', str(self.line_width))
        curves_group.set('stroke-linecap', 'round')
        curves_group.set('stroke-linejoin', 'round')
        
        for curve in curves:
            if not curve['points']:
                continue
                
            path = ET.SubElement(curves_group, 'path')
            path_data = self.create_path_data_from_points(curve['points'], curve['type'])
            path.set('d', path_data)
            
            # Style based on curve type
            if curve['style'] == 'petal':
                path.set('stroke', '#ff6b35')
                path.set('stroke-width', str(self.line_width * 0.8))
            elif curve['style'] == 'flowing':
                path.set('stroke', '#2458ff')
                path.set('stroke-width', str(self.line_width * 1.2))
        
        # Convert to string
        rough_string = ET.tostring(svg, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    def create_path_data_from_points(self, points: List[Tuple[int, int]], curve_type: str) -> str:
        """Create SVG path data from curve points"""
        if not points:
            return ""
        
        path_data = f"M {points[0][0]} {points[0][1]}"
        
        if len(points) < 3:
            # Simple line
            for x, y in points[1:]:
                path_data += f" L {x} {y}"
        else:
            # Smooth curve using quadratic Bezier
            for i in range(1, len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                
                # Use current point as control point for smooth curve
                path_data += f" Q {x1} {y1} {(x1 + x2) // 2} {(y1 + y2) // 2}"
            
            # Final point
            if len(points) > 2:
                x, y = points[-1]
                path_data += f" T {x} {y}"
        
        # Close path if it's a closed curve
        if curve_type == 'closed_curve':
            path_data += " Z"
        
        return path_data

def generate_authentic_dot_grid_kolam(image_path: str) -> str:
    """Generate authentic kolam with proper dot grid and curves"""
    generator = AuthenticDotGridKolamGenerator()
    return generator.generate_kolam_svg(image_path)
#!/usr/bin/env python3
"""
Enhanced Kolam Generator with Polar Sector and Contour-based Methods
Implements:
1. Geometric rangoli → Polar sector method
2. Animal/flower rangoli → Contour + dots + arcs method
"""

import cv2
import numpy as np
import svgwrite
import math
from typing import Tuple, List, Dict, Any, Optional
from scipy import ndimage
from sklearn.cluster import KMeans


class EnhancedKolamGenerator:
    def __init__(self):
        self.pattern_types = {
            'geometric': 'polar_sector',
            'floral': 'contour_dots_arcs', 
            'animal': 'contour_dots_arcs',
            'mandala': 'polar_sector',
            'abstract': 'hybrid'
        }
    
    def analyze_pattern_type(self, img_path: str) -> str:
        """Analyze input image to determine pattern type"""
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
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect circles and radial symmetry for geometric patterns
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, 
                                  param1=50, param2=30, minRadius=0, maxRadius=0)
        
        # Detect radial symmetry
        radial_score = self._calculate_radial_symmetry(gray)
        
        # Detect organic shapes (contours with high curvature)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        organic_score = self._calculate_organic_score(contours)
        
        # Color analysis for floral detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        floral_colors = self._detect_floral_colors(hsv)
        
        print(f"Analysis - Radial: {radial_score:.3f}, Organic: {organic_score:.3f}, Floral: {floral_colors}")
        
        # Decision logic
        if radial_score > 0.4 and circles is not None:
            return 'geometric'
        elif organic_score > 0.6 or floral_colors > 0.5:
            if floral_colors > 0.7:
                return 'floral'
            else:
                return 'animal'
        elif radial_score > 0.3:
            return 'mandala'
        else:
            return 'abstract'

    def _calculate_radial_symmetry(self, gray_img: np.ndarray) -> float:
        """Calculate radial symmetry score"""
        h, w = gray_img.shape
        center_x, center_y = w // 2, h // 2
        
        # Create polar coordinates
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        theta = np.arctan2(y - center_y, x - center_x)
        
        # Sample radial profiles at different angles
        n_angles = 8
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        
        profiles = []
        for angle in angles:
            mask = np.abs(theta - angle) < (np.pi / n_angles)
            if np.any(mask):
                profile = gray_img[mask]
                profiles.append(profile.flatten()[:50])  # Limit profile length
        
        if len(profiles) < 2:
            return 0.0
        
        # Calculate correlation between profiles
        correlations = []
        for i in range(len(profiles)):
            for j in range(i+1, len(profiles)):
                prof1, prof2 = profiles[i], profiles[j]
                min_len = min(len(prof1), len(prof2))
                if min_len > 5:
                    corr = np.corrcoef(prof1[:min_len], prof2[:min_len])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0

    def _calculate_organic_score(self, contours: List) -> float:
        """Calculate organic/natural shape score"""
        if not contours:
            return 0.0
        
        scores = []
        for contour in contours:
            if len(contour) > 10:
                # Calculate curvature variation
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # High curvature variation indicates organic shapes
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0 and area > 100:
                    # Compactness - circles are ~1, irregular shapes are higher
                    compactness = (perimeter**2) / (4 * np.pi * area)
                    
                    # Convexity defects indicate organic shapes
                    try:
                        hull = cv2.convexHull(contour, returnPoints=False)
                        if len(hull) > 3:  # Need at least 4 points for defects
                            defects = cv2.convexityDefects(contour, hull)
                            defect_score = len(defects) if defects is not None else 0
                        else:
                            defect_score = 0
                    except cv2.error:
                        # Skip convexity defects if contour is problematic
                        defect_score = 0
                    
                    organic_score = min(1.0, (compactness - 1) * 0.3 + defect_score * 0.1)
                    scores.append(organic_score)
        
        return np.mean(scores) if scores else 0.0

    def _detect_floral_colors(self, hsv_img: np.ndarray) -> float:
        """Detect floral color patterns"""
        # Typical floral colors in HSV
        floral_ranges = [
            ([0, 100, 100], [10, 255, 255]),    # Red
            ([160, 100, 100], [179, 255, 255]), # Red (wrap)
            ([20, 100, 100], [30, 255, 255]),   # Orange/Yellow
            ([40, 100, 100], [80, 255, 255]),   # Green
            ([100, 100, 100], [130, 255, 255]), # Blue/Purple
        ]
        
        total_floral_pixels = 0
        total_pixels = hsv_img.shape[0] * hsv_img.shape[1]
        
        for lower, upper in floral_ranges:
            mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
            total_floral_pixels += np.sum(mask > 0)
        
        return total_floral_pixels / total_pixels

    def generate_enhanced_kolam(self, img_path: str, spacing: int = 25) -> str:
        """Generate enhanced kolam based on pattern type"""
        pattern_type = self.analyze_pattern_type(img_path)
        print(f"Detected pattern type: {pattern_type}")
        
        if pattern_type in ['geometric', 'mandala']:
            return self._generate_polar_sector_kolam(img_path, spacing)
        elif pattern_type in ['floral', 'animal']:
            return self._generate_contour_dots_arcs_kolam(img_path, spacing)
        else:  # abstract
            return self._generate_hybrid_kolam(img_path, spacing)

    def _generate_polar_sector_kolam(self, img_path: str, spacing: int) -> str:
        """Generate kolam using polar sector method for geometric patterns"""
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Resize if needed
        max_dimension = 400
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
            gray = cv2.resize(gray, (new_width, new_height))
            height, width = new_height, new_width
        
        # Create SVG
        dwg = svgwrite.Drawing(size=(f'{width}px', f'{height}px'), 
                              viewBox=f'0 0 {width} {height}')
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='#0a0a0a'))
        
        # Find center
        center_x, center_y = width // 2, height // 2
        
        # Define polar sectors
        n_sectors = 8  # Number of radial sectors
        max_radius = min(width, height) // 2 - 20
        
        # Generate colors based on image
        colors = self._extract_dominant_colors(img)
        
        # Create concentric rings
        n_rings = max(3, max_radius // (spacing * 2))
        ring_radii = np.linspace(spacing, max_radius, n_rings)
        
        print(f"Generating polar kolam: {n_sectors} sectors, {n_rings} rings")
        
        # For each ring and sector intersection, add kolam elements
        for ring_idx, radius in enumerate(ring_radii):
            for sector in range(n_sectors):
                angle = (2 * np.pi * sector) / n_sectors
                
                # Calculate position
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                
                # Add dot
                dwg.add(dwg.circle(
                    center=(x, y), r=3,
                    fill=colors['dot'],
                    opacity=0.9
                ))
                
                # Add radial patterns
                self._add_radial_pattern(dwg, x, y, angle, radius, colors, spacing, ring_idx)
                
                # Add connecting arcs between adjacent sectors
                if sector < n_sectors - 1:
                    next_angle = (2 * np.pi * (sector + 1)) / n_sectors
                    next_x = center_x + radius * np.cos(next_angle)
                    next_y = center_y + radius * np.sin(next_angle)
                    self._add_arc_connection(dwg, x, y, next_x, next_y, colors['primary'])
        
        # Add connecting rings between radii
        for ring_idx in range(len(ring_radii) - 1):
            r1, r2 = ring_radii[ring_idx], ring_radii[ring_idx + 1]
            for sector in range(n_sectors):
                angle = (2 * np.pi * sector) / n_sectors
                x1 = center_x + r1 * np.cos(angle)
                y1 = center_y + r1 * np.sin(angle)
                x2 = center_x + r2 * np.cos(angle)
                y2 = center_y + r2 * np.sin(angle)
                
                # Curved radial connection
                mid_x = (x1 + x2) / 2 + 10 * np.cos(angle + np.pi/2)
                mid_y = (y1 + y2) / 2 + 10 * np.sin(angle + np.pi/2)
                
                path_data = f"M {x1:.1f} {y1:.1f} Q {mid_x:.1f} {mid_y:.1f} {x2:.1f} {y2:.1f}"
                dwg.add(dwg.path(d=path_data, stroke=colors['secondary'], 
                               stroke_width=2, fill='none', opacity=0.8))
        
        return dwg.tostring()

    def _generate_contour_dots_arcs_kolam(self, img_path: str, spacing: int) -> str:
        """Generate kolam using contour + dots + arcs method for organic patterns"""
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Resize if needed
        max_dimension = 400
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
            gray = cv2.resize(gray, (new_width, new_height))
            height, width = new_height, new_width
        
        # Create SVG
        dwg = svgwrite.Drawing(size=(f'{width}px', f'{height}px'), 
                              viewBox=f'0 0 {width} {height}')
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='#0a0a0a'))
        
        # Extract main contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter significant contours
        significant_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                significant_contours.append(contour)
        
        print(f"Found {len(significant_contours)} significant contours")
        
        # Generate colors
        colors = self._extract_dominant_colors(img)
        
        # Process each contour
        all_dots = []
        
        for contour_idx, contour in enumerate(significant_contours):
            # Simplify contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Generate dots along contour
            contour_dots = self._generate_contour_dots(approx, spacing)
            all_dots.extend(contour_dots)
            
            # Add decorative elements along contour
            self._add_contour_decorations(dwg, approx, colors, contour_idx)
        
        # Create internal structure with dots and arcs
        internal_dots = self._generate_internal_dots(gray, edges, spacing, all_dots)
        all_dots.extend(internal_dots)
        
        print(f"Generated {len(all_dots)} dots total")
        
        # Add all dots
        for dot in all_dots:
            x, y = dot['x'], dot['y']
            dot_color = colors['dot'] if dot.get('type') == 'contour' else colors['accent']
            dwg.add(dwg.circle(
                center=(x, y), r=dot.get('size', 3),
                fill=dot_color,
                opacity=0.9
            ))
        
        # Connect dots with organic arcs
        self._connect_with_organic_arcs(dwg, all_dots, colors, spacing)
        
        return dwg.tostring()

    def _generate_contour_dots(self, contour: np.ndarray, spacing: int) -> List[Dict]:
        """Generate dots along contour"""
        dots = []
        if len(contour) < 3:
            return dots
        
        # Calculate total perimeter
        perimeter = cv2.arcLength(contour, True)
        n_dots = max(6, int(perimeter / spacing))
        
        for i in range(n_dots):
            # Interpolate along contour
            t = i / n_dots
            idx = int(t * (len(contour) - 1))
            
            if idx < len(contour):
                point = contour[idx][0]
                dots.append({
                    'x': float(point[0]),
                    'y': float(point[1]),
                    'type': 'contour',
                    'size': 4
                })
        
        return dots

    def _generate_internal_dots(self, gray: np.ndarray, edges: np.ndarray, 
                               spacing: int, contour_dots: List[Dict]) -> List[Dict]:
        """Generate internal dots based on image structure"""
        height, width = gray.shape
        internal_dots = []
        
        # Create interest point map
        interest_map = cv2.goodFeaturesToTrack(gray, maxCorners=50, 
                                             qualityLevel=0.01, minDistance=spacing)
        
        if interest_map is not None:
            for point in interest_map:
                x, y = point.ravel()
                
                # Check if far enough from contour dots
                min_dist = min([((x - d['x'])**2 + (y - d['y'])**2)**0.5 
                               for d in contour_dots], default=float('inf'))
                
                if min_dist > spacing * 0.7:
                    # Analyze local region
                    local_region = self._get_local_region(edges, int(x), int(y), spacing//3)
                    if local_region is not None and np.mean(local_region) > 30:
                        internal_dots.append({
                            'x': float(x),
                            'y': float(y),
                            'type': 'internal',
                            'size': 3
                        })
        
        return internal_dots

    def _connect_with_organic_arcs(self, dwg: svgwrite.Drawing, dots: List[Dict], 
                                  colors: Dict, spacing: int):
        """Connect dots with organic, flowing arcs"""
        for i, dot1 in enumerate(dots):
            for j, dot2 in enumerate(dots[i+1:], i+1):
                dist = ((dot1['x'] - dot2['x'])**2 + (dot1['y'] - dot2['y'])**2)**0.5
                
                # Only connect nearby dots
                if dist < spacing * 1.5 and dist > spacing * 0.3:
                    # Create organic curve
                    x1, y1 = dot1['x'], dot1['y']
                    x2, y2 = dot2['x'], dot2['y']
                    
                    # Add curvature based on relative position
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    
                    # Perpendicular offset for curve
                    dx, dy = x2 - x1, y2 - y1
                    length = (dx**2 + dy**2)**0.5
                    if length > 0:
                        perp_x = -dy / length * spacing * 0.3
                        perp_y = dx / length * spacing * 0.3
                        
                        ctrl_x = mid_x + perp_x
                        ctrl_y = mid_y + perp_y
                        
                        # Determine color based on dot types
                        if dot1.get('type') == 'contour' and dot2.get('type') == 'contour':
                            stroke_color = colors['primary']
                        else:
                            stroke_color = colors['secondary']
                        
                        path_data = f"M {x1:.1f} {y1:.1f} Q {ctrl_x:.1f} {ctrl_y:.1f} {x2:.1f} {y2:.1f}"
                        dwg.add(dwg.path(d=path_data, stroke=stroke_color, 
                                       stroke_width=1.5, fill='none', opacity=0.7))

    def _generate_hybrid_kolam(self, img_path: str, spacing: int) -> str:
        """Generate hybrid kolam combining both methods"""
        # For now, default to contour method for abstract patterns
        return self._generate_contour_dots_arcs_kolam(img_path, spacing)

    def _extract_dominant_colors(self, img: np.ndarray) -> Dict[str, str]:
        """Extract dominant colors from image for kolam palette"""
        # Reshape image for color clustering
        data = img.reshape((-1, 3))
        
        # Use K-means to find dominant colors
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(data)
        colors_bgr = kmeans.cluster_centers_.astype(int)
        
        # Convert to hex and create palette
        def bgr_to_hex(bgr):
            return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"
        
        hex_colors = [bgr_to_hex(color) for color in colors_bgr]
        
        # Traditional kolam color mapping
        return {
            'primary': hex_colors[0],
            'secondary': hex_colors[1] if len(hex_colors) > 1 else '#E24A90',
            'accent': hex_colors[2] if len(hex_colors) > 2 else '#90E24A',
            'dot': 'white',
            'background': '#0a0a0a'
        }

    def _add_radial_pattern(self, dwg: svgwrite.Drawing, x: float, y: float, 
                           angle: float, radius: float, colors: Dict, 
                           spacing: int, ring_idx: int):
        """Add radial pattern elements"""
        # Petal-like decorations
        petal_length = spacing * 0.6
        
        # Outer petal
        outer_x = x + petal_length * np.cos(angle)
        outer_y = y + petal_length * np.sin(angle)
        
        # Control points for petal curve
        ctrl1_x = x + petal_length * 0.3 * np.cos(angle - 0.3)
        ctrl1_y = y + petal_length * 0.3 * np.sin(angle - 0.3)
        ctrl2_x = x + petal_length * 0.3 * np.cos(angle + 0.3)
        ctrl2_y = y + petal_length * 0.3 * np.sin(angle + 0.3)
        
        path_data = f"M {x:.1f} {y:.1f} Q {ctrl1_x:.1f} {ctrl1_y:.1f} {outer_x:.1f} {outer_y:.1f} Q {ctrl2_x:.1f} {ctrl2_y:.1f} {x:.1f} {y:.1f}"
        
        petal_color = colors['accent'] if ring_idx % 2 == 0 else colors['secondary']
        dwg.add(dwg.path(d=path_data, stroke=petal_color, 
                        stroke_width=1.5, fill='none', opacity=0.6))

    def _add_arc_connection(self, dwg: svgwrite.Drawing, x1: float, y1: float, 
                           x2: float, y2: float, color: str):
        """Add curved connection between two points"""
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Add slight outward curve
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate perpendicular offset
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2)**0.5
        
        if length > 0:
            perp_x = -dy / length * 8
            perp_y = dx / length * 8
            
            ctrl_x = mid_x + perp_x
            ctrl_y = mid_y + perp_y
            
            path_data = f"M {x1:.1f} {y1:.1f} Q {ctrl_x:.1f} {ctrl_y:.1f} {x2:.1f} {y2:.1f}"
            dwg.add(dwg.path(d=path_data, stroke=color, 
                           stroke_width=2, fill='none', opacity=0.8))

    def _add_contour_decorations(self, dwg: svgwrite.Drawing, contour: np.ndarray, 
                               colors: Dict, contour_idx: int):
        """Add decorative elements along contour"""
        if len(contour) < 3:
            return
        
        # Add flowing lines along contour
        for i in range(0, len(contour) - 2, 3):  # Every 3rd point
            p1 = contour[i][0]
            p2 = contour[i + 1][0]
            p3 = contour[i + 2][0]
            
            # Create flowing curve
            path_data = f"M {p1[0]} {p1[1]} Q {p2[0]} {p2[1]} {p3[0]} {p3[1]}"
            
            decoration_color = colors['primary'] if contour_idx % 2 == 0 else colors['secondary']
            dwg.add(dwg.path(d=path_data, stroke=decoration_color, 
                           stroke_width=1.5, fill='none', opacity=0.6))

    def _get_local_region(self, img: np.ndarray, x: int, y: int, radius: int) -> Optional[np.ndarray]:
        """Extract local region around a point"""
        h, w = img.shape[:2]
        x1 = max(0, x - radius)
        x2 = min(w, x + radius)
        y1 = max(0, y - radius)
        y2 = min(h, y + radius)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return img[y1:y2, x1:x2]


def generate_enhanced_kolam_from_image(img_path: str, spacing: int = 25) -> str:
    """Main function to generate enhanced kolam from image"""
    try:
        generator = EnhancedKolamGenerator()
        return generator.generate_enhanced_kolam(img_path, spacing)
    except Exception as e:
        print(f"Error generating enhanced kolam: {e}")
        # Fallback to simple kolam if enhanced fails
        from kolam.improved_authentic_kolam import generate_improved_kolam_from_image
        return generate_improved_kolam_from_image(img_path, spacing)
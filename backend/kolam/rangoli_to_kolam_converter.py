#!/usr/bin/env python3
"""
Rangoli to Kolam Converter
Converts specific rangoli patterns to authentic kolam with proper dot grid foundation.
Uses traditional Tamil kolam techniques: pulli (dots), curved lines, loops, and authentic patterns.
"""

import cv2
import numpy as np
import svgwrite
import math
from typing import Tuple, List, Dict, Any, Optional
from .authentic_dot_grid_kolam_generator import AuthenticDotGridKolamGenerator

class RangoliToKolamConverter:
    def __init__(self):
        # Traditional kolam colors
        self.kolam_colors = {
            'white': '#ffffff',      # Primary kolam color (rice flour)
            'yellow': '#ffd700',     # Turmeric
            'red': '#dc143c',        # Kumkum
            'orange': '#ff6347',     # Marigold
            'blue': '#4169e1',       # Blue chalk
            'green': '#228b22',      # Green powder
            'purple': '#8a2be2'      # Purple powder
        }
        
        # Canvas settings
        self.canvas_size = 500
        self.background_color = '#f8f8f8'  # Light background for kolam

    def convert_rangoli_to_kolam(self, img_path: str) -> str:
        """Main conversion function: rangoli image → authentic kolam SVG with dot grid"""
        try:
            # Use authentic dot grid kolam generator
            authentic_generator = AuthenticDotGridKolamGenerator()
            kolam_svg = authentic_generator.generate_kolam_svg(img_path)
            
            return kolam_svg
            
        except Exception as e:
            print(f"Error converting rangoli to authentic kolam: {e}")
            # Fallback to traditional method if authentic fails
            try:
                # Load and analyze the rangoli image
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Could not load image: {img_path}")
                
                # Extract key features from rangoli
                features = self._extract_rangoli_features(img)
                
                # Convert to kolam-style representation
                kolam_svg = self._generate_kolam_from_features(features)
                
                return kolam_svg
                
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                return self._generate_authentic_fallback_kolam()

    def _extract_rangoli_features(self, img: np.ndarray) -> Dict[str, Any]:
        """Extract key structural features from rangoli image"""
        # Resize for processing
        height, width = img.shape[:2]
        max_dim = 400
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = {
            # Basic image properties
            'height': img.shape[0],
            'width': img.shape[1],
            'center': (img.shape[1]//2, img.shape[0]//2),
            
            # Symmetry analysis
            'radial_symmetry': self._detect_radial_symmetry(gray),
            'symmetry_order': self._detect_symmetry_order(gray),
            
            # Shape analysis
            'central_element': self._detect_central_element(gray),
            'petal_elements': self._detect_petal_elements(gray, img),
            'corner_elements': self._detect_corner_elements(gray),
            
            # Color analysis
            'dominant_colors': self._extract_dominant_colors(img),
            'color_regions': self._segment_color_regions(img),
            
            # Pattern structure
            'concentric_rings': self._detect_concentric_structure(gray),
            'geometric_elements': self._detect_geometric_elements(gray)
        }
        
        return features

    def _detect_radial_symmetry(self, gray: np.ndarray) -> float:
        """Detect strength of radial symmetry"""
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        
        # Create polar representation
        max_radius = min(center_x, center_y) - 10
        symmetry_scores = []
        
        for test_order in [4, 6, 8, 12, 16]:
            angle_step = 2 * math.pi / test_order
            score = 0
            
            for r in range(10, max_radius, 10):
                reference_values = []
                for i in range(test_order):
                    angle = i * angle_step
                    x = center_x + int(r * math.cos(angle))
                    y = center_y + int(r * math.sin(angle))
                    
                    if 0 <= x < w and 0 <= y < h:
                        reference_values.append(gray[y, x])
                
                if len(reference_values) == test_order:
                    # Calculate variance - lower variance means higher symmetry
                    variance = np.var(reference_values)
                    score += 1.0 / (1.0 + variance / 1000.0)
            
            symmetry_scores.append(score)
        
        return max(symmetry_scores) / max_radius * 10 if max_radius > 0 else 0

    def _detect_symmetry_order(self, gray: np.ndarray) -> int:
        """Detect the order of radial symmetry (4-fold, 8-fold, etc.)"""
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        max_radius = min(center_x, center_y) - 20
        
        best_order = 8  # Default for many rangoli patterns
        best_score = 0
        
        for test_order in [4, 6, 8, 12, 16]:
            angle_step = 2 * math.pi / test_order
            score = 0
            
            for r in range(20, max_radius, 15):
                values = []
                for i in range(test_order):
                    angle = i * angle_step
                    x = center_x + int(r * math.cos(angle))
                    y = center_y + int(r * math.sin(angle))
                    
                    if 0 <= x < w and 0 <= y < h:
                        values.append(gray[y, x])
                
                if len(values) == test_order:
                    # Compare each value with the next one (rotational consistency)
                    consistency = sum(1 for i in range(test_order) 
                                    if abs(values[i] - values[(i+1) % test_order]) < 30)
                    score += consistency / test_order
            
            if score > best_score:
                best_score = score
                best_order = test_order
        
        return best_order

    def _detect_central_element(self, gray: np.ndarray) -> Dict[str, Any]:
        """Detect and analyze central element (e.g., peacock)"""
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        
        # Define central region (approximately 25% of image)
        central_size = min(h, w) // 4
        central_region = gray[center_y - central_size:center_y + central_size,
                            center_x - central_size:center_x + central_size]
        
        # Detect if central region has complex structure (like peacock)
        edges = cv2.Canny(central_region, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze complexity
        total_contour_length = sum(cv2.arcLength(c, True) for c in contours)
        has_complex_center = total_contour_length > central_size * 2
        
        # Detect shape characteristics
        shape_type = "complex" if has_complex_center else "simple"
        if has_complex_center and len(contours) > 0:
            # Check if it resembles a bird/peacock shape
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)
            convexity = cv2.contourArea(largest_contour) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            
            if convexity < 0.7:  # Non-convex shapes like birds
                shape_type = "bird_like"
        
        return {
            'has_complex_element': has_complex_center,
            'shape_type': shape_type,
            'central_size': central_size,
            'complexity_score': total_contour_length
        }

    def _detect_petal_elements(self, gray: np.ndarray, color_img: np.ndarray) -> Dict[str, Any]:
        """Detect petal-like radial elements"""
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        
        # Create annular regions for petal detection
        inner_radius = min(h, w) // 6
        outer_radius = min(h, w) // 3
        
        # Create mask for petal region
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        petal_mask = (distances >= inner_radius) & (distances <= outer_radius)
        
        # Apply mask and detect edges
        petal_region = gray.copy()
        petal_region[~petal_mask] = 0
        edges = cv2.Canny(petal_region, 30, 100)
        
        # Find contours in petal region
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze petal-like shapes
        petal_shapes = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small noise
                # Get the centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate angle from center
                    angle = math.atan2(cy - center_y, cx - center_x)
                    distance = math.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                    
                    petal_shapes.append({
                        'angle': angle,
                        'distance': distance,
                        'area': cv2.contourArea(contour),
                        'center': (cx, cy)
                    })
        
        # Estimate petal count based on angular distribution
        if len(petal_shapes) > 0:
            angles = [p['angle'] for p in petal_shapes]
            # Sort angles and find consistent spacing
            angles.sort()
            angle_diffs = []
            for i in range(len(angles)):
                diff = angles[(i+1) % len(angles)] - angles[i]
                if diff < 0:
                    diff += 2 * math.pi
                angle_diffs.append(diff)
            
            # Estimate number of petals based on average angular spacing
            avg_spacing = np.mean(angle_diffs) if angle_diffs else math.pi / 4
            estimated_count = int(2 * math.pi / avg_spacing) if avg_spacing > 0 else 8
            estimated_count = max(4, min(16, estimated_count))  # Reasonable bounds
        else:
            estimated_count = 8  # Default
        
        return {
            'petal_count': estimated_count,
            'petal_shapes': petal_shapes,
            'has_petal_structure': len(petal_shapes) > 0
        }

    def _detect_corner_elements(self, gray: np.ndarray) -> Dict[str, Any]:
        """Detect decorative elements in corners"""
        h, w = gray.shape
        corner_size = min(h, w) // 6
        
        # Define corner regions
        corners = {
            'top_left': gray[0:corner_size, 0:corner_size],
            'top_right': gray[0:corner_size, w-corner_size:w],
            'bottom_left': gray[h-corner_size:h, 0:corner_size],
            'bottom_right': gray[h-corner_size:h, w-corner_size:w]
        }
        
        corner_elements = []
        for corner_name, corner_region in corners.items():
            # Detect features in corner
            edges = cv2.Canny(corner_region, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            total_area = sum(cv2.contourArea(c) for c in contours)
            has_decoration = total_area > corner_size * corner_size * 0.1
            
            corner_elements.append({
                'name': corner_name,
                'has_decoration': has_decoration,
                'complexity': total_area
            })
        
        return {
            'corner_decorations': corner_elements,
            'has_corner_elements': any(c['has_decoration'] for c in corner_elements)
        }

    def _extract_dominant_colors(self, img: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image"""
        # Reshape image to list of pixels
        pixels = img.reshape(-1, 3)
        
        # Use k-means clustering to find dominant colors
        from sklearn.cluster import KMeans
        
        n_colors = 5
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get colors and sort by cluster size
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        color_counts = []
        for i in range(n_colors):
            count = np.sum(labels == i)
            color_counts.append((count, tuple(colors[i])))
        
        # Sort by frequency and return colors
        color_counts.sort(reverse=True)
        return [color for _, color in color_counts]

    def _segment_color_regions(self, img: np.ndarray) -> Dict[str, Any]:
        """Segment image into color regions"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common rangoli colors
        color_ranges = {
            'blue': [(100, 50, 50), (130, 255, 255)],
            'green': [(40, 50, 50), (80, 255, 255)],
            'red': [(0, 50, 50), (20, 255, 255)],
            'orange': [(10, 50, 50), (25, 255, 255)],
            'yellow': [(25, 50, 50), (35, 255, 255)],
            'white': [(0, 0, 200), (180, 30, 255)]
        }
        
        color_regions = {}
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours in color region
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50)
            
            color_regions[color_name] = {
                'area': total_area,
                'contours': len(contours),
                'present': total_area > img.shape[0] * img.shape[1] * 0.01  # At least 1% of image
            }
        
        return color_regions

    def _detect_concentric_structure(self, gray: np.ndarray) -> Dict[str, Any]:
        """Detect concentric ring structures"""
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        max_radius = min(center_x, center_y) - 10
        
        # Sample along radial lines to detect rings
        ring_structures = []
        n_angles = 8  # Sample at 8 different angles
        
        for angle_idx in range(n_angles):
            angle = (2 * math.pi * angle_idx) / n_angles
            radial_profile = []
            
            for r in range(5, max_radius, 2):
                x = center_x + int(r * math.cos(angle))
                y = center_y + int(r * math.sin(angle))
                
                if 0 <= x < w and 0 <= y < h:
                    radial_profile.append(gray[y, x])
            
            if len(radial_profile) > 20:
                # Find peaks in radial profile (indicating ring boundaries)
                profile_array = np.array(radial_profile)
                
                # Find local minima (dark rings) and maxima (bright rings)
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(profile_array, height=100, distance=5)
                valleys, _ = find_peaks(-profile_array, height=-200, distance=5)
                
                ring_structures.append({
                    'angle': angle,
                    'peaks': len(peaks),
                    'valleys': len(valleys),
                    'total_features': len(peaks) + len(valleys)
                })
        
        # Analyze consistency of ring structure
        if ring_structures:
            avg_features = np.mean([r['total_features'] for r in ring_structures])
            std_features = np.std([r['total_features'] for r in ring_structures])
            
            has_concentric = avg_features > 2 and std_features < 2  # Consistent rings
        else:
            has_concentric = False
            avg_features = 0
        
        return {
            'has_concentric_rings': has_concentric,
            'estimated_rings': int(avg_features) if has_concentric else 0,
            'ring_consistency': 1.0 - min(1.0, std_features / max(1.0, avg_features)) if has_concentric else 0
        }

    def _detect_geometric_elements(self, gray: np.ndarray) -> Dict[str, Any]:
        """Detect geometric shapes and patterns"""
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=20, maxLineGap=10)
        
        # Detect circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=100)
        
        # Find contours and approximate to polygons
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        geometric_shapes = {
            'triangles': 0,
            'rectangles': 0,
            'polygons': 0,
            'curved_shapes': 0
        }
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                n_vertices = len(approx)
                if n_vertices == 3:
                    geometric_shapes['triangles'] += 1
                elif n_vertices == 4:
                    geometric_shapes['rectangles'] += 1
                elif n_vertices <= 8:
                    geometric_shapes['polygons'] += 1
                else:
                    geometric_shapes['curved_shapes'] += 1
        
        return {
            'line_count': len(lines) if lines is not None else 0,
            'circle_count': len(circles[0]) if circles is not None else 0,
            'geometric_shapes': geometric_shapes,
            'total_geometric_elements': sum(geometric_shapes.values())
        }

    def _generate_kolam_from_features(self, features: Dict[str, Any]) -> str:
        """Generate kolam SVG based on extracted features"""
        # Create SVG canvas
        dwg = svgwrite.Drawing(size=(f'{self.canvas_size}px', f'{self.canvas_size}px'), 
                              viewBox=f'0 0 {self.canvas_size} {self.canvas_size}')
        
        # Light background for kolam
        dwg.add(dwg.rect(insert=(0, 0), size=(self.canvas_size, self.canvas_size), 
                        fill=self.background_color))
        
        center = self.canvas_size // 2
        
        # Generate kolam based on detected features
        symmetry_order = features.get('symmetry_order', 8)
        
        # 1. Create central element (converted from peacock to geometric bird)
        if features['central_element']['shape_type'] == 'bird_like':
            self._create_geometric_bird_center(dwg, center, features)
        else:
            self._create_simple_center(dwg, center, features)
        
        # 2. Create petal/feather elements around center
        if features['petal_elements']['has_petal_structure']:
            self._create_geometric_petals(dwg, center, features, symmetry_order)
        
        # 3. Create concentric rings if detected
        if features['concentric_rings']['has_concentric_rings']:
            self._create_concentric_kolam_rings(dwg, center, features)
        
        # 4. Create corner decorations if detected
        if features['corner_elements']['has_corner_elements']:
            self._create_kolam_corner_decorations(dwg, features)
        
        return dwg.tostring()

    def _create_geometric_bird_center(self, dwg: svgwrite.Drawing, center: int, features: Dict[str, Any]):
        """Create geometric bird silhouette in kolam style"""
        # Simple geometric representation of bird (peacock → stylized bird)
        
        # Central circle for bird body
        body_radius = 15
        dwg.add(dwg.circle(center=(center, center), r=body_radius,
                         stroke=self.kolam_colors['blue'], stroke_width=2.5,
                         fill=self.kolam_colors['white'], opacity=0.9))
        
        # Bird head (small circle above body)
        head_y = center - body_radius - 8
        dwg.add(dwg.circle(center=(center + 5, head_y), r=6,
                         stroke=self.kolam_colors['blue'], stroke_width=2,
                         fill=self.kolam_colors['blue'], opacity=0.8))
        
        # Stylized tail feathers (geometric curves)
        tail_angles = [-30, -15, 0, 15, 30]  # 5 tail feathers
        for angle in tail_angles:
            angle_rad = math.radians(angle)
            
            # Feather start point
            start_x = center - body_radius * 0.7
            start_y = center + 2
            
            # Feather end point
            feather_length = 25
            end_x = start_x - feather_length * math.cos(angle_rad + math.pi/2)
            end_y = start_y - feather_length * math.sin(angle_rad + math.pi/2)
            
            # Create curved feather
            control_x = start_x - feather_length * 0.6
            control_y = start_y - feather_length * 0.3
            
            path_data = f"M {start_x:.1f} {start_y:.1f} Q {control_x:.1f} {control_y:.1f} {end_x:.1f} {end_y:.1f}"
            
            dwg.add(dwg.path(d=path_data, stroke=self.kolam_colors['orange'], 
                           stroke_width=2.5, fill='none', opacity=0.8))
            
            # Add small circle at feather tip (eye pattern)
            dwg.add(dwg.circle(center=(end_x, end_y), r=3,
                             fill=self.kolam_colors['red'], opacity=0.9))

    def _create_simple_center(self, dwg: svgwrite.Drawing, center: int, features: Dict[str, Any]):
        """Create simple geometric center"""
        # Basic central mandala
        dwg.add(dwg.circle(center=(center, center), r=20,
                         stroke=self.kolam_colors['white'], stroke_width=3,
                         fill='none', opacity=1.0))
        
        # Inner decorative elements
        for i in range(8):
            angle = (2 * math.pi * i) / 8
            x = center + 12 * math.cos(angle)
            y = center + 12 * math.sin(angle)
            
            dwg.add(dwg.circle(center=(x, y), r=2,
                             fill=self.kolam_colors['yellow'], opacity=1.0))

    def _create_geometric_petals(self, dwg: svgwrite.Drawing, center: int, features: Dict[str, Any], symmetry_order: int):
        """Create geometric petal patterns"""
        petal_count = features['petal_elements'].get('petal_count', symmetry_order)
        petal_radius = 60
        
        for i in range(petal_count):
            angle = (2 * math.pi * i) / petal_count
            
            # Petal outer point
            outer_x = center + petal_radius * math.cos(angle)
            outer_y = center + petal_radius * math.sin(angle)
            
            # Petal base points
            base_angle1 = angle - math.pi / petal_count
            base_angle2 = angle + math.pi / petal_count
            base_radius = 25
            
            base1_x = center + base_radius * math.cos(base_angle1)
            base1_y = center + base_radius * math.sin(base_angle1)
            
            base2_x = center + base_radius * math.cos(base_angle2)
            base2_y = center + base_radius * math.sin(base_angle2)
            
            # Create petal shape
            path_data = f"M {base1_x:.1f} {base1_y:.1f} Q {outer_x:.1f} {outer_y:.1f} {base2_x:.1f} {base2_y:.1f}"
            
            # Alternate colors for visual appeal
            color = self.kolam_colors['blue'] if i % 2 == 0 else self.kolam_colors['green']
            dwg.add(dwg.path(d=path_data, stroke=color, stroke_width=3, 
                           fill='none', opacity=0.8))
            
            # Add dot at petal tip (traditional eye pattern)
            dwg.add(dwg.circle(center=(outer_x, outer_y), r=4,
                             fill=self.kolam_colors['orange'], opacity=0.9))

    def _create_concentric_kolam_rings(self, dwg: svgwrite.Drawing, center: int, features: Dict[str, Any]):
        """Create concentric ring patterns"""
        ring_count = features['concentric_rings'].get('estimated_rings', 3)
        
        for ring in range(1, ring_count + 1):
            radius = 80 + ring * 25
            
            # Create decorative ring with wave pattern
            n_points = 32
            path_data = "M "
            
            for i in range(n_points + 1):
                angle = (2 * math.pi * i) / n_points
                wave_amplitude = 8 * math.sin(angle * 6)  # 6 waves around
                
                ring_radius = radius + wave_amplitude
                x = center + ring_radius * math.cos(angle)
                y = center + ring_radius * math.sin(angle)
                
                if i == 0:
                    path_data += f"{x:.1f} {y:.1f} "
                else:
                    path_data += f"L {x:.1f} {y:.1f} "
            
            path_data += "Z"
            
            # Alternate ring colors
            color = self.kolam_colors['white'] if ring % 2 == 1 else self.kolam_colors['yellow']
            dwg.add(dwg.path(d=path_data, stroke=color, stroke_width=2, 
                           fill='none', opacity=0.7))

    def _create_kolam_corner_decorations(self, dwg: svgwrite.Drawing, features: Dict[str, Any]):
        """Create traditional kolam corner decorations"""
        corner_positions = [
            (60, 60),                                    # Top-left
            (self.canvas_size - 60, 60),                # Top-right
            (self.canvas_size - 60, self.canvas_size - 60), # Bottom-right
            (60, self.canvas_size - 60)                 # Bottom-left
        ]
        
        for cx, cy in corner_positions:
            # Create spiral decoration at corner
            self._create_corner_spiral(dwg, cx, cy)

    def _create_corner_spiral(self, dwg: svgwrite.Drawing, cx: int, cy: int):
        """Create spiral corner decoration"""
        path_data = f"M {cx:.1f} {cy:.1f} "
        
        n_turns = 2
        n_points = 30
        max_radius = 25
        
        for i in range(n_points):
            t = i / n_points
            angle = t * n_turns * 2 * math.pi
            radius = t * max_radius
            
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            
            path_data += f"L {x:.1f} {y:.1f} "
        
        dwg.add(dwg.path(d=path_data, stroke=self.kolam_colors['red'], 
                       stroke_width=2, fill='none', opacity=0.8))

    def _generate_fallback_kolam(self) -> str:
        """Generate a simple fallback kolam if conversion fails"""
        dwg = svgwrite.Drawing(size=(f'{self.canvas_size}px', f'{self.canvas_size}px'), 
                              viewBox=f'0 0 {self.canvas_size} {self.canvas_size}')
        
        dwg.add(dwg.rect(insert=(0, 0), size=(self.canvas_size, self.canvas_size), 
                        fill=self.background_color))
        
        center = self.canvas_size // 2
        
        # Simple mandala fallback
        dwg.add(dwg.circle(center=(center, center), r=50,
                         stroke=self.kolam_colors['white'], stroke_width=3,
                         fill='none'))
        
        for i in range(8):
            angle = (2 * math.pi * i) / 8
            x = center + 30 * math.cos(angle)
            y = center + 30 * math.sin(angle)
            
            dwg.add(dwg.circle(center=(x, y), r=5,
                             fill=self.kolam_colors['yellow']))
        
        return dwg.tostring()

    def _generate_authentic_fallback_kolam(self) -> str:
        """Generate an authentic dot-grid fallback kolam with proper structure"""
        try:
            # Use the authentic generator for fallback
            authentic_generator = AuthenticDotGridKolamGenerator()
            
            # Create a simple grid pattern for fallback
            dots = authentic_generator.generate_dot_grid((7, 7), (500, 500))
            
            # Create basic symmetric pattern
            curves = []
            if len(dots) > 4:
                # Simple connecting curves between nearby dots
                for i, dot1 in enumerate(dots):
                    for j, dot2 in enumerate(dots[i+1:], i+1):
                        if j - i < 3:  # Connect only nearby dots
                            curve_points = authentic_generator.generate_bezier_curve(dot1, dot2, 8)
                            curves.append({
                                'type': 'curve',
                                'points': curve_points,
                                'style': 'smooth'
                            })
            
            # Create SVG with dots and curves
            analysis = {
                'pattern_type': 'fallback_kolam',
                'grid_size': (7, 7),
                'symmetry': 4
            }
            
            return authentic_generator.create_svg_from_dots_and_curves(dots, curves, analysis)
            
        except Exception as e:
            print(f"Authentic fallback failed: {e}")
            # Ultimate fallback - simple SVG
            return '''<svg width="500" height="500" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
                <rect width="500" height="500" fill="#f8f8f8"/>
                <circle cx="250" cy="250" r="80" stroke="#2458ff" stroke-width="3" fill="none"/>
                <circle cx="250" cy="250" r="40" stroke="#ff6347" stroke-width="2" fill="none"/>
                <text x="250" y="480" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
                    Authentic Kolam (Fallback Pattern)
                </text>
            </svg>'''


# Main conversion function
def convert_rangoli_to_kolam(img_path: str) -> str:
    """Main function to convert rangoli image to authentic kolam SVG with dot grid"""
    converter = RangoliToKolamConverter()
    return converter.convert_rangoli_to_kolam(img_path)
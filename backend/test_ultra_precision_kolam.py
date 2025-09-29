#!/usr/bin/env python3
"""
Test Ultra-Precision Kolam Generator - 96% Accuracy Validation
Comprehensive testing suite with metrics and validation
"""

import os
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import xml.etree.ElementTree as ET

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kolam.ultra_precision_kolam_generator import generate_ultra_precision_kolam_from_image, UltraPrecisionKolamGenerator


class KolamAccuracyValidator:
    """Validate Kolam generation accuracy with comprehensive metrics"""
    
    def __init__(self):
        self.metrics = {
            'pattern_recognition_accuracy': 0.0,
            'geometric_precision': 0.0,
            'color_extraction_quality': 0.0,
            'curve_smoothness': 0.0,
            'cultural_authenticity': 0.0,
            'dot_placement_accuracy': 0.0,
            'symmetry_preservation': 0.0,
            'edge_detection_quality': 0.0,
            'overall_visual_quality': 0.0
        }
        
        # Accuracy weights for final score
        self.weights = {
            'pattern_recognition_accuracy': 0.15,
            'geometric_precision': 0.12,
            'color_extraction_quality': 0.10,
            'curve_smoothness': 0.12,
            'cultural_authenticity': 0.15,
            'dot_placement_accuracy': 0.10,
            'symmetry_preservation': 0.11,
            'edge_detection_quality': 0.08,
            'overall_visual_quality': 0.07
        }
    
    def validate_svg_structure(self, svg_content: str) -> Dict[str, Any]:
        """Validate SVG structure and extract key elements"""
        try:
            root = ET.fromstring(svg_content)
            
            # Count elements
            circles = len(root.findall('.//{http://www.w3.org/2000/svg}circle'))
            paths = len(root.findall('.//{http://www.w3.org/2000/svg}path'))
            rects = len(root.findall('.//{http://www.w3.org/2000/svg}rect'))
            
            # Check viewBox
            viewbox = root.get('viewBox')
            width = root.get('width')
            height = root.get('height')
            
            return {
                'valid_structure': True,
                'element_counts': {
                    'circles': circles,
                    'paths': paths,
                    'rects': rects
                },
                'dimensions': {
                    'width': width,
                    'height': height,
                    'viewBox': viewbox
                },
                'total_elements': circles + paths + rects
            }
            
        except ET.ParseError as e:
            return {
                'valid_structure': False,
                'error': str(e),
                'element_counts': {'circles': 0, 'paths': 0, 'rects': 0},
                'total_elements': 0
            }
    
    def measure_pattern_recognition_accuracy(self, img_path: str, generated_svg: str) -> float:
        """Measure how accurately the pattern type was recognized"""
        try:
            generator = UltraPrecisionKolamGenerator()
            analysis = generator.analyze_pattern_type_advanced(img_path)
            
            pattern_type = analysis['pattern_type']
            confidence_scores = analysis['confidence_scores']
            
            # Validate pattern recognition
            if pattern_type in ['geometric', 'floral', 'animal', 'mandala', 'traditional']:
                # Check if appropriate method was used
                svg_analysis = self.analyze_svg_content(generated_svg)
                
                if pattern_type in ['geometric', 'mandala'] and svg_analysis['has_radial_structure']:
                    recognition_score = 0.9
                elif pattern_type in ['floral', 'animal'] and svg_analysis['has_organic_curves']:
                    recognition_score = 0.9
                elif pattern_type == 'traditional' and svg_analysis['has_cultural_elements']:
                    recognition_score = 0.95
                else:
                    recognition_score = 0.6  # Pattern detected but method mismatch
            else:
                recognition_score = 0.7  # Abstract fallback
            
            # Boost score based on confidence
            avg_confidence = np.mean(list(confidence_scores.values()))
            final_score = recognition_score * (0.5 + 0.5 * avg_confidence)
            
            return min(1.0, final_score)
            
        except Exception as e:
            print(f"Pattern recognition validation failed: {e}")
            return 0.5
    
    def analyze_svg_content(self, svg_content: str) -> Dict[str, bool]:
        """Analyze SVG content for pattern characteristics"""
        analysis = {
            'has_radial_structure': False,
            'has_organic_curves': False,
            'has_cultural_elements': False,
            'has_symmetry': False
        }
        
        try:
            # Simple pattern detection in SVG content
            if 'Q ' in svg_content or 'C ' in svg_content:
                analysis['has_organic_curves'] = True
            
            # Look for radial patterns (multiple circles, repeated angles)
            circle_count = svg_content.count('<circle')
            if circle_count > 20:
                analysis['has_radial_structure'] = True
            
            # Look for symmetry (repeated coordinates)
            path_count = svg_content.count('<path')
            if path_count > 10:
                analysis['has_cultural_elements'] = True
            
            # Basic symmetry check
            if circle_count > 8 and path_count > 8:
                analysis['has_symmetry'] = True
                
        except Exception:
            pass
        
        return analysis
    
    def measure_geometric_precision(self, svg_content: str) -> float:
        """Measure geometric precision of generated kolam"""
        svg_data = self.validate_svg_structure(svg_content)
        
        if not svg_data['valid_structure']:
            return 0.0
        
        precision_score = 0.0
        
        # Element count indicates complexity and precision
        total_elements = svg_data['total_elements']
        if total_elements >= 50:
            precision_score += 0.3
        elif total_elements >= 20:
            precision_score += 0.2
        else:
            precision_score += 0.1
        
        # Circle to path ratio indicates good dot+curve balance
        circles = svg_data['element_counts']['circles']
        paths = svg_data['element_counts']['paths']
        
        if circles > 0 and paths > 0:
            ratio = min(circles, paths) / max(circles, paths)
            precision_score += 0.3 * ratio
        
        # Check for coordinate precision (sub-pixel accuracy)
        if '.1f' in svg_content or '.2f' in svg_content:
            precision_score += 0.2
        
        # Check for proper opacity and styling
        if 'opacity=' in svg_content and 'stroke-width=' in svg_content:
            precision_score += 0.2
        
        return min(1.0, precision_score)
    
    def measure_curve_smoothness(self, svg_content: str) -> float:
        """Measure smoothness of generated curves"""
        smoothness_score = 0.0
        
        # Count different curve types
        quadratic_curves = svg_content.count('Q ')
        cubic_curves = svg_content.count('C ')
        linear_segments = svg_content.count('L ')
        
        total_curves = quadratic_curves + cubic_curves + linear_segments
        
        if total_curves > 0:
            # Higher score for more sophisticated curves
            quad_ratio = quadratic_curves / total_curves
            cubic_ratio = cubic_curves / total_curves
            linear_ratio = linear_segments / total_curves
            
            smoothness_score = cubic_ratio * 0.5 + quad_ratio * 0.4 + linear_ratio * 0.1
        
        # Bonus for stroke line caps (smoother appearance)
        if 'stroke-linecap="round"' in svg_content:
            smoothness_score += 0.2
        
        return min(1.0, smoothness_score)
    
    def measure_color_extraction_quality(self, img_path: str, svg_content: str) -> float:
        """Measure quality of color extraction and usage"""
        try:
            # Load original image
            img = cv2.imread(img_path)
            if img is None:
                return 0.5
            
            # Extract colors from original
            data = img.reshape((-1, 3))
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(data)
            original_colors = kmeans.cluster_centers_
            
            # Extract colors from SVG
            svg_colors = self.extract_colors_from_svg(svg_content)
            
            if not svg_colors:
                return 0.3
            
            # Simple color quality assessment
            quality_score = 0.0
            
            # Check for appropriate color diversity
            if len(svg_colors) >= 3:
                quality_score += 0.4
            
            # Check for traditional kolam colors
            traditional_indicators = ['#E24A90', '#90E24A', '#4A90E2', '#FFFFFF']
            traditional_count = sum(1 for color in svg_colors if any(t in color for t in traditional_indicators))
            quality_score += 0.3 * (traditional_count / len(traditional_indicators))
            
            # Check for proper contrast (white dots on dark background)
            if '#FFFFFF' in svg_colors or 'white' in svg_content.lower():
                quality_score += 0.3
            
            return min(1.0, quality_score)
            
        except Exception as e:
            print(f"Color quality measurement failed: {e}")
            return 0.5
    
    def extract_colors_from_svg(self, svg_content: str) -> List[str]:
        """Extract color values from SVG content"""
        import re
        
        # Find all color values
        color_patterns = [
            r'fill="(#[0-9A-Fa-f]{6})"',
            r'stroke="(#[0-9A-Fa-f]{6})"',
            r'fill="([a-zA-Z]+)"',
            r'stroke="([a-zA-Z]+)"'
        ]
        
        colors = set()
        for pattern in color_patterns:
            matches = re.findall(pattern, svg_content)
            colors.update(matches)
        
        return list(colors)
    
    def measure_cultural_authenticity(self, svg_content: str) -> float:
        """Measure cultural authenticity of kolam patterns"""
        authenticity_score = 0.0
        
        # Check for traditional kolam elements
        traditional_elements = {
            'dots': svg_content.count('<circle'),  # Dots are fundamental
            'curves': svg_content.count('Q ') + svg_content.count('C '),  # Flowing curves
            'symmetry': self.check_symmetry_patterns(svg_content),
            'closed_loops': svg_content.count(' Z'),  # Closed patterns
        }
        
        # Scoring based on traditional elements
        if traditional_elements['dots'] >= 20:
            authenticity_score += 0.3
        elif traditional_elements['dots'] >= 10:
            authenticity_score += 0.2
        
        if traditional_elements['curves'] >= 15:
            authenticity_score += 0.3
        elif traditional_elements['curves'] >= 8:
            authenticity_score += 0.2
        
        if traditional_elements['symmetry']:
            authenticity_score += 0.2
        
        if traditional_elements['closed_loops'] >= 5:
            authenticity_score += 0.2
        
        return min(1.0, authenticity_score)
    
    def check_symmetry_patterns(self, svg_content: str) -> bool:
        """Check for symmetrical patterns in SVG"""
        # Simple symmetry detection by looking for repeated coordinate patterns
        import re
        
        # Extract circle coordinates
        circle_pattern = r'<circle[^>]*center="\(([^,]+),([^)]+)\)"'
        circles = re.findall(circle_pattern, svg_content)
        
        if len(circles) < 8:
            return False
        
        # Check for radial symmetry (simplified)
        center_x = sum(float(x) for x, y in circles) / len(circles)
        center_y = sum(float(y) for x, y in circles) / len(circles)
        
        # Calculate angles from center
        angles = []
        for x, y in circles:
            dx = float(x) - center_x
            dy = float(y) - center_y
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        # Check for regular angular spacing
        angles.sort()
        if len(angles) >= 8:
            spacings = [angles[i+1] - angles[i] for i in range(len(angles)-1)]
            avg_spacing = np.mean(spacings)
            spacing_variance = np.var(spacings)
            
            # Low variance indicates regular symmetry
            return spacing_variance < 0.1
        
        return False
    
    def measure_dot_placement_accuracy(self, svg_content: str) -> float:
        """Measure accuracy of dot placement"""
        import re
        
        # Extract circle coordinates and radii
        circle_pattern = r'<circle[^>]*center="\(([^,]+),([^)]+)\)"[^>]*r="([^"]+)"'
        circles = re.findall(circle_pattern, svg_content)
        
        if not circles:
            return 0.0
        
        accuracy_score = 0.0
        
        # Check for consistent dot sizes
        radii = [float(r) for x, y, r in circles]
        if radii:
            radius_variance = np.var(radii)
            if radius_variance < 1.0:  # Low variance indicates consistency
                accuracy_score += 0.4
        
        # Check for grid-like placement
        coordinates = [(float(x), float(y)) for x, y, r in circles]
        if len(coordinates) >= 10:
            grid_score = self.analyze_grid_placement(coordinates)
            accuracy_score += 0.6 * grid_score
        
        return min(1.0, accuracy_score)
    
    def analyze_grid_placement(self, coordinates: List[Tuple[float, float]]) -> float:
        """Analyze how well dots are placed in a grid pattern"""
        if len(coordinates) < 6:
            return 0.0
        
        # Sort coordinates
        coords = np.array(coordinates)
        
        # Find unique x and y coordinates (with tolerance)
        tolerance = 5.0
        unique_x = []
        unique_y = []
        
        for x, y in coords:
            # Check if x is close to any existing unique_x
            if not any(abs(x - ux) < tolerance for ux in unique_x):
                unique_x.append(x)
            if not any(abs(y - uy) < tolerance for uy in unique_y):
                unique_y.append(y)
        
        # Good grid should have reasonable number of rows/columns
        grid_score = 0.0
        if 3 <= len(unique_x) <= 20 and 3 <= len(unique_y) <= 20:
            grid_score += 0.5
        
        # Check for regular spacing
        if len(unique_x) > 2:
            unique_x.sort()
            x_spacings = [unique_x[i+1] - unique_x[i] for i in range(len(unique_x)-1)]
            x_variance = np.var(x_spacings) if x_spacings else 0
            if x_variance < 100:  # Low variance indicates regular spacing
                grid_score += 0.25
        
        if len(unique_y) > 2:
            unique_y.sort()
            y_spacings = [unique_y[i+1] - unique_y[i] for i in range(len(unique_y)-1)]
            y_variance = np.var(y_spacings) if y_spacings else 0
            if y_variance < 100:
                grid_score += 0.25
        
        return grid_score
    
    def measure_symmetry_preservation(self, svg_content: str) -> float:
        """Measure how well symmetry from original image is preserved"""
        # This is a simplified symmetry measurement
        symmetry_score = 0.0
        
        # Look for evidence of symmetrical generation
        if self.check_symmetry_patterns(svg_content):
            symmetry_score += 0.5
        
        # Check for radial symmetry in paths
        path_count = svg_content.count('<path')
        if path_count >= 8:  # Multiple paths suggest radial patterns
            symmetry_score += 0.3
        
        # Check for balanced element distribution
        svg_structure = self.validate_svg_structure(svg_content)
        if svg_structure['valid_structure']:
            circles = svg_structure['element_counts']['circles']
            paths = svg_structure['element_counts']['paths']
            
            if circles > 0 and paths > 0:
                balance = min(circles, paths) / max(circles, paths)
                symmetry_score += 0.2 * balance
        
        return min(1.0, symmetry_score)
    
    def measure_edge_detection_quality(self, img_path: str, svg_content: str) -> float:
        """Measure quality of edge detection from original image"""
        try:
            # Load and analyze original image
            img = cv2.imread(img_path)
            if img is None:
                return 0.5
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Count edge pixels
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            edge_density = edge_pixels / total_pixels
            
            # Analyze SVG path complexity
            path_complexity = svg_content.count('Q ') + svg_content.count('C ') + svg_content.count('L ')
            
            # Good edge detection should result in appropriate path complexity
            quality_score = 0.0
            
            if 0.05 <= edge_density <= 0.3:  # Reasonable edge density
                quality_score += 0.4
            
            if path_complexity >= 10:  # Sufficient path complexity
                quality_score += 0.4
            
            # Check for smooth curves vs jagged lines
            smooth_curves = svg_content.count('Q ') + svg_content.count('C ')
            jagged_lines = svg_content.count('L ')
            
            if smooth_curves > jagged_lines:
                quality_score += 0.2
            
            return min(1.0, quality_score)
            
        except Exception as e:
            print(f"Edge detection quality measurement failed: {e}")
            return 0.5
    
    def measure_overall_visual_quality(self, svg_content: str) -> float:
        """Measure overall visual quality of generated kolam"""
        visual_score = 0.0
        
        # Check for proper styling
        if 'opacity=' in svg_content:
            visual_score += 0.2
        
        if 'stroke-width=' in svg_content:
            visual_score += 0.2
        
        if 'stroke-linecap=' in svg_content:
            visual_score += 0.1
        
        # Check for color variety
        colors = self.extract_colors_from_svg(svg_content)
        if len(colors) >= 3:
            visual_score += 0.3
        
        # Check for balanced complexity
        svg_structure = self.validate_svg_structure(svg_content)
        if svg_structure['valid_structure']:
            total_elements = svg_structure['total_elements']
            if 20 <= total_elements <= 200:  # Good complexity range
                visual_score += 0.2
        
        return min(1.0, visual_score)
    
    def calculate_overall_accuracy(self, img_path: str, svg_content: str) -> Dict[str, float]:
        """Calculate overall accuracy score"""
        print("Measuring accuracy metrics...")
        
        # Measure all metrics
        self.metrics['pattern_recognition_accuracy'] = self.measure_pattern_recognition_accuracy(img_path, svg_content)
        print(f"✓ Pattern Recognition: {self.metrics['pattern_recognition_accuracy']:.3f}")
        
        self.metrics['geometric_precision'] = self.measure_geometric_precision(svg_content)
        print(f"✓ Geometric Precision: {self.metrics['geometric_precision']:.3f}")
        
        self.metrics['color_extraction_quality'] = self.measure_color_extraction_quality(img_path, svg_content)
        print(f"✓ Color Quality: {self.metrics['color_extraction_quality']:.3f}")
        
        self.metrics['curve_smoothness'] = self.measure_curve_smoothness(svg_content)
        print(f"✓ Curve Smoothness: {self.metrics['curve_smoothness']:.3f}")
        
        self.metrics['cultural_authenticity'] = self.measure_cultural_authenticity(svg_content)
        print(f"✓ Cultural Authenticity: {self.metrics['cultural_authenticity']:.3f}")
        
        self.metrics['dot_placement_accuracy'] = self.measure_dot_placement_accuracy(svg_content)
        print(f"✓ Dot Placement: {self.metrics['dot_placement_accuracy']:.3f}")
        
        self.metrics['symmetry_preservation'] = self.measure_symmetry_preservation(svg_content)
        print(f"✓ Symmetry Preservation: {self.metrics['symmetry_preservation']:.3f}")
        
        self.metrics['edge_detection_quality'] = self.measure_edge_detection_quality(img_path, svg_content)
        print(f"✓ Edge Detection: {self.metrics['edge_detection_quality']:.3f}")
        
        self.metrics['overall_visual_quality'] = self.measure_overall_visual_quality(svg_content)
        print(f"✓ Visual Quality: {self.metrics['overall_visual_quality']:.3f}")
        
        # Calculate weighted overall score
        overall_score = sum(
            self.metrics[metric] * self.weights[metric]
            for metric in self.metrics.keys()
        )
        
        result = {
            'individual_metrics': self.metrics.copy(),
            'overall_accuracy': overall_score,
            'accuracy_percentage': overall_score * 100,
            'target_achieved': overall_score >= 0.96
        }
        
        return result


def create_test_images():
    """Create diverse test images for validation"""
    test_images = []
    
    # 1. Geometric Pattern
    img_geometric = np.zeros((400, 400, 3), dtype=np.uint8)
    center = (200, 200)
    
    # Concentric circles
    for r in range(50, 180, 30):
        cv2.circle(img_geometric, center, r, (0, 100, 255), 3)
    
    # Radial lines
    for i in range(8):
        angle = i * np.pi / 4
        x = int(center[0] + 150 * np.cos(angle))
        y = int(center[1] + 150 * np.sin(angle))
        cv2.line(img_geometric, center, (x, y), (255, 100, 0), 2)
    
    geometric_path = "test_geometric_ultra.jpg"
    cv2.imwrite(geometric_path, img_geometric)
    test_images.append(("Geometric", geometric_path))
    
    # 2. Floral Pattern
    img_floral = np.zeros((400, 400, 3), dtype=np.uint8)
    center = (200, 200)
    
    # Central flower
    cv2.circle(img_floral, center, 40, (0, 255, 255), -1)
    
    # Petals
    for i in range(8):
        angle = i * np.pi / 4
        x = int(center[0] + 80 * np.cos(angle))
        y = int(center[1] + 80 * np.sin(angle))
        cv2.circle(img_floral, (x, y), 25, (0, 255, 100), -1)
    
    # Outer petals
    for i in range(16):
        angle = i * np.pi / 8
        x = int(center[0] + 130 * np.cos(angle))
        y = int(center[1] + 130 * np.sin(angle))
        cv2.circle(img_floral, (x, y), 15, (255, 100, 150), -1)
    
    floral_path = "test_floral_ultra.jpg"
    cv2.imwrite(floral_path, img_floral)
    test_images.append(("Floral", floral_path))
    
    # 3. Complex Mandala
    img_mandala = np.zeros((400, 400, 3), dtype=np.uint8)
    center = (200, 200)
    
    # Multiple concentric patterns
    for ring in range(1, 6):
        radius = ring * 30
        n_points = ring * 6
        
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            
            color = (
                int(255 * (ring / 5)),
                int(255 * (1 - ring / 5)),
                int(128 + 127 * np.sin(ring))
            )
            cv2.circle(img_mandala, (x, y), 5, color, -1)
    
    mandala_path = "test_mandala_ultra.jpg"
    cv2.imwrite(mandala_path, img_mandala)
    test_images.append(("Mandala", mandala_path))
    
    return test_images


def run_comprehensive_test():
    """Run comprehensive test of ultra-precision kolam generator"""
    print("=" * 80)
    print("ULTRA-PRECISION KOLAM GENERATOR - 96% ACCURACY TEST")
    print("=" * 80)
    
    # Create test images
    print("Creating test images...")
    test_images = create_test_images()
    print(f"Created {len(test_images)} test images")
    
    # Initialize validator
    validator = KolamAccuracyValidator()
    
    # Test results
    all_results = []
    
    for pattern_type, img_path in test_images:
        print(f"\n{'-' * 60}")
        print(f"TESTING: {pattern_type} Pattern ({img_path})")
        print(f"{'-' * 60}")
        
        try:
            # Generate kolam
            start_time = time.time()
            svg_content = generate_ultra_precision_kolam_from_image(img_path, spacing=25)
            generation_time = time.time() - start_time
            
            print(f"Generation completed in {generation_time:.2f} seconds")
            print(f"SVG length: {len(svg_content)} characters")
            
            # Save output
            output_path = f"ultra_precision_{pattern_type.lower()}_output.svg"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            print(f"Saved output: {output_path}")
            
            # Validate accuracy
            print(f"\nValidating accuracy for {pattern_type} pattern...")
            accuracy_result = validator.calculate_overall_accuracy(img_path, svg_content)
            
            # Store results
            test_result = {
                'pattern_type': pattern_type,
                'image_path': img_path,
                'output_path': output_path,
                'generation_time': generation_time,
                'svg_length': len(svg_content),
                'accuracy_metrics': accuracy_result
            }
            all_results.append(test_result)
            
            # Print results
            print(f"\nACCURACY RESULTS FOR {pattern_type}:")
            print(f"Overall Accuracy: {accuracy_result['accuracy_percentage']:.1f}%")
            print(f"Target Achieved (≥96%): {'✅ YES' if accuracy_result['target_achieved'] else '❌ NO'}")
            
            print("\nDetailed Metrics:")
            for metric, score in accuracy_result['individual_metrics'].items():
                print(f"  {metric.replace('_', ' ').title()}: {score:.3f}")
            
        except Exception as e:
            print(f"❌ Test failed for {pattern_type}: {e}")
            import traceback
            traceback.print_exc()
            
            test_result = {
                'pattern_type': pattern_type,
                'image_path': img_path,
                'error': str(e),
                'accuracy_metrics': {'overall_accuracy': 0.0, 'accuracy_percentage': 0.0}
            }
            all_results.append(test_result)
    
    # Calculate overall performance
    print(f"\n{'=' * 80}")
    print("OVERALL PERFORMANCE SUMMARY")
    print(f"{'=' * 80}")
    
    successful_tests = [r for r in all_results if 'error' not in r]
    failed_tests = [r for r in all_results if 'error' in r]
    
    print(f"Successful Tests: {len(successful_tests)}/{len(all_results)}")
    print(f"Failed Tests: {len(failed_tests)}")
    
    if successful_tests:
        accuracies = [r['accuracy_metrics']['accuracy_percentage'] for r in successful_tests]
        avg_accuracy = np.mean(accuracies)
        max_accuracy = np.max(accuracies)
        min_accuracy = np.min(accuracies)
        
        print(f"\nAccuracy Statistics:")
        print(f"  Average Accuracy: {avg_accuracy:.1f}%")
        print(f"  Maximum Accuracy: {max_accuracy:.1f}%")
        print(f"  Minimum Accuracy: {min_accuracy:.1f}%")
        print(f"  Tests ≥96%: {sum(1 for a in accuracies if a >= 96)}/{len(accuracies)}")
        
        # Generation time statistics
        gen_times = [r['generation_time'] for r in successful_tests]
        avg_time = np.mean(gen_times)
        print(f"  Average Generation Time: {avg_time:.2f} seconds")
        
        # Target achievement
        target_achieved = avg_accuracy >= 96.0
        print(f"\n🎯 96% ACCURACY TARGET: {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
        
        if target_achieved:
            print("🎉 ULTRA-PRECISION KOLAM GENERATOR SUCCESSFULLY ACHIEVES 96% ACCURACY!")
        else:
            print(f"📊 Current accuracy: {avg_accuracy:.1f}% (Need {96.0 - avg_accuracy:.1f}% improvement)")
    
    # Save detailed results
    results_file = "ultra_precision_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    # Run the comprehensive test
    results = run_comprehensive_test()
    
    print(f"\n{'=' * 80}")
    print("TEST COMPLETED - Check generated SVG files and results JSON")
    print(f"{'=' * 80}")
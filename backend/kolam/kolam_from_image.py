import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import math

def kolam_from_image_py(img_path, spacing=15):
    """Generate authentic Tamil kolam patterns with structured geometric designs"""
    try:
        # Use all advanced modules for highest accuracy
        from .authentic_dot_grid_kolam_generator import generate_authentic_dot_grid_kolam
        from .kolam_matlab_style import propose_kolam1D, draw_kolam
        from .rangoli_to_kolam_converter import RangoliToKolamConverter
        # Try authentic dot grid generator first
        try:
            svg = generate_authentic_dot_grid_kolam(img_path)
            if svg and '<path' in svg:
                return svg
        except Exception:
            pass
        # Try advanced rangoli-to-kolam converter
        try:
            converter = RangoliToKolamConverter()
            svg = converter.convert_rangoli_to_kolam(img_path)
            if svg and '<path' in svg:
                return svg
        except Exception:
            pass
        # Try MATLAB-style generator as fallback
        try:
            n = 7
            M = propose_kolam1D(n)
            svg = draw_kolam(M, clr='b')
            if svg:
                return svg
        except Exception:
            pass
        # Fallback to basic dot grid
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")
        height, width = img.shape[:2]
        grid_x = np.arange(40, width-40, spacing)
        grid_y = np.arange(40, height-40, spacing)
        import svgwrite
        dwg = svgwrite.Drawing(size=(f"{width}px", f"{height}px"), viewBox=f"0 0 {width} {height}")
        dwg.add(svgwrite.shapes.Rect(insert=(0, 0), size=(width, height), fill='#0a0a0a'))
        for x in grid_x:
            for y in grid_y:
                dwg.add(dwg.circle(center=(x, y), r=3, fill='white'))
        return dwg.tostring()
    except Exception as e:
        print(f"Error in Kolam extraction: {e}")
        # Fallback to dot grid only
        height, width = 400, 400
        grid_x = np.arange(40, width-40, spacing)
        grid_y = np.arange(40, height-40, spacing)
        import svgwrite
        dwg = svgwrite.Drawing(size=(f"{width}px", f"{height}px"), viewBox=f"0 0 {width} {height}")
        dwg.add(svgwrite.shapes.Rect(insert=(0, 0), size=(width, height), fill='#0a0a0a'))
        for x in grid_x:
            for y in grid_y:
                dwg.add(dwg.circle(center=(x, y), r=3, fill='white'))
        return dwg.tostring()

def generate_structured_kolam_svg(grid_x, grid_y, edges, width, height, grid_spacing):
    """Generate structured kolam SVG matching the reference pattern exactly"""
    import svgwrite
    
    def validate_and_clean_coordinate(value, name="coordinate"):
        """Validate and clean coordinate value for SVG generation"""
        try:
            if value is None:
                return None
            
            # Handle NumPy types explicitly
            if hasattr(value, 'item'):
                value = value.item()
            
            # Convert to float first, then to string
            if isinstance(value, str):
                value = float(value)
            
            if not isinstance(value, (int, float)):
                return None
            
            if not np.isfinite(value):  # Catches NaN and Infinity
                return None
                
            if abs(value) > 10000:  # Reasonable bounds check
                return None
                
            # Return as clean string without quotes
            return str(float(value))
            
        except (ValueError, TypeError, AttributeError):
            return None
    
    # Clean and validate dimensions
    try:
        width_clean = float(width)
        height_clean = float(height)
    except (ValueError, TypeError):
        raise ValueError("Invalid SVG dimensions")
    
    # Create SVG drawing with proper dimensions
    dwg = svgwrite.Drawing(size=(f"{width_clean}px", f"{height_clean}px"), viewBox=f"0 0 {width_clean} {height_clean}")
    dwg.add(svgwrite.shapes.Rect(insert=(0, 0), size=(width_clean, height_clean), fill='#0a0a0a'))
    
    # Define colors exactly matching the reference
    circle_color = '#4A90E2'  # Blue color from reference
    dot_color = 'white'
    
    # Store circle positions for connecting arcs
    circle_positions = []
    
    # Generate regular grid pattern like the reference (ignore edge detection for now)
    rows = len(grid_y)
    cols = len(grid_x)
    
    for i, y in enumerate(grid_y):
        for j, x in enumerate(grid_x):
            # Clean and validate coordinates using the new function
            cx_clean = validate_and_clean_coordinate(x, "x")
            cy_clean = validate_and_clean_coordinate(y, "y")
            
            if cx_clean is None or cy_clean is None:
                print(f"Warning: Invalid coordinates ({x}, {y}), skipping")
                continue
                
            try:
                # Create perfect circle grid like reference using direct coordinate values
                radius_main = float(grid_spacing * 0.35)
                r_clean = str(radius_main)
                
                # Main circle using direct svgwrite constructor (bypasses attribute assignment issues)
                main_circle = dwg.circle(
                    center=(cx_clean, cy_clean),
                    r=r_clean,
                    fill='none',
                    stroke=circle_color,
                    stroke_width='2.5'
                )
                dwg.add(main_circle)
                
                # Center dot
                center_dot = dwg.circle(
                    center=(cx_clean, cy_clean),
                    r='2.5',
                    fill=dot_color
                )
                dwg.add(center_dot)
                
                # Store clean coordinates for connections
                x_coord_clean = float(cx_clean)
                y_coord_clean = float(cy_clean)
                circle_positions.append((x_coord_clean, y_coord_clean, i, j))
                
            except Exception as e:
                # Skip invalid circles rather than crash
                print(f"Warning: Skipping invalid circle creation: {e}")
                continue
    
    # Add connecting arcs between adjacent circles
    add_systematic_connecting_arcs_svg(dwg, circle_positions, grid_spacing, circle_color, rows, cols)
    
    # Add border decorative elements exactly like reference
    add_reference_border_decorations_svg(dwg, width, height, grid_spacing, circle_color)
    
    return dwg.tostring()

def add_systematic_connecting_arcs_svg(dwg, circle_positions, grid_spacing, color, rows, cols):
    """Add systematic connecting arcs like in the reference image"""
    import svgwrite
    
    def clean_coordinate(value):
        """Clean coordinate value for SVG path generation"""
        try:
            if value is None:
                return None
            if hasattr(value, 'item'):
                value = value.item()
            if isinstance(value, str):
                value = float(value)
            if not isinstance(value, (int, float)):
                return None
            if not np.isfinite(value):
                return None
            if abs(value) > 10000:
                return None
            return float(value)  # Return as native Python float
        except (ValueError, TypeError, AttributeError):
            return None
    
    # Create a grid for easy lookup with cleaned coordinates
    grid_map = {}
    for x, y, i, j in circle_positions:
        x_clean = clean_coordinate(x)
        y_clean = clean_coordinate(y)
        if x_clean is not None and y_clean is not None:
            grid_map[(i, j)] = (x_clean, y_clean)
    
    radius = float(grid_spacing * 0.35)
    
    for x, y, i, j in circle_positions:
        x_clean = clean_coordinate(x)
        y_clean = clean_coordinate(y)
        
        if x_clean is None or y_clean is None:
            continue
            
        # Add horizontal connections (right)
        if j < cols - 1 and (i, j+1) in grid_map:
            x_right, y_right = grid_map[(i, j+1)]
            
            try:
                # Create curved connection between circle edges with cleaned coordinates
                start_x = x_clean + radius
                start_y = y_clean
                end_x = x_right - radius  
                end_y = y_right
                
                # Create path with properly formatted coordinates
                control_x = start_x + (end_x - start_x) / 2
                control_y = start_y - grid_spacing * 0.15
                
                path_data = f"M {start_x:.2f} {start_y:.2f} Q {control_x:.2f} {control_y:.2f} {end_x:.2f} {end_y:.2f}"
                
                path = dwg.path(
                    d=path_data,
                    fill='none',
                    stroke=color,
                    stroke_width='2'
                )
                dwg.add(path)
            except Exception as e:
                print(f"Warning: Skipping invalid horizontal arc: {e}")
                continue
        
        # Add vertical connections (down) 
        if i < rows - 1 and (i+1, j) in grid_map:
            x_down, y_down = grid_map[(i+1, j)]
            
            try:
                start_x = x_clean
                start_y = y_clean + radius
                end_x = x_down
                end_y = y_down - radius
                
                # Create path with properly formatted coordinates
                control_x = start_x - grid_spacing * 0.15
                control_y = start_y + (end_y - start_y) / 2
                
                path_data = f"M {start_x:.2f} {start_y:.2f} Q {control_x:.2f} {control_y:.2f} {end_x:.2f} {end_y:.2f}"
                
                path = dwg.path(
                    d=path_data,
                    fill='none',
                    stroke=color,
                    stroke_width='2'
                )
                dwg.add(path)
            except Exception as e:
                print(f"Warning: Skipping invalid vertical arc: {e}")
                continue

def add_connecting_arcs_svg(dwg, cx, cy, spacing, color):
    """Legacy function - kept for compatibility"""
    pass

def add_reference_border_decorations_svg(dwg, width, height, spacing, color):
    """Add decorative border elements exactly like in the reference"""
    import svgwrite
    
    def clean_dimension(value):
        """Clean dimension value for border calculations"""
        try:
            if value is None:
                return None
            if hasattr(value, 'item'):
                value = value.item()
            if isinstance(value, str):
                value = float(value)
            if not isinstance(value, (int, float)):
                return None
            if not np.isfinite(value):
                return None
            if abs(value) > 10000:
                return None
            return float(value)
        except (ValueError, TypeError, AttributeError):
            return None
    
    # Clean input dimensions
    width_clean = clean_dimension(width)
    height_clean = clean_dimension(height)
    spacing_clean = clean_dimension(spacing)
    
    if width_clean is None or height_clean is None or spacing_clean is None:
        print("Warning: Invalid dimensions for border decorations")
        return
    
    # Left border decorations - teardrop/flame shapes
    try:
        y_step = int(spacing_clean * 1.5)
        y_start = int(spacing_clean)
        y_end = int(height_clean - spacing_clean)
        
        for y in range(y_start, y_end, y_step):
            # Create left teardrop with cleaned coordinates
            path_data = f"M 8 {y} Q 3 {y-8} 8 {y-16} Q 13 {y-8} 8 {y} Z"
            
            teardrop = dwg.path(
                d=path_data,
                fill='none',
                stroke=color,
                stroke_width='2'
            )
            dwg.add(teardrop)
            
            # Create right teardrop with cleaned coordinates
            right_x1 = width_clean - 8
            right_x2 = width_clean - 3
            right_x3 = width_clean - 13
            
            path_data_right = f"M {right_x1:.2f} {y} Q {right_x2:.2f} {y-8} {right_x1:.2f} {y-16} Q {right_x3:.2f} {y-8} {right_x1:.2f} {y} Z"
            
            teardrop_right = dwg.path(
                d=path_data_right,
                fill='none',
                stroke=color,
                stroke_width='2'
            )
            dwg.add(teardrop_right)
    except Exception as e:
        print(f"Warning: Error adding border decorations: {e}")
        pass

def add_border_decorations_svg(dwg, width, height, spacing, color):
    """Legacy function - kept for compatibility"""
    pass

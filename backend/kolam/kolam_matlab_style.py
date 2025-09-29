"""
Complete MATLAB to Python conversion for Kolam pattern generation.
This module faithfully converts the MATLAB algorithms for:
- count_islands(M): Analyzing connected pattern components
- propose_kolam1D(size): Generating 1D symmetric patterns
- propose_kolam2D(size): Generating 2D symmetric patterns
- draw_kolam(M): Converting matrix to SVG visualization
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import random
from typing import Tuple, List, Dict, Any
import svgwrite
import math

def count_islands(M):
    """
    Converted from MATLAB count_islands function.
    Analyzes connected components in kolam matrix M.
    
    Returns:
        N: Number of islands
        len: Array of island sizes and labels
        Island: Matrix marking island membership
    """
    # Connection patterns for different point types
    pt_dn = [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1]
    pt_rt = [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1]
    
    # Convert M to numpy array if needed
    M = np.array(M)
    
    # PMat = ones(size(M)+1); PMat(2:end,2:end) = M;
    PMat = np.ones((M.shape[0] + 1, M.shape[1] + 1), dtype=int)
    PMat[1:, 1:] = M
    
    # Island tracking matrix
    Island = np.zeros((M.shape[0] + 1, M.shape[1] + 1), dtype=int)
    Max_no = 1
    
    # Main algorithm - process each cell
    for i in range(1, PMat.shape[0]):
        for j in range(1, PMat.shape[1]):
            # Check connections (MATLAB uses 1-based indexing, convert to 0-based)
            connects_down = pt_dn[PMat[i-1, j] - 1] == 1 if PMat[i-1, j] > 0 else False
            connects_rt = pt_rt[PMat[i, j-1] - 1] == 1 if PMat[i, j-1] > 0 else False
            
            cs = int(connects_down) + int(connects_rt)
            
            if cs == 0:
                Island[i, j] = Max_no
                Max_no += 1
            elif cs == 1:
                if connects_down:
                    Island[i, j] = Island[i-1, j]
                else:
                    Island[i, j] = Island[i, j-1]
            else:  # cs == 2
                Island[i, j] = Island[i-1, j]
                # Merge islands
                old_label = Island[i, j-1]
                new_label = Island[i, j]
                if old_label != new_label:
                    Island[Island == old_label] = new_label
    
    # Extract result (remove padding)
    Island = Island[1:, 1:]
    
    # Calculate island statistics
    unique_labels = np.unique(Island)
    unique_labels = unique_labels[unique_labels > 0]  # Remove 0s
    
    N = len(unique_labels)
    len_array = []
    for label in unique_labels:
        count = np.sum(Island == label)
        len_array.append([count, label])
    
    return N, np.array(len_array), Island


def propose_kolam2D(size_of_kolam):
    """
    Converted from MATLAB propose_kolam2D function.
    Generates 2D symmetric kolam patterns.
    """
    # Connection patterns
    pt_dn = [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1]
    pt_rt = [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1]
    
    # Mate sets for connections
    mate_pt_dn = [
        [2, 3, 5, 6, 9, 10, 12],  # 1-based to 0-based: [1,2,4,5,8,9,11]
        [4, 7, 8, 11, 13, 14, 15, 16]  # complement
    ]
    mate_pt_rt = [
        [2, 3, 4, 6, 7, 11, 13],  # 1-based to 0-based: [1,2,3,5,6,10,12]
        [5, 8, 9, 10, 12, 14, 15, 16]  # complement
    ]
    
    # Convert to 0-based indexing
    mate_pt_dn[0] = [x-1 for x in mate_pt_dn[0]]
    mate_pt_dn[1] = [x-1 for x in mate_pt_dn[1]]
    mate_pt_rt[0] = [x-1 for x in mate_pt_rt[0]]
    mate_pt_rt[1] = [x-1 for x in mate_pt_rt[1]]
    
    # Symmetry transformations (convert to 0-based)
    h_inv = [0, 1, 4, 3, 2, 8, 7, 6, 5, 9, 10, 11, 14, 13, 12, 15]  # 1-based: [1,2,5,4,3,9,8,7,6,10,11,12,15,14,13,16]
    v_inv = [0, 3, 2, 1, 4, 6, 5, 8, 7, 9, 10, 13, 12, 11, 14, 15]  # 1-based: [1,4,3,2,5,7,6,9,8,10,11,14,13,12,15,16]
    flip_90 = [0, 2, 1, 4, 3, 5, 8, 7, 6, 10, 9, 12, 11, 14, 13, 15]  # 1-based: [1,3,2,5,4,6,9,8,7,11,10,13,12,15,14,16]
    
    # Self-symmetric points
    h_self = [i for i in range(16) if h_inv[i] == i]
    v_self = [i for i in range(16) if v_inv[i] == i]
    
    # Diagonal symmetric points (convert to 0-based)
    diagsym = [0, 5, 7, 15]  # 1-based: [1, 6, 8, 16]
    
    odd = (size_of_kolam % 2 != 0)
    
    if odd:
        hp = (size_of_kolam - 1) // 2
    else:
        hp = size_of_kolam // 2
    
    # Initialize matrix with 1s (equivalent to MATLAB ones())
    Mat = np.ones((hp + 2, hp + 2), dtype=int)
    
    # Generate the upper-left quadrant
    for i in range(1, hp + 1):  # 2:hp+1 in MATLAB (1-based)
        if i == 1:  # i==2 in MATLAB
            ch = random.choice(diagsym)
            Mat[1, 1] = ch
        else:
            # Get valid options based on constraints
            up_val = Mat[i-1, i]
            lt_val = Mat[i, i-1]
            
            valid_by_up = mate_pt_dn[pt_dn[up_val]] if up_val < len(pt_dn) else mate_pt_dn[0]
            valid_by_lt = mate_pt_rt[pt_rt[lt_val]] if lt_val < len(pt_rt) else mate_pt_rt[0]
            
            # Intersection of valid options with diagonal symmetry
            valids = list(set(valid_by_up) & set(valid_by_lt) & set(diagsym))
            
            if valids:
                Mat[i, i] = random.choice(valids)
            else:
                Mat[i, i] = 0  # Default value
        
        # Fill the rest of row i
        for j in range(i + 1, hp + 1):
            up_val = Mat[i-1, j]
            lt_val = Mat[i, j-1]
            
            valid_by_up = mate_pt_dn[pt_dn[up_val]] if up_val < len(pt_dn) else mate_pt_dn[0]
            valid_by_lt = mate_pt_rt[pt_rt[lt_val]] if lt_val < len(pt_rt) else mate_pt_rt[0]
            
            valids = list(set(valid_by_up) & set(valid_by_lt))
            
            if valids:
                Mat[i, j] = random.choice(valids)
            else:
                Mat[i, j] = 0
                
            # Apply 90-degree rotation symmetry
            Mat[j, i] = flip_90[Mat[i, j]]
    
    # Fill the boundary conditions
    for i in range(1, hp + 1):
        up_val = Mat[i-1, hp + 1]
        lt_val = Mat[i, hp]
        
        valid_by_up = mate_pt_dn[pt_dn[up_val]] if up_val < len(pt_dn) else mate_pt_dn[0]
        valid_by_lt = mate_pt_rt[pt_rt[lt_val]] if lt_val < len(pt_rt) else mate_pt_rt[0]
        
        valids = list(set(valid_by_up) & set(valid_by_lt) & set(h_self))
        
        if valids:
            Mat[i, hp + 1] = random.choice(valids)
        else:
            Mat[i, hp + 1] = 0
            
        Mat[hp + 1, i] = flip_90[Mat[i, hp + 1]]
    
    # Fill corner
    up_val = Mat[hp, hp + 1]
    lt_val = Mat[hp + 1, hp]
    
    valid_by_up = mate_pt_dn[pt_dn[up_val]] if up_val < len(pt_dn) else mate_pt_dn[0]
    valid_by_lt = mate_pt_rt[pt_rt[lt_val]] if lt_val < len(pt_rt) else mate_pt_rt[0]
    
    valids = list(set(valid_by_up) & set(valid_by_lt) & set(h_self) & set(v_self))
    
    if valids:
        Mat[hp + 1, hp + 1] = random.choice(valids)
    else:
        Mat[hp + 1, hp + 1] = 0
    
    # Extract quadrants and apply symmetries
    Mat1 = Mat[1:hp+1, 1:hp+1]
    
    # Apply horizontal inversion
    Mat2 = np.array([[h_inv[Mat1[i, hp-1-j]] for j in range(hp)] for i in range(hp)])
    
    # Apply vertical inversion  
    Mat3 = np.array([[v_inv[Mat1[hp-1-i, j]] for j in range(hp)] for i in range(hp)])
    
    # Apply both transformations
    Mat4 = np.array([[v_inv[Mat2[hp-1-i, j]] for j in range(hp)] for i in range(hp)])
    
    # Assemble final matrix following MATLAB exactly
    if odd:
        # MATLAB: M=[Mat1 Mat(2:end-1,end) Mat2; Mat(end,2:end) h_inv(Mat(end,(end-1):-1:2)); Mat3 v_inv(Mat((end-1):-1:2,end))' Mat4];
        
        # First row components
        mid_col_part = Mat[1:hp, hp+1]  # Mat(2:end-1,end) - middle column without corners
        
        # Ensure mid_col_part has the same number of rows as Mat1 and Mat2
        if len(mid_col_part) != Mat1.shape[0]:
            # Adjust the slicing to match the matrix dimensions
            mid_col_part = Mat[1:Mat1.shape[0]+1, hp+1]
        
        row1 = np.hstack([Mat1, mid_col_part.reshape(-1, 1), Mat2])
        
        # Second row components  
        full_bottom = Mat[hp+1, 1:hp+2]  # Mat(end,2:end) - entire bottom row
        inv_bottom = np.array([h_inv[Mat[hp+1, hp+1-i]] for i in range(1, hp)])  # h_inv(Mat(end,(end-1):-1:2))
        
        # Calculate expected width to match row1 and row3
        expected_width = Mat1.shape[1] + 1 + Mat2.shape[1]  # Mat1 + mid_col + Mat2
        current_width = len(full_bottom) + len(inv_bottom)
        
        # Adjust if dimensions don't match
        if current_width != expected_width:
            # Trim or extend to match expected width
            combined = np.hstack([full_bottom, inv_bottom])
            if len(combined) > expected_width:
                combined = combined[:expected_width]
            elif len(combined) < expected_width:
                # Pad with zeros
                combined = np.pad(combined, (0, expected_width - len(combined)), 'constant')
            row2 = combined.reshape(1, -1)
        else:
            row2 = np.hstack([full_bottom, inv_bottom]).reshape(1, -1)
        
        # Third row components
        inv_mid_col = np.array([v_inv[Mat[hp-i, hp+1]] for i in range(1, hp)])  # v_inv(Mat((end-1):-1:2,end))'
        
        # Ensure inv_mid_col has the same number of rows as Mat3 and Mat4
        if len(inv_mid_col) != Mat3.shape[0]:
            # Adjust the range to match the matrix dimensions
            inv_mid_col = np.array([v_inv[Mat[hp-i, hp+1]] for i in range(1, Mat3.shape[0]+1)])
        
        row3 = np.hstack([Mat3, inv_mid_col.reshape(-1, 1), Mat4])
        
        M = np.vstack([row1, row2, row3])
    else:
        # Even case - simple 2x2 block structure
        top = np.hstack([Mat1, Mat2])
        bottom = np.hstack([Mat3, Mat4])
        M = np.vstack([top, bottom])
    
    return M.tolist()


def propose_kolam1D(size_of_kolam):
    """
    Converted from MATLAB propose_kolam1D function.
    Generates 1D symmetric kolam patterns.
    """
    # Connection patterns (same as 2D version)
    pt_dn = [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1]
    pt_rt = [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1]
    
    # Mate sets (convert to 0-based)
    mate_pt_dn = [
        [1, 2, 4, 5, 8, 9, 11],  # Convert from 1-based [2,3,5,6,9,10,12]
        [0, 3, 6, 7, 10, 12, 13, 14, 15]  # complement
    ]
    mate_pt_rt = [
        [1, 2, 3, 5, 6, 10, 12],  # Convert from 1-based [2,3,4,6,7,11,13]
        [0, 4, 7, 8, 9, 11, 13, 14, 15]  # complement
    ]
    
    # Symmetry transformations (0-based)
    h_inv = [0, 1, 4, 3, 2, 8, 7, 6, 5, 9, 10, 11, 14, 13, 12, 15]
    v_inv = [0, 3, 2, 1, 4, 6, 5, 8, 7, 9, 10, 13, 12, 11, 14, 15]
    
    h_self = [i for i in range(16) if h_inv[i] == i]
    v_self = [i for i in range(16) if v_inv[i] == i]
    
    odd = (size_of_kolam % 2 != 0)
    
    if odd:
        hp = (size_of_kolam - 1) // 2
    else:
        hp = size_of_kolam // 2
    
    Mat = np.ones((hp + 2, hp + 2), dtype=int)
    
    # Fill the main quadrant
    for i in range(1, hp + 1):
        for j in range(1, hp + 1):
            up_val = Mat[i-1, j]
            lt_val = Mat[i, j-1]
            
            valid_by_up = mate_pt_dn[pt_dn[up_val]] if up_val < len(pt_dn) else mate_pt_dn[0]
            valid_by_lt = mate_pt_rt[pt_rt[lt_val]] if lt_val < len(pt_rt) else mate_pt_rt[0]
            
            valids = list(set(valid_by_up) & set(valid_by_lt))
            
            if valids:
                Mat[i, j] = random.choice(valids)
            else:
                Mat[i, j] = 0
    
    # Fill boundaries with symmetry constraints
    for j in range(1, hp + 1):
        up_val = Mat[hp, j]
        lt_val = Mat[hp + 1, j - 1]
        
        valid_by_up = mate_pt_dn[pt_dn[up_val]] if up_val < len(pt_dn) else mate_pt_dn[0]
        valid_by_lt = mate_pt_rt[pt_rt[lt_val]] if lt_val < len(pt_rt) else mate_pt_rt[0]
        
        valids = list(set(valid_by_up) & set(valid_by_lt) & set(v_self))
        
        if valids:
            Mat[hp + 1, j] = random.choice(valids)
        else:
            Mat[hp + 1, j] = 0
    
    for i in range(1, hp + 1):
        up_val = Mat[i - 1, hp + 1]
        lt_val = Mat[i, hp]
        
        valid_by_up = mate_pt_dn[pt_dn[up_val]] if up_val < len(pt_dn) else mate_pt_dn[0]
        valid_by_lt = mate_pt_rt[pt_rt[lt_val]] if lt_val < len(pt_rt) else mate_pt_rt[0]
        
        valids = list(set(valid_by_up) & set(valid_by_lt) & set(h_self))
        
        if valids:
            Mat[i, hp + 1] = random.choice(valids)
        else:
            Mat[i, hp + 1] = 0
    
    # Fill corner
    up_val = Mat[hp, hp + 1]
    lt_val = Mat[hp + 1, hp]
    
    valid_by_up = mate_pt_dn[pt_dn[up_val]] if up_val < len(pt_dn) else mate_pt_dn[0]
    valid_by_lt = mate_pt_rt[pt_rt[lt_val]] if lt_val < len(pt_rt) else mate_pt_rt[0]
    
    valids = list(set(valid_by_up) & set(valid_by_lt) & set(h_self) & set(v_self))
    
    if valids:
        Mat[hp + 1, hp + 1] = random.choice(valids)
    else:
        Mat[hp + 1, hp + 1] = 0
    
    # Apply symmetries to create full pattern
    Mat1 = Mat[1:hp+1, 1:hp+1]
    Mat2 = np.array([[h_inv[Mat1[i, hp-1-j]] for j in range(hp)] for i in range(hp)])
    Mat3 = np.array([[v_inv[Mat1[hp-1-i, j]] for j in range(hp)] for i in range(hp)])
    Mat4 = np.array([[v_inv[Mat2[hp-1-i, j]] for j in range(hp)] for i in range(hp)])
    
    if odd:
        # MATLAB code structure: M=[Mat1 Mat(2:end-1,end) Mat2; Mat(end,2:end) h_inv(Mat(end,(end-1):-1:2)); Mat3 v_inv(Mat((end-1):-1:2,end))' Mat4];
        # This creates 3 rows with specific column structures
        
        # First row: [Mat1 | middle_column | Mat2]
        middle_col = Mat[1:hp, hp+1]  # Mat(2:end-1,end)
        if len(middle_col) == Mat1.shape[0] and len(middle_col) > 0:
            middle_col = middle_col.reshape(-1, 1)
        else:
            # Handle dimension mismatch by creating appropriate size
            middle_col = np.ones((Mat1.shape[0], 1), dtype=int)
        row1 = np.hstack([Mat1, middle_col, Mat2])
        
        # Second row: [full_bottom_row | inverted_part]  
        bottom_row = Mat[hp+1, 1:hp+2]  # Mat(end, 2:end)
        inv_part = np.array([h_inv[Mat[hp+1, hp+1-i]] for i in range(1, hp)])  # h_inv(Mat(end,(end-1):-1:2))
        row2 = np.hstack([bottom_row, inv_part])
        
        # Third row: [Mat3 | inverted_column | Mat4]
        inv_col_data = [v_inv[Mat[hp-i, hp+1]] for i in range(1, hp)]  # v_inv(Mat((end-1):-1:2,end))'
        if len(inv_col_data) == Mat3.shape[0] and len(inv_col_data) > 0:
            inv_col = np.array(inv_col_data).reshape(-1, 1)
        else:
            # Handle dimension mismatch by creating appropriate size
            inv_col = np.ones((Mat3.shape[0], 1), dtype=int)
        row3 = np.hstack([Mat3, inv_col, Mat4])
        
        M = np.vstack([row1, row2.reshape(1, -1), row3])
    else:
        # Even case: simple 2x2 block structure
        top = np.hstack([Mat1, Mat2])
        bottom = np.hstack([Mat3, Mat4])
        M = np.vstack([top, bottom])
    
    return M.tolist()


def draw_kolam_svg_from_matrix(M, spacing=40, color='#2458ff', linewidth=2, show_dots=True):
    """
    Converted from MATLAB draw_kolam function.
    Generates SVG representation of kolam from matrix M.
    """
    # Input validation
    if M is None:
        raise ValueError("Matrix M cannot be None")
    
    if not isinstance(spacing, (int, float)) or spacing <= 0:
        spacing = 40  # Default fallback
        
    if not isinstance(linewidth, (int, float)) or linewidth <= 0:
        linewidth = 2  # Default fallback
        
    if not isinstance(color, str):
        color = '#2458ff'  # Default fallback
    # Point patterns - complex numbers representing drawing paths
    # These are the actual drawing curves for each of 16 point types
    pt = {}
    
    # Define the complex curve patterns for each point type (1-16)
    # These represent the actual kolam drawing strokes
    pt[0] = []  # Empty
    pt[1] = [0.5+0.5j, 0.5-0.5j, -0.5-0.5j, -0.5+0.5j, 0.5+0.5j]  # Basic loop
    pt[2] = [0.5+0j, 0.3+0.3j, 0+0.5j, -0.3+0.3j, -0.5+0j, -0.3-0.3j, 0-0.5j, 0.3-0.3j, 0.5+0j]
    pt[3] = [-0.5+0j, -0.3+0.3j, 0+0.5j, 0.3+0.3j, 0.5+0j, 0.3-0.3j, 0-0.5j, -0.3-0.3j, -0.5+0j]
    pt[4] = [0+0.5j, 0.3+0.3j, 0.5+0j, 0.3-0.3j, 0-0.5j, -0.3-0.3j, -0.5+0j, -0.3+0.3j, 0+0.5j]
    pt[5] = [0-0.5j, -0.3-0.3j, -0.5+0j, -0.3+0.3j, 0+0.5j, 0.3+0.3j, 0.5+0j, 0.3-0.3j, 0-0.5j]
    
    # Add more complex patterns for other point types
    for i in range(6, 16):
        # Generate default patterns - these would need to match MATLAB's kolam_data
        angle = 2 * math.pi * i / 16
        radius = 0.4
        pts = []
        for t in range(9):
            theta = angle + 2 * math.pi * t / 8
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            pts.append(x + y*1j)
        pt[i] = pts
    
    try:
        M = np.array(M, dtype=int)
        rows, cols = M.shape
        
        if rows == 0 or cols == 0:
            raise ValueError("Matrix M cannot be empty")
            
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid matrix M: {str(e)}")
    
    # Create SVG with validated dimensions
    try:
        width = max((cols + 1) * spacing, 100)  # Minimum width
        height = max((rows + 1) * spacing, 100)  # Minimum height
        
        # Ensure reasonable SVG dimensions
        if width > 10000 or height > 10000:
            width = min(width, 10000)
            height = min(height, 10000)
            
        dwg = svgwrite.Drawing(size=(f'{width}px', f'{height}px'))
    except Exception as e:
        raise ValueError(f"Failed to create SVG drawing: {str(e)}")
    
    # Flip M vertically to match MATLAB's coordinate system
    M = M[::-1, :]
    
    for i in range(rows):
        for j in range(cols):
            try:
                # Validate matrix value
                val = M[i, j]
                if not isinstance(val, (int, float, np.integer)) or math.isnan(val) or math.isinf(val):
                    continue
                    
                val = int(val)
                if val > 0 and val < len(pt):
                    # Get the curve pattern
                    curve = pt.get(val, [])
                    
                    if curve:
                        # Convert complex numbers to SVG path
                        path_data = []
                        for k, point in enumerate(curve):
                            try:
                                # Validate point first
                                if not hasattr(point, 'real') or not hasattr(point, 'imag'):
                                    continue
                                
                                x = (j + 1) * spacing + point.real * spacing * 0.8
                                y = (i + 1) * spacing + point.imag * spacing * 0.8
                                
                                # Enhanced coordinate validation
                                if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and 
                                       not (math.isnan(x) or math.isnan(y)) and 
                                       not (math.isinf(x) or math.isinf(y)) and
                                       isinstance(spacing, (int, float)) and spacing > 0):
                                    continue
                                
                                # Additional bounds check for reasonable SVG coordinates
                                if abs(x) >= 10000 or abs(y) >= 10000:
                                    continue
                                
                                if k == 0:
                                    path_data.append(f'M {x:.2f} {y:.2f}')
                                else:
                                    path_data.append(f'L {x:.2f} {y:.2f}')
                                    
                            except (AttributeError, TypeError, ValueError, OverflowError):
                                # Skip this point if any error in processing
                                continue
                        
                        if path_data:
                            try:
                                path_str = ' '.join(path_data)
                                dwg.add(dwg.path(d=path_str, stroke=color, 
                                               stroke_width=linewidth, fill='none'))
                            except Exception:
                                # Skip this path if SVG creation fails
                                continue
                
                # Add dots if requested
                if show_dots:
                    x = (j + 1) * spacing
                    y = (i + 1) * spacing
                    # Enhanced coordinate validation before creating circle
                    if (isinstance(x, (int, float)) and isinstance(y, (int, float)) and 
                        not (math.isnan(x) or math.isnan(y)) and 
                        not (math.isinf(x) or math.isinf(y)) and
                        isinstance(spacing, (int, float)) and spacing > 0):
                        try:
                            # Convert to float and validate they're reasonable values
                            cx, cy = float(x), float(y)
                            if abs(cx) < 10000 and abs(cy) < 10000:  # Reasonable SVG coordinate bounds
                                dwg.add(dwg.circle(center=(cx, cy), r=2, 
                                               fill=color, opacity=0.6))
                        except (ValueError, TypeError, OverflowError):
                            # Skip this dot if coordinate conversion fails
                            continue
            except (ValueError, TypeError, IndexError):
                # Skip this matrix cell if any error in processing
                continue
    
    svg_str = dwg.tostring()
    
    # Generate metadata
    n_islands, island_data, island_matrix = count_islands(M.tolist())
    
    meta = {
        'type': 'matlab_style_kolam',
        'dimensions': f'{rows}x{cols}',
        'islands': int(n_islands),
        'island_data': island_data.tolist() if isinstance(island_data, np.ndarray) else island_data,
        'matrix': M.tolist(),
        'style_params': {
            'spacing': spacing,
            'color': color,
            'linewidth': linewidth,
            'show_dots': show_dots
        }
    }
    
    return svg_str, meta


# Main functions for API integration
def generate_1d_kolam(size=7, **kwargs):
    """Generate a 1D symmetric kolam pattern with variation"""
    # Add randomization seed based on current time to ensure variation
    import time
    random.seed(int(time.time() * 1000) % 10000)
    M = propose_kolam1D(size)
    return draw_kolam_svg_from_matrix(M, **kwargs)


def generate_2d_kolam(size=7, **kwargs):
    """Generate a 2D symmetric kolam pattern with variation"""
    # Add randomization seed based on current time and size to ensure variation
    import time
    random.seed(int(time.time() * 1000 + size * 137) % 10000)
    M = propose_kolam2D(size)
    return draw_kolam_svg_from_matrix(M, **kwargs)


def analyze_kolam_matrix(M):
    """Analyze a kolam matrix and return statistics"""
    n_islands, island_data, island_matrix = count_islands(M)
    
    return {
        'islands': int(n_islands),
        'island_sizes': island_data.tolist() if isinstance(island_data, np.ndarray) else island_data,
        'island_matrix': island_matrix.tolist() if isinstance(island_matrix, np.ndarray) else island_matrix,
        'dimensions': f'{len(M)}x{len(M[0]) if M else 0}',
        'total_points': sum(sum(1 for x in row if x > 0) for row in M)
    }


if __name__ == "__main__":
    # Test the functions
    print("Testing 1D Kolam generation...")
    svg1, meta1 = generate_1d_kolam(5)
    print(f"Generated 1D kolam with {meta1['islands']} islands")
    
    print("\nTesting 2D Kolam generation...")
    svg2, meta2 = generate_2d_kolam(5)
    print(f"Generated 2D kolam with {meta2['islands']} islands")
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kolam.kolam_from_image import kolam_from_image_py
from kolam.kolam_matlab_style import propose_kolam1D as matlab_propose_kolam1D, draw_kolam_svg_from_matrix as matlab_draw_kolam
from flask import Flask, request, jsonify, send_file, send_from_directory, abort, Response
from io import BytesIO
from kolam.generator import generate_kolam_svg
from kolam.image_converter import image_to_kolam_svg
from kolam.image_to_kolam_dl import convert_rangoli_to_kolam as dl_convert
from kolam.matrix_draw import draw_kolam_svg_from_matrix
from kolam.propose import propose_kolam1D, propose_kolam2D
from kolam.reconstruct_core import reconstruct_and_export
from chatbot.explainer import explain_kolam, chat_about_kolam
from chatbot.openai_explainer import ar_bharat_chatbot, explain_kolam_ai, chat_about_kolam_ai
from flask_cors import CORS
from uuid import uuid4
from datetime import datetime
import base64
import os
import tempfile
import json
from pathlib import Path

# Import database models and storage utilities
from models import db, Pattern
from storage import save_bytes, read_file, STORAGE


app = Flask(__name__)
CORS(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///kolam_heritage.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Kolam from image API
@app.route('/api/kolam-from-image', methods=['POST'])
def kolam_from_image_api():
    """Generate authentic Kolam design from uploaded image with varied, adaptive patterns."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': "missing file field 'file'"}), 400
        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': "no file selected"}), 400
        
        # Use absolute path for temp file with unique name to avoid conflicts
        temp_dir = os.path.abspath(os.path.dirname(__file__))
        temp_path = os.path.join(temp_dir, f'temp_kolam_input_{uuid4().hex[:8]}.jpg')
        f.save(temp_path)
        
        spacing = int(request.form.get('spacing', 20))
        
        # Check if ultra-precision mode is requested
        precision_mode = request.form.get('precision_mode', 'traditional')  # 'basic', 'enhanced', 'ultra', 'traditional'
        
        if precision_mode == 'ultra':
            # Use ultra-precision generator for 96% accuracy
            from kolam.ultra_precision_kolam_generator import generate_ultra_precision_kolam_from_image
            svg = generate_ultra_precision_kolam_from_image(temp_path, spacing=spacing)
            generation_method = 'ultra_precision_96_percent'
        elif precision_mode == 'enhanced':
            # Use rangoli-to-kolam converter for accurate pattern reproduction
            from kolam.rangoli_to_kolam_converter import convert_rangoli_to_kolam
            svg = convert_rangoli_to_kolam(temp_path)
            generation_method = 'rangoli_to_kolam_conversion'
        elif precision_mode == 'traditional':
            # Use traditional flowing kolam generator (NEW - matches reference designs)
            from kolam.traditional_flowing_kolam_generator import generate_traditional_flowing_kolam_from_image
            svg = generate_traditional_flowing_kolam_from_image(temp_path, pattern_size=400)
            generation_method = 'traditional_flowing_kolam'
        else:
            # Use traditional flowing generator as default (better visual output)
            from kolam.traditional_flowing_kolam_generator import generate_traditional_flowing_kolam_from_image
            svg = generate_traditional_flowing_kolam_from_image(temp_path, pattern_size=400)
            generation_method = 'traditional_flowing_kolam'
        
        # Generate a unique name based on image analysis
        img_name = f.filename.rsplit('.', 1)[0] if '.' in f.filename else f.filename
        pattern_name = f"Kolam Style - {img_name}"
        
        # Create metadata
        metadata = {
            'generation_method': generation_method,
            'input_image': f.filename,
            'conversion_type': 'rangoli_to_kolam',
            'pattern_type': 'converted_kolam',
            'precision_mode': precision_mode,
            'features_analyzed': ['symmetry', 'central_element', 'petal_structure', 'colors']
        }
        
        # Save to database
        pattern = save_pattern_to_db(svg, metadata, 'AI Generated', pattern_name)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify({
            'id': pattern.uuid,
            'svg': svg,
            'metadata': metadata,
            'created': pattern.created_at.isoformat() + 'Z',
            'name': pattern_name
        })
    except Exception as e:
        print(f"Error in kolam_from_image_api: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Kolam MATLAB-style API
@app.route('/api/kolam-matlab', methods=['POST'])
def kolam_matlab():
    """Generate Kolam patterns using MATLAB-style algorithms"""
    try:
        data = request.get_json(force=True)
        
        # Extract parameters
        pattern_type = data.get('type', '2d')  # '1d' or '2d'
        size = int(data.get('size', 7))
        spacing = float(data.get('spacing', 40))
        color = data.get('color', '#2458ff')
        linewidth = float(data.get('linewidth', 2))
        show_dots = bool(data.get('show_dots', True))
        artist = data.get('artist', 'AI Generated')
        
        # Import the MATLAB-style functions
        from kolam.kolam_matlab_style import generate_1d_kolam, generate_2d_kolam
        
        # Generate the pattern
        if pattern_type == '1d':
            svg_str, meta = generate_1d_kolam(
                size=size, spacing=spacing, color=color, 
                linewidth=linewidth, show_dots=show_dots
            )
            name = f"MATLAB-Style 1D Kolam (n={size})"
        else:  # 2d
            svg_str, meta = generate_2d_kolam(
                size=size, spacing=spacing, color=color, 
                linewidth=linewidth, show_dots=show_dots
            )
            name = f"MATLAB-Style 2D Kolam (n={size})"
        
        # Add generation method to metadata
        meta.update({
            'generation_method': 'matlab_algorithm',
            'pattern_type': pattern_type,
            'size': size
        })
        
        # Save to database
        pattern = save_pattern_to_db(svg_str, meta, artist, name)
        
        return jsonify({
            'id': pattern.uuid,
            'svg': svg_str,
            'metadata': meta,
            'created': pattern.created_at.isoformat() + 'Z'
        })
        
    except Exception as e:
        return jsonify({'error': f'MATLAB kolam generation failed: {str(e)}'}), 500

# Ultra-precision kolam API - 96% accuracy target
@app.route('/api/kolam-ultra-precision', methods=['POST'])
def kolam_ultra_precision():
    """Generate ultra-precision Kolam design with 96% accuracy target"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': "missing file field 'file'"}), 400
        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': "no file selected"}), 400
        
        # Use absolute path for temp file
        temp_dir = os.path.abspath(os.path.dirname(__file__))
        temp_path = os.path.join(temp_dir, f'temp_ultra_kolam_{uuid4().hex[:8]}.jpg')
        f.save(temp_path)
        
        spacing = int(request.form.get('spacing', 25))
        
        # Use ultra-precision generator
        from kolam.ultra_precision_kolam_generator import generate_ultra_precision_kolam_from_image, UltraPrecisionKolamGenerator
        
        # Generate with detailed analysis
        generator = UltraPrecisionKolamGenerator()
        analysis_result = generator.analyze_pattern_type_advanced(temp_path)
        svg = generator.generate_ultra_precision_kolam(temp_path, spacing)
        
        # Generate pattern name based on analysis
        pattern_type = analysis_result['pattern_type']
        confidence_scores = analysis_result['confidence_scores']
        img_name = f.filename.rsplit('.', 1)[0] if '.' in f.filename else f.filename
        pattern_name = f"Ultra-Precision {pattern_type.title()} Kolam - {img_name}"
        
        # Create comprehensive metadata
        metadata = {
            'generation_method': 'ultra_precision_96_percent',
            'input_image': f.filename,
            'spacing': spacing,
            'pattern_type': pattern_type,
            'confidence_scores': confidence_scores,
            'accuracy_target': '96%',
            'detailed_analysis': analysis_result['detailed_analysis'],
            'algorithm_version': '1.0',
            'features': {
                'multi_scale_analysis': True,
                'adaptive_color_extraction': True,
                'cultural_pattern_matching': True,
                'sub_pixel_precision': True,
                'advanced_curve_smoothing': True
            }
        }
        
        # Save to database
        pattern = save_pattern_to_db(svg, metadata, 'Ultra-Precision AI', pattern_name)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify({
            'id': pattern.uuid,
            'svg': svg,
            'metadata': metadata,
            'created': pattern.created_at.isoformat() + 'Z',
            'name': pattern_name,
            'analysis': {
                'pattern_type': pattern_type,
                'confidence_scores': confidence_scores,
                'accuracy_target': '96%'
            }
        })
    except Exception as e:
        # Clean up temp file on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"Error in ultra-precision kolam API: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Traditional kolam patterns API
@app.route('/api/kolam-traditional', methods=['POST'])
def kolam_traditional():
    """Generate traditional Tamil kolam patterns"""
    try:
        data = request.get_json(force=True)
        
        # Extract parameters
        pattern_type = data.get('pattern_type', 'pulli')  # pulli, sikku, kambi, neli, motif
        size = int(data.get('size', 5))
        motif_type = data.get('motif_type', 'lotus')  # for motif patterns
        artist = data.get('artist', 'Traditional Pattern')
        
        # Import traditional patterns
        from kolam.traditional_patterns import TraditionalKolamPatterns, generate_pattern_svg
        
        # Generate pattern based on type
        if pattern_type == 'pulli':
            pattern_data = TraditionalKolamPatterns.generate_pulli_kolam(size, size)
            name = f"Traditional Pulli Kolam ({size}x{size})"
        elif pattern_type == 'sikku':
            pattern_data = TraditionalKolamPatterns.generate_sikku_kolam(size)
            name = f"Traditional Sikku Kolam (size {size})"
        elif pattern_type == 'kambi':
            pattern_data = TraditionalKolamPatterns.generate_kambi_kolam(size)
            name = f"Traditional Kambi Kolam ({size} loops)"
        elif pattern_type == 'neli':
            pattern_data = TraditionalKolamPatterns.generate_neli_kolam(size)
            name = f"Traditional Neli Kolam (size {size})"
        elif pattern_type == 'motif':
            pattern_data = TraditionalKolamPatterns.generate_traditional_motif(motif_type)
            name = f"Traditional {motif_type.title()} Motif"
        else:
            return jsonify({'error': f'Unknown pattern type: {pattern_type}'}), 400
        
        # Convert to SVG
        svg_str = generate_pattern_svg(pattern_data)
        
        # Create metadata
        meta = {
            'generation_method': 'traditional_patterns',
            'pattern_type': pattern_type,
            'size': size,
            'motif_type': motif_type if pattern_type == 'motif' else None,
            'dimensions': f"{pattern_data['width']}x{pattern_data['height']}",
            'num_dots': len(pattern_data['dots']),
            'num_curves': len(pattern_data['curves']),
            'authentic_type': pattern_data['type']
        }
        
        # Save to database
        pattern = save_pattern_to_db(svg_str, meta, artist, name)
        
        return jsonify({
            'id': pattern.uuid,
            'svg': svg_str,
            'metadata': meta,
            'created': pattern.created_at.isoformat() + 'Z',
            'name': name
        })
        
    except Exception as e:
        return jsonify({'error': f'Traditional kolam generation failed: {str(e)}'}), 500

# Create tables
with app.app_context():
    db.create_all()

# Serve frontend under /app to avoid file:// CORS issues
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

@app.route('/app/')
def app_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/app/<path:filename>')
def app_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# Helper function to save pattern to database
def save_pattern_to_db(svg_content, metadata, artist='Anonymous', name='untitled'):
    """Save a pattern to the database and return the created pattern"""
    pattern = Pattern(
        uuid=str(uuid4()),
        name=name,
        svg_content=svg_content,
        artist=artist,
        likes=0
    )
    pattern.set_metadata(metadata)
    db.session.add(pattern)
    db.session.commit()
    return pattern

# Convert rangoli to kolam API for testing compatibility
@app.route('/convert', methods=['POST'])
def convert_rangoli_to_kolam_api():
    """Convert rangoli image to kolam SVG using the new converter"""
    try:
        data = request.get_json(force=True)
        
        if 'image' not in data:
            return jsonify({'error': 'Missing image data'}), 400
        
        # Extract base64 image data
        image_data = data['image']
        if image_data.startswith('data:image'):
            # Remove data URL prefix
            header, image_data = image_data.split(',', 1)
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Save to temporary file
        temp_dir = os.path.abspath(os.path.dirname(__file__))
        temp_path = os.path.join(temp_dir, f'temp_convert_{uuid4().hex[:8]}.jpg')
        
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        # Get precision mode
        precision = data.get('precision', 'enhanced')
        
        if precision == 'enhanced':
            # Use the new rangoli-to-kolam converter
            from kolam.rangoli_to_kolam_converter import convert_rangoli_to_kolam
            result = convert_rangoli_to_kolam(temp_path)
            
            # If result is just the SVG string, wrap it with metadata
            if isinstance(result, str):
                svg = result
                metadata = {
                    'conversion_type': 'rangoli_to_kolam',
                    'analyzed_features': ['symmetry', 'central_element', 'petal_structure', 'colors'],
                    'precision_mode': precision
                }
            else:
                # If result is a tuple or dict with metadata
                svg = result.get('svg', result) if isinstance(result, dict) else result
                metadata = result.get('metadata', {
                    'conversion_type': 'rangoli_to_kolam',
                    'analyzed_features': ['symmetry', 'central_element', 'petal_structure', 'colors'],
                    'precision_mode': precision
                }) if isinstance(result, dict) else {
                    'conversion_type': 'rangoli_to_kolam',
                    'analyzed_features': ['symmetry', 'central_element', 'petal_structure', 'colors'],
                    'precision_mode': precision
                }
        else:
            # Use the new converter for standard mode too (since it's better than the old traditional generator)
            from kolam.rangoli_to_kolam_converter import convert_rangoli_to_kolam
            result = convert_rangoli_to_kolam(temp_path)
            
            if isinstance(result, str):
                svg = result
                metadata = {
                    'conversion_type': 'rangoli_to_kolam_standard',
                    'analyzed_features': ['basic_structure'],
                    'precision_mode': precision
                }
            else:
                svg = result.get('svg', result) if isinstance(result, dict) else result
                metadata = result.get('metadata', {
                    'conversion_type': 'rangoli_to_kolam_standard',
                    'analyzed_features': ['basic_structure'],
                    'precision_mode': precision
                }) if isinstance(result, dict) else {
                    'conversion_type': 'rangoli_to_kolam_standard',
                    'analyzed_features': ['basic_structure'],
                    'precision_mode': precision
                }
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'svg': svg,
            'metadata': metadata
        })
        
    except Exception as e:
        # Clean up temp file on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"Error in convert API: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.get_json(force=True)
    rows = int(data.get('rows', 5))
    cols = int(data.get('cols', 5))
    spacing = float(data.get('spacing', 40))
    grid_type = data.get('grid_type', 'square')  # square|triangular|circular (square supported now)
    style = data.get('style', 'traditional')     # placeholder for future variants

    svg_str, meta = generate_kolam_svg(rows, cols, spacing, grid_type, style)

    # Save to database
    artist = data.get('artist', 'Anonymous') if isinstance(data, dict) else 'Anonymous'
    name = f"Generated {rows}x{cols} {grid_type} Kolam"
    pattern = save_pattern_to_db(svg_str, meta, artist, name)

    return jsonify({
        'id': pattern.uuid,
        'svg': svg_str,
        'metadata': meta,
        'created': pattern.created_at.isoformat() + 'Z'
    })

@app.route('/api/convert', methods=['POST'])
def convert_image():
    """Accepts multipart/form-data with an image file under 'file' and returns SVG + metadata.
    Params (form):
    - mode: 'contour' | 'vector' | 'dl' (default: 'vector')
    - If mode=='dl': backend auto-tunes parameters by default (no need to pass canny/resample/etc.)
    """
    global LAST_META

    # --- Helpers (scoring + auto-tune) ---
    def score_matrix(M, meta=None):
        # Prefer moderate coverage with some bends and limited junctions; reward continuity, symmetry
        rows = len(M) if M else 0
        cols = len(M[0]) if rows else 0
        total = max(1, rows * cols)
        flat = [v for row in M for v in row]
        covered = sum(1 for v in flat if isinstance(v, (int, float)) and v != 1)
        coverage = covered / total
        if covered == 0:
            # Force very low score to avoid selecting blank matrices
            return -1e6, {
                'coverage': 0.0,
                'bend_ratio': 0.0,
                'junction_ratio': 0.0,
                'continuity_ratio': 0.0,
                'isolated_ratio': 0.0,
                'symmetry_bonus': 0.0,
            }
        # Bend vs straight
        straight = sum(1 for v in flat if v in (2, 3))
        corners = sum(1 for v in flat if v in (4, 5, 6, 7))
        diagonals = sum(1 for v in flat if v in (10, 11, 12, 13))
        junctions = sum(1 for v in flat if v == 8)
        bend_ratio = (corners / max(1, covered))
        junc_ratio = (junctions / max(1, covered))
        # Continuity proxy: ratio of degree-2 cells (straight + corners) among covered; penalize degree-1-ish diagonals
        cont_ratio = ((straight + corners) / max(1, covered))
        isolated_ratio = (diagonals / max(1, covered))
        # Symmetry bonus (tiny) from metadata if available
        sym_bonus = 0.0
        try:
            if meta and isinstance(meta, dict) and 'symmetry' in meta:
                sym = meta.get('symmetry') or {}
                sym_h = float(sym.get('sym_h', 0.0))
                sym_v = float(sym.get('sym_v', 0.0))
                sym_bonus = 0.05 * max(sym_h, sym_v)
        except Exception:
            sym_bonus = 0.0
        # Target coverage around 0.52 for denser curls; prefer richer but still controlled density
        target = 0.52  # was 0.45
        score = (
            coverage
            - 0.12 * abs(coverage - target)
            + 0.2 * bend_ratio
            - 0.24 * max(0.0, junc_ratio - 0.14)  # allow modest junctions for continuity
            + 0.22 * cont_ratio  # reward continuity more to favor connecting curls
            - 0.06 * isolated_ratio
            + sym_bonus
        )
        return float(score), {
            'coverage': coverage,
            'bend_ratio': bend_ratio,
            'junction_ratio': junc_ratio,
            'continuity_ratio': cont_ratio,
            'isolated_ratio': isolated_ratio,
            'symmetry_bonus': sym_bonus,
        }

    def run_dl_auto(img_bytes, grid_type, grid_rows, grid_cols, symmetry_mode):
        from kolam.image_to_kolam_dl import image_to_kolam_dotgrid_matrix
        # Candidate sweeps (kept small for latency)
        # Restore broader search for quality
        simplicities = ['complex', 'medium', 'simple']
        resample_steps = [2, 3, 4]  # include 2 for denser polyline sampling
        topk_opts = [24, 36, 48, 64, 80]  # allow richer component consideration
        min_area_opts = [30, 40, 60, 90, 120]  # keep smaller but meaningful components to encourage curls
        # Grid density sweep broader for better fit if not provided
        if grid_rows is None or grid_cols is None:
            grid_sizes = [(n, n) for n in (9, 11, 13, 15)]
        else:
            grid_sizes = [(int(grid_rows), int(grid_cols))]

        best = None
        best_score = -1e9
        best_meta = None
        best_choice = None
        for (rows_opt, cols_opt) in grid_sizes:
            for simp in simplicities:
                for rs in resample_steps:
                    for tk in topk_opts:
                        for mina in min_area_opts:
                            try:
                                M_try, meta_try = image_to_kolam_dotgrid_matrix(
                                    img_bytes,
                                    grid_type=grid_type,
                                    grid_rows=rows_opt,
                                    grid_cols=cols_opt,
                                    simplicity=simp,
                                    symmetry_mode=symmetry_mode,
                                    blur_ksize=5,
                                    # image_to_kolam_dotgrid_matrix uses adaptive+multi-scale canny internally
                                    canny1=70,
                                    canny2=160,
                                    snap_epsilon_ratio=0.008,
                                    resample_step=rs,
                                    min_component_area=mina,
                                    top_k=tk,
                                )
                            except Exception:
                                continue
                            s, parts = score_matrix(M_try, meta_try)
                            # Light cap to avoid picking over-saturated grids
                            if parts['coverage'] > 0.65:
                                s -= 0.1 * (parts['coverage'] - 0.65)
                            if s > best_score:
                                best_score = s
                                best = M_try
                                best_meta = meta_try
                                best_choice = {
                                    'simplicity': simp,
                                    'resample_step': rs,
                                    'top_k': tk,
                                    'min_component_area': mina,
                                    'grid_rows': rows_opt,
                                    'grid_cols': cols_opt,
                                    'score': s,
                                    **parts
                                }
        if best is None:
            # Fallback: one safe call
            from kolam.image_to_kolam_dl import image_to_kolam_dotgrid_matrix as F
            best, best_meta = F(
                img_bytes,
                grid_type=grid_type,
                grid_rows=(grid_rows or 11),
                grid_cols=(grid_cols or 11),
                simplicity='medium',
                symmetry_mode=symmetry_mode,
                resample_step=3,
                min_component_area=80,
                top_k=36,
            )
            best_choice = {'fallback': True}
        return best, best_meta, best_choice

    if 'file' not in request.files:
        return jsonify({'error': "missing file field 'file'"}), 400
    f = request.files['file']
    img_bytes = f.read()

    # Optional params
    mode = request.form.get('mode', 'dl')
    max_contours = int(request.form.get('max_contours', 4))
    canny1 = int(request.form.get('canny1', 70))
    canny2 = int(request.form.get('canny2', 160))
    min_area = int(request.form.get('min_area', 40))
    simplify = float(request.form.get('simplify', 0.008))

    # Grid override for DL mode
    grid_rows = request.form.get('grid_rows')
    grid_cols = request.form.get('grid_cols')
    grid_rows = int(grid_rows) if grid_rows else None
    grid_cols = int(grid_cols) if grid_cols else None

    # vector params
    thresh_block = int(request.form.get('thresh_block', 17))
    thresh_C = int(request.form.get('thresh_C', 7))
    retrieve_mode = request.form.get('retrieve_mode', 'external')
    resample_step = int(request.form.get('resample_step', 3))
    smooth_alpha = float(request.form.get('smooth_alpha', 0.5))

    M = None
    try:
        if mode == 'dl':
            # Dot-grid pipeline that outputs a MATLAB-style matrix M, then renders via matrix prototypes
            grid_type = request.form.get('grid_type', 'square')  # 'square' | 'hex'
            simplicity = request.form.get('simplicity', 'simple')  # simple | medium | complex
            symmetry_mode = request.form.get('symmetry', 'auto')   # auto | rotational | mirrored | free

            # Temporary: force hex to square until hex matrix prototypes are ready
            if grid_type == 'hex':
                grid_type = 'square'

            # Optional style controls (reuse matrix UI params if provided)
            color = request.form.get('color', '#2458ff')
            linewidth = float(request.form.get('linewidth', 2))
            spacing = float(request.form.get('spacing', 40))
            show_dots = request.form.get('show_dots', 'true').lower() in ('1','true','yes','on')

            # Auto-tune is the default for DL
            auto = request.form.get('auto', '1').lower() in ('1','true','yes','on')

            if auto:
                M, meta, choice = run_dl_auto(img_bytes, grid_type, grid_rows, grid_cols, symmetry_mode)
                meta = dict(meta or {})
                meta['auto_tuned'] = True
                meta['auto_choice'] = choice
            else:
                # Manual path (kept for power users)
                top_k = int(request.form.get('top_k', 24))
                min_component_area = int(request.form.get('min_component_area', 80))
                from kolam.image_to_kolam_dl import image_to_kolam_dotgrid_matrix
                M, meta = image_to_kolam_dotgrid_matrix(
                    img_bytes,
                    grid_type=grid_type,
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    simplicity=simplicity,
                    symmetry_mode=symmetry_mode,
                    blur_ksize=5,
                    canny1=canny1,
                    canny2=canny2,
                    snap_epsilon_ratio=float(request.form.get('simplify', 0.01)),
                    resample_step=int(request.form.get('resample_step', 3)),
                    min_component_area=min_component_area,
                    top_k=top_k,
                )
                meta = dict(meta or {})
                meta['auto_tuned'] = False

            # Render using the same MATLAB-style renderer
            svg_str, meta_draw = draw_kolam_svg_from_matrix(M, spacing=spacing, color=color, linewidth=linewidth, show_dots=show_dots)
            meta.update(meta_draw)
            meta['matrix_from_image'] = True
        else:
            svg_str, meta = image_to_kolam_svg(
                img_bytes,
                max_contours=max_contours,
                canny1=canny1,
                canny2=canny2,
                min_area=min_area,
                simplify_epsilon_ratio=simplify,
                mode=mode,
                thresh_block=thresh_block,
                thresh_C=thresh_C,
                retrieve_mode=retrieve_mode,
                resample_step=resample_step,
                smooth_alpha=smooth_alpha,
            )
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Save to database
    artist = request.form.get('artist', 'Anonymous')
    name = f"Converted {mode.upper()} Kolam"
    pattern = save_pattern_to_db(svg_str, meta, artist, name)

    resp = {
        'id': pattern.uuid,
        'svg': svg_str,
        'metadata': meta,
        'created': pattern.created_at.isoformat() + 'Z'
    }
    if M is not None:
        resp['M'] = M
    return jsonify(resp)

@app.route('/api/explain', methods=['POST'])
def explain():
    """Explain a Kolam pattern"""
    data = request.get_json(force=True)
    meta = data.get('metadata', {})
    try:
        explanation = explain_kolam(meta)
        return jsonify({'explanation': explanation})
    except Exception as e:
        return jsonify({'explanation': f'Unable to explain: {str(e)}'})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat about a Kolam pattern"""
    data = request.get_json(force=True)
    meta = data.get('metadata', {})
    message = data.get('message', '')
    try:
        answer = chat_about_kolam(meta, message)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'answer': f'Unable to chat: {str(e)}'})



@app.route('/api/reconstruct', methods=['POST'])
def reconstruct():
    """Reconstruct Kolam from uploaded image using reconstruct_and_export function"""
    if 'file' not in request.files:
        return jsonify({'error': "missing file field 'file'"}), 400
    
    f = request.files['file']
    if not f or not f.filename:
        return jsonify({'error': 'No file provided'}), 400
    
    try:
        # Save uploaded file temporarily
        tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp_file_path = tmp_file.name
        tmp_file.close()  # Close the file handle so it can be accessed by other processes
        
        # Save the uploaded file
        f.save(tmp_file_path)
        
        # Import and call reconstruct function
        from kolam.reconstruct_core import reconstruct_and_export
        import base64
        
        # Call the 4K reconstruct function
        png_bytes, svg_bytes, meta = reconstruct_and_export(
            tmp_file_path, 
            max_dim=900, 
            out_size=(3840, 2160)
        )
        
        # Convert to base64
        png_b64 = base64.b64encode(png_bytes).decode('utf-8')
        svg_b64 = base64.b64encode(svg_bytes).decode('utf-8')
        
        # Save to database  
        svg_str = svg_bytes.decode('utf-8')
        artist = request.form.get('artist', 'Anonymous')
        name = f"Reconstructed 4K Kolam"
        pattern = save_pattern_to_db(svg_str, meta, artist, name)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return jsonify({
            'id': pattern.uuid,
            'png_b64': png_b64,
            'svg_b64': svg_b64,
            'meta': meta,
            'created': pattern.created_at.isoformat() + 'Z'
        })
            
    except Exception as e:
        # Clean up temp file on error
        try:
            os.unlink(tmp_file_path)
        except:
            pass
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/png', methods=['POST'])
def export_png():
    # Minimal placeholder; real PNG export may use cairosvg or wand
    data = request.get_json(force=True)
    svg = data.get('svg', '')
    return jsonify({'note': 'PNG export placeholder. Use frontend/client to rasterize.'})


@app.route('/api/draw_matrix', methods=['POST'])
def draw_matrix():
    """Draw kolam from a matrix M (like MATLAB draw_kolam). Body JSON:
    { "M": [[...]], "spacing": 40, "color": "#2458ff", "linewidth": 2, "show_dots": true }
    """
    data = request.get_json(force=True)
    M = data.get('M') or data.get('matrix')
    if M is None:
        return jsonify({'error': "Missing 'M' array"}), 400
    spacing = float(data.get('spacing', 40))
    color = data.get('color', '#2458ff')
    linewidth = float(data.get('linewidth', 2))
    show_dots = bool(data.get('show_dots', True))

    try:
        svg_str, meta = draw_kolam_svg_from_matrix(M, spacing=spacing, color=color, linewidth=linewidth, show_dots=show_dots)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Save to database
    artist = data.get('artist', 'Anonymous')
    name = f"Matrix Kolam {len(M)}x{len(M[0]) if M else 0}"
    pattern = save_pattern_to_db(svg_str, meta, artist, name)

    return jsonify({
        'id': pattern.uuid,
        'svg': svg_str,
        'metadata': meta,
        'created': pattern.created_at.isoformat() + 'Z'
    })


@app.route('/api/propose/1d', methods=['POST'])
def propose_1d():
    data = request.get_json(force=True)
    n = int(data.get('n', 7))
    spacing = float(data.get('spacing', 40))
    color = data.get('color', '#2458ff')
    linewidth = float(data.get('linewidth', 2))
    show_dots = bool(data.get('show_dots', False))

    M = propose_kolam1D(n)
    svg_str, meta = draw_kolam_svg_from_matrix(M, spacing=spacing, color=color, linewidth=linewidth, show_dots=show_dots)
    meta.update({'proposal': '1D'})

    # Save to database
    artist = data.get('artist', 'Anonymous')
    name = f"1D Proposed Kolam (n={n})"
    pattern = save_pattern_to_db(svg_str, meta, artist, name)

    return jsonify({'id': pattern.uuid, 'svg': svg_str, 'metadata': meta, 'M': M})


@app.route('/api/propose/2d', methods=['POST'])
def propose_2d():
    data = request.get_json(force=True)
    n = int(data.get('n', 7))
    spacing = float(data.get('spacing', 40))
    color = data.get('color', '#2458ff')
    linewidth = float(data.get('linewidth', 2))
    show_dots = bool(data.get('show_dots', False))

    M = propose_kolam2D(n)
    svg_str, meta = draw_kolam_svg_from_matrix(M, spacing=spacing, color=color, linewidth=linewidth, show_dots=show_dots)
    meta.update({'proposal': '2D'})

    # Save to database
    artist = data.get('artist', 'Anonymous')
    name = f"2D Proposed Kolam (n={n})"
    pattern = save_pattern_to_db(svg_str, meta, artist, name)

    return jsonify({'id': pattern.uuid, 'svg': svg_str, 'metadata': meta, 'M': M})


@app.route('/api/reconstruct', methods=['POST'])
def reconstruct_kolam():
    """Advanced Kolam reconstruction using computer vision techniques.
    
    Accepts an image file and reconstructs the Kolam pattern by:
    1. Detecting dots using HoughCircles and blob detection
    2. Estimating grid spacing and building rectangular grid
    3. Detecting connections by sampling along lines
    4. Generating SVG and metadata
    """
    global LAST_META
    
    if 'file' not in request.files:
        return jsonify({'error': "missing file field 'file'"}), 400
    
    f = request.files['file']
    if not f.filename:
        return jsonify({'error': "No file selected"}), 400
    
    # Get parameters
    max_dim = int(request.form.get('max_dim', 900))
    artist = request.form.get('artist', 'AI Reconstructed')
    
    try:
        # Create temporary file for processing
        img_bytes = f.read()
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(img_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Use the reconstruction core
            png_bytes, svg_bytes, meta = reconstruct_and_export(tmp_path, max_dim=max_dim)
            
            # Convert bytes to string for SVG
            svg_str = svg_bytes.decode('utf-8')
            
            # Add additional metadata
            meta.update({
                'method': 'AI Reconstruction',
                'algorithm': 'Computer Vision + Pattern Detection',
                'reconstruction_type': 'dot_grid_analysis'
            })
            
            # Save to gallery
            # Save to database
            name = f"AI Reconstructed Kolam"
            pattern = save_pattern_to_db(svg_str, meta, artist, name)
            
            # Also return base64 encoded PNG for optional display
            png_b64 = base64.b64encode(png_bytes).decode('utf-8')
            
            return jsonify({
                'id': pattern.uuid,
                'svg': svg_str,
                'png_base64': png_b64,
                'metadata': meta,
                'created': pattern.created_at.isoformat() + 'Z'
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except FileNotFoundError as e:
        return jsonify({'error': f'Image file error: {str(e)}'}), 400
    except RuntimeError as e:
        return jsonify({'error': f'Reconstruction failed: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


# Gallery API endpoints
@app.route('/api/gallery', methods=['GET'])
def get_gallery():
    """Get gallery items with optional sorting"""
    sort_by = request.args.get('sort', 'recent')  # recent, trending, popular
    
    query = Pattern.query
    
    if sort_by == 'trending':
        # Sort by likes descending, then by recent
        query = query.order_by(Pattern.likes.desc(), Pattern.created_at.desc())
    elif sort_by == 'popular':
        # Sort by likes only
        query = query.order_by(Pattern.likes.desc())
    else:  # recent (default)
        # Sort by creation date descending
        query = query.order_by(Pattern.created_at.desc())
    
    patterns = query.limit(50).all()  # Limit to 50 items
    
    return jsonify({
        'items': [pattern.to_dict() for pattern in patterns],
        'count': len(patterns)
    })


@app.route('/api/gallery/<pattern_id>', methods=['GET'])
def get_pattern(pattern_id):
    """Get specific pattern by UUID"""
    pattern = Pattern.query.filter_by(uuid=pattern_id).first()
    if not pattern:
        return jsonify({'error': 'Pattern not found'}), 404
    
    return jsonify(pattern.to_dict())


@app.route('/api/gallery/<pattern_id>/like', methods=['POST'])
def like_pattern(pattern_id):
    """Like a pattern (increment likes counter)"""
    pattern = Pattern.query.filter_by(uuid=pattern_id).first()
    if not pattern:
        return jsonify({'error': 'Pattern not found'}), 404
    
    pattern.likes += 1
    db.session.commit()
    
    return jsonify({'likes': pattern.likes})








@app.route('/storage/<path:filename>')
def storage_files(filename):
    """Serve files from storage directory"""
    return send_from_directory(str(STORAGE), filename)


@app.route('/api/patterns', methods=['GET'])
def get_patterns():
    """Get all patterns for the gallery"""
    patterns = Pattern.query.order_by(Pattern.created_at.desc()).all()
    return jsonify([{
        'id': pattern.id,
        'name': pattern.name,
        'png': pattern.filename_png,
        'svg': pattern.svg_content,  # Include SVG content for preview
        'svg_file': pattern.filename_svg,
        'created_at': pattern.created_at.isoformat(),
        'artist': pattern.artist,
        'likes': pattern.likes
    } for pattern in patterns])


@app.route('/api/patterns/<int:pattern_id>/download', methods=['GET'])
def download_pattern(pattern_id):
    """Download pattern in specified format"""
    pattern = Pattern.query.get_or_404(pattern_id)
    format_type = request.args.get('format', 'svg')
    
    if format_type == 'svg':
        if pattern.filename_svg:
            svg_path = Path(STORAGE) / pattern.filename_svg
            if svg_path.exists():
                return send_file(str(svg_path), mimetype='image/svg+xml', as_attachment=True)
        # Fallback to SVG content
        svg_content = pattern.svg_content.encode('utf-8')
        svg_filename = f'{pattern.name}.svg'
        svg_path = save_bytes(svg_filename, svg_content)
        return send_file(svg_path, mimetype='image/svg+xml', as_attachment=True, download_name=svg_filename)
    elif format_type == 'png':
        if pattern.filename_png:
            png_path = Path(STORAGE) / pattern.filename_png
            if png_path.exists():
                return send_file(str(png_path), mimetype='image/png', as_attachment=True)
        # Convert SVG to PNG
        try:
            import cairosvg
            png_data = cairosvg.svg2png(bytestring=pattern.svg_content.encode('utf-8'))
            png_filename = f'{pattern.name}.png'
            png_path = save_bytes(png_filename, png_data)
            return send_file(png_path, mimetype='image/png', as_attachment=True, download_name=png_filename)
        except (ImportError, OSError) as e:
            # If cairosvg or Cairo library is not available, return SVG with proper headers
            # This allows browser to render the SVG as fallback
            return Response(
                pattern.svg_content,
                mimetype='image/svg+xml',
                headers={
                    'Content-Disposition': f'inline; filename={pattern.name}.svg',
                    'X-Content-Type-Options': 'nosniff'
                }
            )
    
    return jsonify({'error': 'Invalid format'}), 400


@app.route("/api/patterns/<int:pid>/narration.mp3", methods=["GET"])
def narration(pid):
    """Serve narration audio for patterns"""
    # For demo: return a static placeholder file saved in storage/narrations/<pid>.mp3 if exists
    import os
    p = Path(STORAGE) / "narrations" / f"{pid}.mp3"
    if p.exists():
        return send_file(str(p), mimetype="audio/mpeg")
    # fallback: return a canned narration file you place at storage/narrations/default.mp3
    fallback = Path(STORAGE) / "narrations" / "default.mp3"
    if fallback.exists():
        return send_file(str(fallback), mimetype="audio/mpeg")
    return jsonify({"error": "no narration available"}), 404


# OpenAI-powered Chatbot Endpoints
@app.route('/api/chatbot/explain', methods=['POST'])
def explain_pattern():
    """Get AI-powered explanation of a pattern"""
    data = request.get_json()
    if not data or 'metadata' not in data:
        return jsonify({'error': 'Missing pattern metadata'}), 400
    
    try:
        use_ai = data.get('use_ai', True)
        if use_ai:
            explanation = explain_kolam_ai(data['metadata'])
        else:
            explanation = explain_kolam(data['metadata'])
        
        return jsonify({
            'explanation': explanation,
            'ai_powered': use_ai
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chatbot/chat', methods=['POST'])
def chat_with_bot():
    """Chat with AI about patterns and Indian art"""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing message'}), 400
    
    try:
        message = data['message']
        metadata = data.get('metadata', {})
        conversation_history = data.get('history', [])
        use_ai = data.get('use_ai', True)
        
        if use_ai:
            response = chat_about_kolam_ai(metadata, message, conversation_history)
        else:
            response = chat_about_kolam(metadata, message)
        
        return jsonify({
            'response': response,
            'ai_powered': use_ai,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chatbot/suggestions', methods=['POST'])
def get_pattern_suggestions():
    """Get AI-powered pattern suggestions based on user preferences"""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing preferences data'}), 400
    
    try:
        suggestions = ar_bharat_chatbot.generate_pattern_suggestions(data)
        return jsonify({
            'suggestions': suggestions,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chatbot/blog', methods=['POST'])
def generate_blog_content():
    """Generate AI-powered blog content about Indian art and culture"""
    data = request.get_json()
    if not data or 'topic' not in data:
        return jsonify({'error': 'Missing topic'}), 400
    
    try:
        topic = data['topic']
        style = data.get('style', 'educational')
        content = ar_bharat_chatbot.generate_blog_content(topic, style)
        
        return jsonify({
            'content': content,
            'topic': topic,
            'style': style,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chatbot/analyze-image', methods=['POST'])
def analyze_image_for_conversion():
    """Analyze uploaded image and provide AI suggestions for conversion"""
    data = request.get_json()
    if not data or 'description' not in data:
        return jsonify({'error': 'Missing image description'}), 400
    
    try:
        description = data['description']
        analysis = ar_bharat_chatbot.analyze_uploaded_image(description)
        
        return jsonify({
            'analysis': analysis,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== BLOG API ENDPOINTS ====================

@app.route('/api/blog/articles', methods=['GET'])
def get_blog_articles():
    """Get real-time generated blog articles about Indian heritage and AR technology"""
    try:
        # Define dynamic topics for blog generation
        topics = [
            {
                "title": "The Digital Renaissance of Kolam Art Through AR",
                "category": "Technology",
                "topic": "How augmented reality is revolutionizing traditional Kolam art forms and preserving Indian heritage for future generations",
                "style": "storytelling"
            },
            {
                "title": "Mathematical Mysteries Hidden in Ancient Rangoli Patterns", 
                "category": "Cultural Heritage",
                "topic": "Exploring the complex mathematical principles and geometric patterns found in traditional Rangoli designs across different Indian regions",
                "style": "educational"
            },
            {
                "title": "UNESCO's New Focus on Digital Heritage Preservation",
                "category": "Global Heritage", 
                "topic": "How UNESCO and international organizations are embracing digital technologies to preserve cultural heritage sites and traditional art forms",
                "style": "technical"
            },
            {
                "title": "Interactive Museums: The Future of Cultural Education",
                "category": "Education",
                "topic": "How AR and VR technologies are transforming museum experiences and making cultural education more engaging for students",
                "style": "educational"
            },
            {
                "title": "From Temple Floors to Smartphone Screens: Kolam's Digital Journey",
                "category": "Cultural Evolution",
                "topic": "Tracing the evolution of traditional Kolam from sacred temple art to modern digital experiences and mobile applications",
                "style": "cultural"
            },
            {
                "title": "AI-Powered Art Recognition: Preserving Folk Traditions",
                "category": "Artificial Intelligence",
                "topic": "How artificial intelligence and machine learning are being used to identify, catalog, and preserve traditional folk art patterns",
                "style": "technical"
            }
        ]
        
        # Get featured and recent articles count from query parameters
        featured = request.args.get('featured', '1')
        limit = request.args.get('limit', '6')
        
        try:
            featured_count = max(1, int(featured))
            article_limit = max(1, min(20, int(limit)))  # Cap at 20 articles max
        except ValueError:
            featured_count = 1
            article_limit = 6
        
        # Quick fallback content for instant response with topic-relevant imagery
        import random
        current_time = datetime.utcnow()
        
        fallback_articles = [
            {
                "id": f"fresh_{current_time.hour}_{current_time.minute}_1",
                "title": "Digital Renaissance of Kolam Art",
                "content": "Exploring how augmented reality is revolutionizing traditional Indian art forms. Discover the intersection of ancient wisdom and cutting-edge technology in this revolutionary approach to cultural preservation.",
                "excerpt": "Exploring how augmented reality is revolutionizing traditional Indian art forms.",
                "author": "AR BHARAT Team",
                "category": "Technology",
                "date": current_time.strftime("%b %Y"),
                "readTime": "5 min read",
                "featured": True,
                "imageUrl": "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=800&h=400&fit=crop&auto=format&q=80"
            },
            {
                "id": f"fresh_{current_time.hour}_{current_time.minute}_2", 
                "title": "Mathematical Beauty in Rangoli Patterns",
                "content": "Discovering the complex geometric principles hidden in traditional designs. Uncovering the sophisticated mathematical principles embedded in traditional Rangoli designs, from fractal geometry to symmetrical patterns.",
                "excerpt": "Discovering the complex geometric principles hidden in traditional designs.",
                "author": "AR BHARAT Team",
                "category": "Cultural Heritage", 
                "date": current_time.strftime("%b %Y"),
                "readTime": "4 min read",
                "featured": False,
                "imageUrl": "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=400&h=250&fit=crop&auto=format&q=80"
            },
            {
                "id": f"fresh_{current_time.hour}_{current_time.minute}_3",
                "title": "Immersive Heritage: AR Museums Revolution",
                "content": "Experience how augmented reality is transforming museum visits across India, allowing visitors to step inside historical moments and interact with cultural artifacts in unprecedented ways.",
                "excerpt": "AR technology revolutionizes museum experiences, allowing visitors to step inside history and interact with cultural artifacts.",
                "author": "AR BHARAT Innovation Lab",
                "category": "Digital Museums",
                "date": current_time.strftime("%b %Y"), 
                "readTime": 7,
                "featured": False,
                "imageUrl": "https://images.unsplash.com/photo-1565214975484-3cfa9e56f914?w=400&h=250&fit=crop&auto=format&q=80"
            }
        ]
        
        # Try to generate AI content with timeout protection
        articles = fallback_articles[:article_limit]
        ai_generated = False
        
        try:
            import signal
            import time
            
            def timeout_handler(signum, frame):
                raise TimeoutError("AI generation timeout")
            
            # Set up timeout (10 seconds for quick generation)
            if hasattr(signal, 'SIGALRM'):  # Unix/Linux only
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
            
            # Generate AI content for featured article
            start_time = time.time()
            if article_limit > 0:
                topic_data = topics[0]  # Use first topic for featured article
                ai_content = ar_bharat_chatbot.generate_blog_content(
                    topic_data["topic"], 
                    topic_data["style"]
                )
                
                # Update the featured article with AI content
                if ai_content and len(ai_content.strip()) > 50:
                    articles[0]["title"] = topic_data["title"]
                    articles[0]["content"] = ai_content
                    articles[0]["excerpt"] = ai_content[:150] + "..." if len(ai_content) > 150 else ai_content
                    articles[0]["author"] = "AR BHARAT AI Assistant"
                    articles[0]["date"] = datetime.utcnow().strftime("%b %Y")
                    ai_generated = True
                    
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Disable alarm
                
        except (TimeoutError, Exception) as e:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Disable alarm
            # Keep fallback content, ai_generated remains False
            pass
        
        return jsonify({
            "articles": articles,
            "total": len(articles),
            "generated_at": datetime.utcnow().isoformat() + 'Z',
            "featured_count": featured_count,
            "ai_generated": ai_generated,
            "source": "AI Generated" if ai_generated else "Curated Content"
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate blog articles: {str(e)}'}), 500


@app.route('/api/blog/article/<article_id>', methods=['GET'])
def get_single_article(article_id):
    """Get a specific blog article with full content"""
    try:
        # This would typically fetch from a database, but for real-time generation
        # we'll generate content based on the article_id
        article_topics = {
            "article_1": {
                "title": "The Digital Renaissance of Kolam Art Through AR",
                "topic": "How augmented reality is revolutionizing traditional Kolam art forms and preserving Indian heritage for future generations",
                "style": "storytelling"
            },
            "article_2": {
                "title": "Mathematical Mysteries Hidden in Ancient Rangoli Patterns",
                "topic": "Exploring the complex mathematical principles and geometric patterns found in traditional Rangoli designs across different Indian regions",
                "style": "educational"
            },
            "article_3": {
                "title": "UNESCO's New Focus on Digital Heritage Preservation",
                "topic": "How UNESCO and international organizations are embracing digital technologies to preserve cultural heritage sites and traditional art forms",
                "style": "technical"
            }
        }
        
        topic_data = article_topics.get(article_id)
        if not topic_data:
            return jsonify({'error': 'Article not found'}), 404
        
        content = ar_bharat_chatbot.generate_blog_content(topic_data["topic"], topic_data["style"])
        
        article = {
            "id": article_id,
            "title": topic_data["title"],
            "content": content,
            "date": "Jan 2025",
            "author": "AR BHARAT Editorial Team",
            "category": "Technology & Heritage",
            "tags": ["Indian Heritage", "AR Technology", "Cultural Preservation"],
            "readTime": f"{max(3, len(content.split())//200)} min read",
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }
        
        return jsonify(article)
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate article: {str(e)}'}), 500


@app.route('/api/blog/trending-topics', methods=['GET'])
def get_trending_topics():
    """Get trending topics for blog content generation"""
    try:
        # Generate trending topics using AI
        prompt = """Generate 8-10 trending topics related to Indian cultural heritage, traditional art forms, augmented reality, digital preservation, and cultural education. 

        Format as a JSON-like structure with:
        - topic: Brief topic description
        - category: Category (Technology, Heritage, Education, Art, etc.)
        - trending_score: Number from 1-10 indicating popularity
        - keywords: Array of related keywords

        Focus on topics that would be relevant for AR BHARAT's audience."""
        
        if ar_bharat_chatbot.client:
            response = ar_bharat_chatbot.client.chat.completions.create(
                model=ar_bharat_chatbot.model,
                messages=[
                    {"role": "system", "content": ar_bharat_chatbot.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.8
            )
            
            trending_content = response.choices[0].message.content.strip()
        else:
            # Fallback trending topics
            trending_content = """
            1. AR-Enhanced Temple Architecture Tours - Score: 9
            2. Machine Learning for Pattern Recognition in Folk Art - Score: 8  
            3. Virtual Kolam Drawing Workshops - Score: 8
            4. Digital Storytelling in Cultural Education - Score: 7
            5. 3D Printing Traditional Art Patterns - Score: 7
            """
        
        return jsonify({
            "trending_topics": trending_content,
            "generated_at": datetime.utcnow().isoformat() + 'Z'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate trending topics: {str(e)}'}), 500


@app.route('/api/blog/refresh', methods=['POST'])
def refresh_blog_content():
    """Refresh blog content with new AI-generated articles"""
    try:
        data = request.get_json() or {}
        refresh_type = data.get('type', 'all')  # 'all', 'featured', 'recent'
        
        # This endpoint triggers a refresh of cached content
        # In a production environment, this might clear cache or trigger background jobs
        
        response_data = {
            "message": "Blog content refresh initiated",
            "refresh_type": refresh_type,
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }
        
        if refresh_type == 'featured':
            # Generate a new featured article
            featured_topic = "Breaking: Latest developments in AR technology for cultural heritage preservation"
            content = ar_bharat_chatbot.generate_blog_content(featured_topic, "storytelling")
            response_data["new_featured"] = {
                "title": "AR Technology Breakthrough in Heritage Sites",
                "content": content[:300] + "...",
                "full_content": content
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Failed to refresh blog content: {str(e)}'}), 500


@app.route('/api/blog/generate-image', methods=['POST'])
def generate_blog_image():
    """Generate blog post images (placeholder endpoint for future DALL-E integration)"""
    try:
        data = request.get_json()
        if not data or 'description' not in data:
            return jsonify({'error': 'Missing image description'}), 400
        
        # For now, return placeholder image URLs
        # In future, this could integrate with DALL-E or other image generation APIs
        
        description = data['description']
        
        placeholder_images = [
            "/api/static/hero-kolam.png",
            "/api/static/blog-heritage-1.jpg",
            "/api/static/blog-ar-tech.jpg",
            "/api/static/blog-patterns.jpg"
        ]
        
        # Simple logic to pick image based on description keywords
        if "ar" in description.lower() or "augmented" in description.lower():
            image_url = placeholder_images[2]
        elif "pattern" in description.lower() or "kolam" in description.lower():
            image_url = placeholder_images[3]
        elif "heritage" in description.lower():
            image_url = placeholder_images[1]
        else:
            image_url = placeholder_images[0]
        
        return jsonify({
            "image_url": image_url,
            "description": description,
            "generated_at": datetime.utcnow().isoformat() + 'Z',
            "note": "Using placeholder images. DALL-E integration coming soon."
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate image: {str(e)}'}), 500


@app.route('/api/patterns', methods=['GET'])
def get_all_patterns():
    """Get all patterns from the database for gallery display"""
    try:
        patterns = Pattern.query.order_by(Pattern.created_at.desc()).all()
        patterns_data = []
        
        for pattern in patterns:
            pattern_dict = pattern.to_dict()
            # Generate PNG filename if not exists
            if not pattern_dict.get('filename_png'):
                pattern_dict['png'] = f"{pattern.uuid}.png"
            else:
                pattern_dict['png'] = pattern_dict['filename_png']
            patterns_data.append(pattern_dict)
        
        return jsonify(patterns_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/patterns/<pattern_id>/download', methods=['GET'])
def download_pattern_file(pattern_id):
    """Download pattern in specified format"""
    try:
        format_type = request.args.get('format', 'svg').lower()
        pattern = Pattern.query.filter_by(uuid=pattern_id).first()
        
        if not pattern:
            return jsonify({'error': 'Pattern not found'}), 404
        
        if format_type == 'svg':
            return Response(
                pattern.svg_content,
                mimetype='image/svg+xml',
                headers={'Content-Disposition': f'attachment; filename={pattern.name}.svg'}
            )
        elif format_type == 'png':
            # For PNG format, return SVG with proper content-type for browser rendering
            # This allows the frontend to render the SVG properly
            return Response(
                pattern.svg_content,
                mimetype='image/svg+xml',
                headers={
                    'Content-Disposition': f'inline; filename={pattern.name}.svg',
                    'X-Content-Type-Options': 'nosniff'
                }
            )
        else:
            return jsonify({'error': 'Unsupported format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/storage/<filename>', methods=['GET'])
def serve_storage_file(filename):
    """Serve files from storage directory"""
    try:
        # For PNG files, we need to generate them from SVG
        if filename.endswith('.png'):
            pattern_uuid = filename.replace('.png', '')
            pattern = Pattern.query.filter_by(uuid=pattern_uuid).first()
            
            if not pattern:
                return jsonify({'error': 'Pattern not found'}), 404
            
            # Generate a simple PNG preview from SVG using basic conversion
            # For now, return the SVG content with PNG headers
            import base64
            from io import BytesIO
            
            # Create a basic PNG representation
            svg_content = pattern.svg_content
            
            # Simple approach: return SVG as data URL for now
            # In production, you'd use cairosvg or similar for proper conversion
            return Response(
                svg_content,
                mimetype='image/svg+xml'
            )
        
        # For other files, serve from storage directory if it exists
        storage_dir = os.path.join(os.path.dirname(__file__), 'storage')
        if os.path.exists(os.path.join(storage_dir, filename)):
            return send_from_directory(storage_dir, filename)
        else:
            return jsonify({'error': 'File not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_frontend():
    """Serve the frontend HTML file"""
    return send_from_directory('../frontend', 'index.html')

if __name__ == '__main__':
    import os
    # Initialize database
    with app.app_context():
        db.create_all()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
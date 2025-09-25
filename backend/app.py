from flask import Flask, request, jsonify, send_file, send_from_directory
from io import BytesIO
from kolam.generator import generate_kolam_svg
from kolam.image_converter import image_to_kolam_svg
from kolam.image_to_kolam_dl import convert_rangoli_to_kolam as dl_convert
from kolam.matrix_draw import draw_kolam_svg_from_matrix
from kolam.propose import propose_kolam1D, propose_kolam2D
from chatbot.explainer import explain_kolam, chat_about_kolam
from flask_cors import CORS
from uuid import uuid4
from datetime import datetime
import base64
import os

app = Flask(__name__)
CORS(app)

# Serve frontend under /app to avoid file:// CORS issues
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

@app.route('/app/')
def app_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/app/<path:filename>')
def app_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# In-memory state (demo)
GALLERY = []  # list of {id, svg, metadata, created}
MAX_GALLERY = 20
LAST_META = None

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/api/generate', methods=['POST'])
def generate():
    global LAST_META
    data = request.get_json(force=True)
    rows = int(data.get('rows', 5))
    cols = int(data.get('cols', 5))
    spacing = float(data.get('spacing', 40))
    grid_type = data.get('grid_type', 'square')  # square|triangular|circular (square supported now)
    style = data.get('style', 'traditional')     # placeholder for future variants

    svg_str, meta = generate_kolam_svg(rows, cols, spacing, grid_type, style)

    # Save to in-memory gallery
    item = {
        'id': str(uuid4()),
        'svg': svg_str,
        'metadata': meta,
        'created': datetime.utcnow().isoformat() + 'Z',
        'likes': 0,
        'artist': data.get('artist', 'Anonymous') if isinstance(data, dict) else 'Anonymous'
    }
    GALLERY.insert(0, item)
    if len(GALLERY) > MAX_GALLERY:
        GALLERY.pop()

    LAST_META = meta

    return jsonify({
        'id': item['id'],
        'svg': svg_str,
        'metadata': meta,
        'created': item['created']
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
    mode = request.form.get('mode', 'vector')
    max_contours = int(request.form.get('max_contours', 10))
    canny1 = int(request.form.get('canny1', 100))
    canny2 = int(request.form.get('canny2', 200))
    min_area = int(request.form.get('min_area', 800))
    simplify = float(request.form.get('simplify', 0.01))

    # Grid override for DL mode
    grid_rows = request.form.get('grid_rows')
    grid_cols = request.form.get('grid_cols')
    grid_rows = int(grid_rows) if grid_rows else None
    grid_cols = int(grid_cols) if grid_cols else None

    # vector params
    thresh_block = int(request.form.get('thresh_block', 31))
    thresh_C = int(request.form.get('thresh_C', 3))
    retrieve_mode = request.form.get('retrieve_mode', 'tree')
    resample_step = int(request.form.get('resample_step', 8))
    smooth_alpha = float(request.form.get('smooth_alpha', 0.8))

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

    # Save to gallery
    item = {
        'id': str(uuid4()),
        'svg': svg_str,
        'metadata': meta,
        'created': datetime.utcnow().isoformat() + 'Z',
        'likes': 0,
        'artist': request.form.get('artist', 'Anonymous')
    }
    GALLERY.insert(0, item)
    if len(GALLERY) > MAX_GALLERY:
        GALLERY.pop()

    LAST_META = meta

    resp = {
        'id': item['id'],
        'svg': svg_str,
        'metadata': meta,
        'created': item['created']
    }
    if M is not None:
        resp['M'] = M
    return jsonify(resp)

@app.route('/api/explain', methods=['POST'])
def explain():
    data = request.get_json(force=True)
    meta = data.get('metadata', {}) or LAST_META or {}
    explanation = explain_kolam(meta)
    return jsonify({'explanation': explanation})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    meta = data.get('metadata', {}) or LAST_META or {}
    message = data.get('message', '')
    answer = chat_about_kolam(meta, message)
    return jsonify({'answer': answer})

@app.route('/api/gallery', methods=['GET'])
def gallery_list():
    # Return recent items (include svg for simplicity; a real app might send thumbnails)
    # Optional: sort parameter (recent|trending)
    sort = request.args.get('sort', 'recent')
    items = list(GALLERY)
    if sort == 'trending':
        items = sorted(items, key=lambda x: (x.get('likes', 0), x.get('created', '')), reverse=True)
    return jsonify({
        'items': [
            {
                'id': it['id'],
                'created': it['created'],
                'metadata': it['metadata'],
                'svg': it['svg'],
                'likes': it.get('likes', 0),
                'artist': it.get('artist', 'Anonymous')
            } for it in items
        ]
    })

@app.route('/api/gallery/<item_id>', methods=['GET'])
def gallery_get(item_id):
    for it in GALLERY:
        if it['id'] == item_id:
            return jsonify(it)
    return jsonify({'error': 'not found'}), 404

@app.route('/api/gallery/<item_id>/like', methods=['POST'])
def gallery_like(item_id):
    for it in GALLERY:
        if it['id'] == item_id:
            it['likes'] = it.get('likes', 0) + 1
            return jsonify({'id': item_id, 'likes': it['likes']})
    return jsonify({'error': 'not found'}), 404

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

    item = {
        'id': str(uuid4()),
        'svg': svg_str,
        'metadata': meta,
        'created': datetime.utcnow().isoformat() + 'Z',
        'likes': 0,
        'artist': data.get('artist', 'Anonymous')
    }
    GALLERY.insert(0, item)
    if len(GALLERY) > MAX_GALLERY:
        GALLERY.pop()

    return jsonify({
        'id': item['id'],
        'svg': svg_str,
        'metadata': meta,
        'created': item['created']
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

    item = {
        'id': str(uuid4()),
        'svg': svg_str,
        'metadata': meta,
        'created': datetime.utcnow().isoformat() + 'Z',
        'likes': 0,
        'artist': data.get('artist', 'Anonymous')
    }
    GALLERY.insert(0, item)
    if len(GALLERY) > MAX_GALLERY:
        GALLERY.pop()

    return jsonify({'id': item['id'], 'svg': svg_str, 'metadata': meta, 'M': M})


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

    item = {
        'id': str(uuid4()),
        'svg': svg_str,
        'metadata': meta,
        'created': datetime.utcnow().isoformat() + 'Z',
        'likes': 0,
        'artist': data.get('artist', 'Anonymous')
    }
    GALLERY.insert(0, item)
    if len(GALLERY) > MAX_GALLERY:
        GALLERY.pop()

    return jsonify({'id': item['id'], 'svg': svg_str, 'metadata': meta, 'M': M})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
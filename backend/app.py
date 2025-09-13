from flask import Flask, request, jsonify, send_file
from io import BytesIO
from kolam.generator import generate_kolam_svg
from kolam.image_converter import image_to_kolam_svg
from chatbot.explainer import explain_kolam, chat_about_kolam
from flask_cors import CORS
from uuid import uuid4
from datetime import datetime
import base64

app = Flask(__name__)
CORS(app)

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
        'created': datetime.utcnow().isoformat() + 'Z'
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
    - mode: 'contour' | 'vector' (default: 'vector' for smoother kolam-like curves)
    - max_contours, canny1, canny2, min_area, simplify
    - thresh_block, thresh_C, retrieve_mode, resample_step, smooth_alpha (vector mode)
    """
    global LAST_META
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

    # vector params
    thresh_block = int(request.form.get('thresh_block', 31))
    thresh_C = int(request.form.get('thresh_C', 3))
    retrieve_mode = request.form.get('retrieve_mode', 'tree')
    resample_step = int(request.form.get('resample_step', 8))
    smooth_alpha = float(request.form.get('smooth_alpha', 0.8))

    try:
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
        'created': datetime.utcnow().isoformat() + 'Z'
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
    return jsonify({
        'items': [
            {
                'id': it['id'],
                'created': it['created'],
                'metadata': it['metadata'],
                'svg': it['svg']
            } for it in GALLERY
        ]
    })

@app.route('/api/gallery/<item_id>', methods=['GET'])
def gallery_get(item_id):
    for it in GALLERY:
        if it['id'] == item_id:
            return jsonify(it)
    return jsonify({'error': 'not found'}), 404

@app.route('/api/export/png', methods=['POST'])
def export_png():
    # Minimal placeholder; real PNG export may use cairosvg or wand
    data = request.get_json(force=True)
    svg = data.get('svg', '')
    return jsonify({'note': 'PNG export placeholder. Use frontend/client to rasterize.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
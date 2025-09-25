from pathlib import Path
import cv2
import numpy as np
from kolam.image_to_kolam_dl import image_to_kolam_dotgrid_svg

# Parameters from user
ROWS = 8
COLS = 8
GRID_TYPE = 'hex'  # hexagonal
SIMPLICITY = 'medium'
SYMMETRY = 'mirrored'
CANNY1 = 100
CANNY2 = 200
RESAMPLE = 8
SIMPLIFY = 0.01

# Create a blank image to drive the dot-grid fallback (no contours => serpentine)
H = W = 800
blank = np.zeros((H, W, 3), dtype=np.uint8) + 255  # white canvas
ok, enc = cv2.imencode('.png', blank)
assert ok
img_bytes = enc.tobytes()

# Generate SVG using dot-grid pipeline
svg_str, meta = image_to_kolam_dotgrid_svg(
    image_bytes=img_bytes,
    grid_type=GRID_TYPE,
    grid_rows=ROWS,
    grid_cols=COLS,
    simplicity=SIMPLICITY,
    symmetry_mode=SYMMETRY,
    canny1=CANNY1,
    canny2=CANNY2,
    snap_epsilon_ratio=SIMPLIFY,
    resample_step=RESAMPLE,
)

# Post-process colors to white-on-black
svg = svg_str
# Ensure a solid black background
svg = svg.replace(
    '<rect x="0" y="0" width="{w}" height="{h}" fill="#fff" opacity="0" />'.format(w=meta['image_size']['width'], h=meta['image_size']['height']),
    f'<rect x="0" y="0" width="{meta["image_size"]["width"]}" height="{meta["image_size"]["height"]}" fill="#000" />'
)
# Path stroke to white
svg = svg.replace('stroke="#000"', 'stroke="#fff"')
# Dot fill to subtle gray/white for visibility on black
svg = svg.replace('fill="#111"', 'fill="#888"')

out_path = Path(r"c:\Users\MAAN DUBEY\Desktop\SIH project\data\kolams\out_hex_8x8.svg")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(svg, encoding='utf-8')
print(f"Saved: {out_path}")
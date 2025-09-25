from pathlib import Path
import cv2
import numpy as np
from kolam.image_to_kolam_dl import image_to_kolam_dotgrid_svg

OUT_DIR = Path(r"c:\Users\MAAN DUBEY\Desktop\SIH project\data\kolams")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Low-contrast image: white kolam-like strokes on light gray background
h, w = 900, 900
img_low = np.full((h, w, 3), 230, dtype=np.uint8)  # light gray background
# draw faint white curves
cv2.circle(img_low, (450, 450), 280, (245, 245, 245), 6, lineType=cv2.LINE_AA)
cv2.circle(img_low, (450, 450), 180, (245, 245, 245), 6, lineType=cv2.LINE_AA)
cv2.ellipse(img_low, (450, 450), (240, 160), 0, 0, 360, (245, 245, 245), 6, lineType=cv2.LINE_AA)
ok, enc_low = cv2.imencode('.png', img_low)
img_low_bytes = enc_low.tobytes()
svg_low, meta_low = image_to_kolam_dotgrid_svg(
    img_low_bytes,
    grid_type='hex', grid_rows=8, grid_cols=8,
    simplicity='medium', symmetry_mode='mirrored',
    canny1=100, canny2=200, snap_epsilon_ratio=0.01, resample_step=8,
    enhance_contrast=True, adaptive_canny=True, multi_component=True, min_component_area=200
)
# White-on-black post
svg_low = svg_low.replace('fill="#111"', 'fill="#888"').replace('stroke="#000"', 'stroke="#fff"')
svg_low = svg_low.replace('opacity="0"', '')
svg_low = svg_low.replace('fill="#fff"', 'fill="#000"', 1)
(OUT_DIR / 'challenge_low_contrast.svg').write_text(svg_low, encoding='utf-8')

# 2) Highly intricate: many overlapping lines
img_complex = np.full((h, w, 3), 255, dtype=np.uint8)
for r in range(80, 440, 40):
    cv2.circle(img_complex, (450, 450), r, (0, 0, 0), 2, lineType=cv2.LINE_AA)
for a in range(0, 360, 15):
    x = int(450 + 400 * np.cos(np.deg2rad(a)))
    y = int(450 + 400 * np.sin(np.deg2rad(a)))
    cv2.line(img_complex, (450, 450), (x, y), (0, 0, 0), 1, lineType=cv2.LINE_AA)
ok, enc_c = cv2.imencode('.png', img_complex)
img_c_bytes = enc_c.tobytes()
svg_complex, meta_complex = image_to_kolam_dotgrid_svg(
    img_c_bytes,
    grid_type='hex', grid_rows=8, grid_cols=8,
    simplicity='medium', symmetry_mode='mirrored',
    canny1=100, canny2=200, snap_epsilon_ratio=0.01, resample_step=8,
    enhance_contrast=True, adaptive_canny=True, multi_component=True, min_component_area=300
)
svg_complex = svg_complex.replace('fill="#111"', 'fill="#888"').replace('stroke="#000"', 'stroke="#fff"')
svg_complex = svg_complex.replace('opacity="0"', '')
svg_complex = svg_complex.replace('fill="#fff"', 'fill="#000"', 1)
(OUT_DIR / 'challenge_complex.svg').write_text(svg_complex, encoding='utf-8')

# 3) Multi-component shapes: separate blobs
img_multi = np.full((h, w, 3), 255, dtype=np.uint8)
cv2.circle(img_multi, (250, 250), 120, (0, 0, 0), 4, lineType=cv2.LINE_AA)
cv2.circle(img_multi, (650, 250), 120, (0, 0, 0), 4, lineType=cv2.LINE_AA)
cv2.circle(img_multi, (450, 650), 160, (0, 0, 0), 4, lineType=cv2.LINE_AA)
ok, enc_m = cv2.imencode('.png', img_multi)
img_m_bytes = enc_m.tobytes()
svg_multi, meta_multi = image_to_kolam_dotgrid_svg(
    img_m_bytes,
    grid_type='hex', grid_rows=8, grid_cols=8,
    simplicity='medium', symmetry_mode='mirrored',
    canny1=100, canny2=200, snap_epsilon_ratio=0.01, resample_step=8,
    enhance_contrast=True, adaptive_canny=True, multi_component=True, min_component_area=200
)
svg_multi = svg_multi.replace('fill="#111"', 'fill="#888"').replace('stroke="#000"', 'stroke="#fff"')
svg_multi = svg_multi.replace('opacity="0"', '')
svg_multi = svg_multi.replace('fill="#fff"', 'fill="#000"', 1)
(OUT_DIR / 'challenge_multi_component.svg').write_text(svg_multi, encoding='utf-8')

print("Saved:")
print(OUT_DIR / 'challenge_low_contrast.svg')
print(OUT_DIR / 'challenge_complex.svg')
print(OUT_DIR / 'challenge_multi_component.svg')
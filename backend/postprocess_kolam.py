import sys
from pathlib import Path
import io
import math
import numpy as np
import cv2
import cairosvg

from kolam.image_converter import (
    _svg_header, _svg_footer, _svg_path_poly, _svg_path_beziers,
    _find_main_contours, _resample, _catmull_rom_to_bezier,
)


def svg_to_rgba_bytes(svg_text: str, scale: float = 2.0) -> bytes:
    # Render SVG to PNG bytes for OpenCV processing
    return cairosvg.svg2png(bytestring=svg_text.encode('utf-8'), scale=scale)


def png_bytes_to_bgr(png_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    # Convert RGBA -> BGR
    if img is None:
        raise RuntimeError('Failed to decode rendered PNG')
    if img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        bgr = img
    return bgr


def binarize_and_denoise(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)
    # Otsu to binarize
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Morphology to clean noise
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bin_img


def largest_symmetrical_contours(bin_img: np.ndarray, keep: int = 3) -> list:
    cnts = _find_main_contours(bin_img, mode='external')
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    return cnts[:keep]


def smooth_contour(cnt: np.ndarray, resample_step: int = 6, alpha: float = 0.8):
    pts = _resample(cnt, step=max(1, resample_step))
    if len(pts) < 4:
        return []
    return _catmull_rom_to_bezier(pts, closed=True, alpha=float(alpha))


def symmetry_enhance(beziers_list, w, h, rotations: int = 8, mirror: bool = True):
    cx, cy = w / 2.0, h / 2.0

    def rot(x, y, th):
        ct, st = math.cos(th), math.sin(th)
        dx, dy = x - cx, y - cy
        return (cx + dx * ct - dy * st, cy + dx * st + dy * ct)

    def rot_seg(seg, th):
        x0, y0, x1, y1, x2, y2, x3, y3 = seg
        r = []
        for (x, y) in [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]:
            rx, ry = rot(x, y, th)
            r.extend([rx, ry])
        return r

    def mirror_seg(seg):
        x0, y0, x1, y1, x2, y2, x3, y3 = seg
        mx = cx
        return [2*mx - x0, y0, 2*mx - x1, y1, 2*mx - x2, y2, 2*mx - x3, y3]

    out_paths = []
    for bz in beziers_list:
        for i in range(rotations):
            th = (2 * math.pi * i) / rotations
            rbz = [rot_seg(seg, th) for seg in bz]
            out_paths.append(_svg_path_beziers(rbz, stroke="#0D9488", stroke_width=3))
            if mirror:
                mbz = [mirror_seg(seg) for seg in rbz]
                out_paths.append(_svg_path_beziers(mbz, stroke="#0D9488", stroke_width=3))
    return out_paths


def fill_closed_contours(bin_img: np.ndarray, step: int = 60) -> list:
    # Extract filled areas for visual appeal
    h, w = bin_img.shape[:2]
    contours, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    fills = []
    # Simple palette/gradient via opacity bands
    palette = ["#F59E0B", "#10B981", "#6366F1", "#EF4444", "#14B8A6"]
    for i, cnt in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:10]):
        approx = cv2.approxPolyDP(cnt, 2.0, True)
        # Build SVG polygon fill
        pts = approx[:, 0, :]
        d = f"M {pts[0][0]} {pts[0][1]} " + " ".join([f"L {int(x)} {int(y)}" for (x, y) in pts[1:]]) + " Z"
        color = palette[i % len(palette)]
        fills.append(f'<path d="{d}" fill="{color}" fill-opacity="0.25" stroke="none" />')
    return fills


def process_kolam_svg(svg_path: Path, out_svg: Path, rotations: int = 8, mirror: bool = True, add_fills: bool = True):
    svg_text = svg_path.read_text(encoding='utf-8')
    # Render SVG to raster for preprocessing
    png_bytes = svg_to_rgba_bytes(svg_text, scale=2.0)
    img = png_bytes_to_bgr(png_bytes)

    # 1) Binarize and denoise
    bin_img = binarize_and_denoise(img)

    # 2) Largest contours (symmetry-preserving proxy)
    kept = largest_symmetrical_contours(bin_img, keep=3)

    # 3) Smooth jagged edges -> beziers
    beziers_list = []
    for cnt in kept:
        bz = smooth_contour(cnt, resample_step=8, alpha=0.85)
        if bz:
            beziers_list.append(bz)

    h, w = bin_img.shape[:2]

    # 4) Symmetry enhance (rotation + optional mirror)
    sym_paths = symmetry_enhance(beziers_list, w, h, rotations=rotations, mirror=mirror)

    # 5) Optional color fills
    fills = fill_closed_contours(bin_img) if add_fills else []

    # Compose final SVG
    parts = [_svg_header(w, h)]
    if fills:
        parts.append('<g id="fills">' + "\n".join(fills) + '</g>')
    parts.append('<g id="kolam">' + "\n".join(sym_paths) + '</g>')
    parts.append(_svg_footer())
    out_svg.write_text("\n".join(parts), encoding='utf-8')
    return out_svg


def main():
    if len(sys.argv) < 2:
        print("Usage: python backend/postprocess_kolam.py <kolam_svg> [output_svg]", file=sys.stderr)
        sys.exit(1)
    in_svg = Path(sys.argv[1])
    if not in_svg.exists():
        print(f"File not found: {in_svg}", file=sys.stderr)
        sys.exit(1)
    out_svg = Path(sys.argv[2]) if len(sys.argv) > 2 else in_svg.with_name(in_svg.stem + "_processed.svg")

    process_kolam_svg(in_svg, out_svg, rotations=8, mirror=True, add_fills=True)
    print(f"Wrote: {out_svg}")


if __name__ == "__main__":
    main()
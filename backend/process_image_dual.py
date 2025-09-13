import sys
from pathlib import Path
import math
import cv2
import numpy as np

# Reuse helpers from our converter
from kolam.image_converter import (
    _svg_header, _svg_footer, _svg_path_poly, _svg_path_beziers,
    _preprocess_vector, _find_main_contours, _resample, _catmull_rom_to_bezier,
)


def rotate_point(x: float, y: float, cx: float, cy: float, theta: float):
    ct, st = math.cos(theta), math.sin(theta)
    dx, dy = x - cx, y - cy
    return (cx + dx * ct - dy * st, cy + dx * st + dy * ct)


def reflect_vertical(x: float, y: float, cx: float):
    return (2 * cx - x, y)


def rotate_beziers(beziers, cx, cy, theta):
    out = []
    for seg in beziers:
        x0, y0, x1, y1, x2, y2, x3, y3 = seg
        rx0, ry0 = rotate_point(x0, y0, cx, cy, theta)
        rx1, ry1 = rotate_point(x1, y1, cx, cy, theta)
        rx2, ry2 = rotate_point(x2, y2, cx, cy, theta)
        rx3, ry3 = rotate_point(x3, y3, cx, cy, theta)
        out.append([rx0, ry0, rx1, ry1, rx2, ry2, rx3, ry3])
    return out


def reflect_beziers_vertical(beziers, cx):
    out = []
    for seg in beziers:
        x0, y0, x1, y1, x2, y2, x3, y3 = seg
        out.append([
            2*cx - x0, y0,
            2*cx - x1, y1,
            2*cx - x2, y2,
            2*cx - x3, y3,
        ])
    return out


def build_svg(paths, w, h):
    parts = [_svg_header(w, h), '<g id="pattern">']
    parts.extend(paths)
    parts.append('</g>')
    parts.append(_svg_footer())
    return "\n".join(parts)


def vector_trace(img: np.ndarray, min_area: int = 200, epsilon_px: float = 0.75):
    # Minimal preprocessing to avoid stylization: grayscale + adaptive threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    den = cv2.bilateralFilter(gray, 7, 75, 75)
    bin_img = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)

    contours_ext = _find_main_contours(bin_img, mode='external')
    contours_all = _find_main_contours(bin_img, mode='tree')
    # Combine and de-duplicate by contour length
    all_cnts = list({id(c): c for c in (contours_ext + contours_all)}.values())

    # Sort by area desc and keep reasonable amount
    all_cnts = sorted(all_cnts, key=cv2.contourArea, reverse=True)[:50]

    paths = []
    stats = []
    for cnt in all_cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = max(epsilon_px, 0.005 * peri)  # keep outlines sharp
        approx = cv2.approxPolyDP(cnt, eps, True)
        paths.append(_svg_path_poly(approx, stroke="#111", stroke_width=2))
        stats.append({"area": float(area), "perimeter": float(peri), "points": int(len(approx))})
    return paths, stats


def kolam_convert(img: np.ndarray,
                  min_area: int = 800,
                  thresh_block: int = 31,
                  thresh_C: int = 3,
                  retrieve_mode: str = 'tree',
                  resample_step: int = 8,
                  smooth_alpha: float = 0.8,
                  rotations: int = 8,
                  mirror: bool = True):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Preprocess and extract main contours
    bin_img = _preprocess_vector(img, thresh_block=thresh_block, thresh_C=thresh_C)
    contours = _find_main_contours(bin_img, mode=retrieve_mode)
    contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)

    # Keep only significant contours
    kept = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            kept.append(c)
        if len(kept) >= 3:
            break

    # Smooth each kept contour into beziers
    base_beziers = []
    for cnt in kept:
        pts = _resample(cnt, step=max(1, resample_step))
        if len(pts) < 4:
            continue
        bz = _catmull_rom_to_bezier(pts, closed=True, alpha=float(smooth_alpha))
        if bz:
            base_beziers.append(bz)

    # Radial repetition (rotational symmetry) and optional mirror
    all_paths = []
    for bz in base_beziers:
        # Rotations
        for i in range(rotations):
            theta = (2 * math.pi * i) / rotations
            rbz = rotate_beziers(bz, cx, cy, theta)
            all_paths.append(_svg_path_beziers(rbz, stroke="#0D9488", stroke_width=3))
            if mirror:
                mbz = reflect_beziers_vertical(rbz, cx)
                all_paths.append(_svg_path_beziers(mbz, stroke="#0D9488", stroke_width=3))

    return all_paths


def main():
    if len(sys.argv) < 2:
        print("Usage: python backend/process_image_dual.py <input_image> [out_dir]", file=sys.stderr)
        sys.exit(1)
    in_path = Path(sys.argv[1])
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else in_path.parent

    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to read image", file=sys.stderr)
        sys.exit(1)

    h, w = img.shape[:2]

    # Vector trace (exact outlines)
    vt_paths, _ = vector_trace(img)
    vt_svg = build_svg(vt_paths, w, h)
    vt_file = out_dir / (in_path.stem + "_trace.svg")
    vt_file.write_text(vt_svg, encoding='utf-8')

    # Kolam conversion (symmetry, smooth curves)
    kolam_paths = kolam_convert(img)
    kolam_svg = build_svg(kolam_paths, w, h)
    kolam_file = out_dir / (in_path.stem + "_kolam.svg")
    kolam_file.write_text(kolam_svg, encoding='utf-8')

    print(f"Wrote: {vt_file}")
    print(f"Wrote: {kolam_file}")


if __name__ == '__main__':
    main()
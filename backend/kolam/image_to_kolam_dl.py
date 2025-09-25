# backend/kolam/image_to_kolam_dl.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Image-to-Kolam conversion (DL-ready + Dot-grid snapping).

Provides two pipelines:
- convert_rangoli_to_kolam: raster PNG single-stroke over inferred grid (legacy/demo)
- image_to_kolam_dotgrid_svg: symmetry-aware dot-grid + snapped single stroke, returns SVG

Steps for dot-grid pipeline:
1) Symmetry and shape analysis (center, principal angle, density)
2) Grid generation (square/hex), optionally using explicit rows/cols
3) Line extraction and simplification (largest contour)
4) Snapping simplified line to nearest grid dots to form a single path
5) Output SVG with dots and a single black stroke
"""

from typing import Union, Tuple, Optional, Dict, List
import base64
import os

import cv2
import numpy as np

# Optional PyTorch integration for future DL edge maps
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    nn = None  # type: ignore


# ------------------------------
# Optional DL Edge Model (stub)
# ------------------------------
if TORCH_AVAILABLE:
    class SimpleEdgeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1)
            )
        def forward(self, x):
            return self.net(x)
else:
    SimpleEdgeNet = None  # type: ignore


def _load_torch_model(model_path: Optional[str]) -> Optional[object]:
    if not TORCH_AVAILABLE:
        return None
    model = SimpleEdgeNet()  # type: ignore[operator]
    model.eval()
    if model_path and os.path.exists(model_path):
        # model.load_state_dict(torch.load(model_path, map_location="cpu"))
        pass
    return model


# ------------------------------
# Image I/O utilities
# ------------------------------

def _read_image(input_source: Union[str, bytes, bytearray]) -> np.ndarray:
    if isinstance(input_source, str):
        img = cv2.imread(input_source, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image from path: {input_source}")
        return img
    if isinstance(input_source, (bytes, bytearray)):
        buf = np.frombuffer(input_source, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from bytes.")
        return img
    raise TypeError("input_source must be a file path (str) or image bytes.")


def _encode_png_base64(image_bgr: np.ndarray) -> str:
    ok, enc = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed.")
    return base64.b64encode(enc.tobytes()).decode("ascii")


# ------------------------------
# Processing primitives
# ------------------------------

def _preprocess(img_bgr: np.ndarray, blur_ksize: int = 5, enhance_contrast: bool = False) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if enhance_contrast:
        # CLAHE for low-contrast images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # Normalize to full range
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        # Illumination flattening (removes gradual shading)
        try:
            bg = cv2.GaussianBlur(gray, (0, 0), 15)
            gray_corr = cv2.addWeighted(gray, 1.0, bg, -1.0, 128)
        except Exception:
            gray_corr = gray
        # Enhance strokes using top-hat/black-hat and pick higher-contrast result
        try:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            th = cv2.morphologyEx(gray_corr, cv2.MORPH_TOPHAT, k)
            bh = cv2.morphologyEx(gray_corr, cv2.MORPH_BLACKHAT, k)
            # choose variant with greater standard deviation (proxy for contrast)
            c_base = float(gray_corr.std())
            c_th = float(th.std())
            c_bh = float(bh.std())
            gray = th if c_th >= c_bh and c_th >= c_base else (bh if c_bh >= c_base else gray_corr)
        except Exception:
            gray = gray_corr
    if blur_ksize > 1 and blur_ksize % 2 == 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return gray


def _dl_edge_map(gray: np.ndarray, model: Optional[object]) -> Optional[np.ndarray]:
    if model is None or not TORCH_AVAILABLE:
        return None
    import torch
    with torch.no_grad():
        inp = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        logits = model(inp)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)
        return probs


def _detect_edges(gray: np.ndarray,
                  use_deep: bool = False,
                  model: Optional[object] = None,
                  canny1: int = 100,
                  canny2: int = 200,
                  adaptive: bool = True,
                  multi_scale: bool = True,
                  scales: tuple = (1.0, 0.75, 0.5)) -> np.ndarray:
    """Detect edges with optional DL map, adaptive Canny, and multi-scale union.
    - If use_deep=True and a model is available, uses a simple DL edge map.
    - If adaptive=True, thresholds are derived from image median.
    - If multi_scale=True, runs detection at multiple scales and unions results.
    """
    if use_deep:
        probs = _dl_edge_map(gray, model)
        if probs is not None:
            thr = max(0.1, float(np.mean(probs) + 0.5 * np.std(probs)))
            edges = (probs >= thr).astype(np.uint8) * 255
            if hasattr(cv2, "ximgproc"):
                try:
                    edges = cv2.ximgproc.thinning(edges)
                except Exception:
                    pass
            return edges

    def canny_once(img: np.ndarray) -> np.ndarray:
        if adaptive:
            med = float(np.median(img))
            sigma = 0.33
            low = int(max(0, (1.0 - sigma) * med))
            high = int(min(255, (1.0 + sigma) * med))
            return cv2.Canny(img, low, high)
        return cv2.Canny(img, canny1, canny2)

    if not multi_scale:
        return canny_once(gray)

    # Multi-scale: downscale, detect, upscale, and union
    H, W = gray.shape[:2]
    acc = np.zeros((H, W), dtype=np.uint8)
    for s in scales:
        try:
            if s == 1.0:
                e = canny_once(gray)
            else:
                resized = cv2.resize(gray, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
                e_small = canny_once(resized)
                e = cv2.resize(e_small, (W, H), interpolation=cv2.INTER_NEAREST)
            acc = np.maximum(acc, e)
        except Exception:
            continue

    # Light post-processing to connect thin gaps
    try:
        # Stronger closing to bridge tiny breaks and improve continuity
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # was (3,3)
        acc = cv2.morphologyEx(acc, cv2.MORPH_CLOSE, kernel, iterations=2)  # was 1
    except Exception:
        pass
    return acc


def _estimate_symmetry(edges: np.ndarray) -> Dict[str, float]:
    h, w = edges.shape[:2]
    e = (edges > 0).astype(np.uint8)
    ys, xs = np.where(e)
    if len(xs) < 10:
        return {"cx": w/2, "cy": h/2, "angle": 0.0, "sym_h": 0.0, "sym_v": 0.0, "complexity": 0.0}

    # Center and orientation via PCA on edge pixels
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    mean = pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    # Principal axis angle in image coords (x to the right, y down)
    principal_vec = eigvecs[:, np.argmax(eigvals)]
    angle = float(np.arctan2(principal_vec[1], principal_vec[0]))  # radians

    # Symmetry via correlation with flips
    ef = edges.astype(np.float32) / 255.0 + 1e-6
    def ncc(a, b):
        a0, b0 = a - a.mean(), b - b.mean()
        denom = (np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-6)
        return float(np.clip((a0 * b0).sum() / denom, -1.0, 1.0))
    sym_h = ncc(ef, np.flipud(ef))
    sym_v = ncc(ef, np.fliplr(ef))
    complexity = float(e.sum()) / float(h * w)

    return {"cx": float(mean[0]), "cy": float(mean[1]), "angle": angle, "sym_h": sym_h, "sym_v": sym_v, "complexity": complexity}


def _rotate_points(points: np.ndarray, center: Tuple[float, float], angle_rad: float) -> np.ndarray:
    cx, cy = center
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    p = points.copy().astype(np.float32)
    p[:, 0] -= cx; p[:, 1] -= cy
    x = p[:, 0] * ca - p[:, 1] * sa
    y = p[:, 0] * sa + p[:, 1] * ca
    p[:, 0], p[:, 1] = x + cx, y + cy
    return p


def _generate_grid(center: Tuple[float, float],
                   img_size: Tuple[int, int],
                   rows: int, cols: int,
                   grid_type: str = 'square',
                   angle_rad: float = 0.0) -> Tuple[np.ndarray, float]:
    """Generate dot grid around center. Returns (points, spacing)."""
    h, w = img_size
    # spacing to fit 84% of min dimension
    span = 0.84 * min(h, w)
    if grid_type == 'hex':
        # For hex, effective cols ~ cols + 0.5 due to staggering
        cols_eff = max(cols - 1, 1) + 0.5
        rows_eff = max(rows - 1, 1)
        dx = span / cols_eff
        dy = (np.sqrt(3)/2) * dx
        xs = np.arange(cols, dtype=np.float32) * dx
        ys = np.arange(rows, dtype=np.float32) * dy
        pts = []
        for r, y in enumerate(ys):
            offset = (dx * 0.5) if (r % 2 == 1) else 0.0
            for c, x in enumerate(xs):
                pts.append([x + offset, y])
        pts = np.array(pts, dtype=np.float32)
        # Center the grid
        pts[:, 0] -= (pts[:, 0].min() + pts[:, 0].max())/2 - center[0]
        pts[:, 1] -= (pts[:, 1].min() + pts[:, 1].max())/2 - center[1]
        spacing = dx
    else:
        # square grid
        dx = span / max(cols - 1, 1)
        dy = span / max(rows - 1, 1)
        xs = np.linspace(center[0] - span/2, center[0] + span/2, cols)
        ys = np.linspace(center[1] - span/2, center[1] + span/2, rows)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')
        pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
        spacing = float((dx + dy) * 0.5)

    if angle_rad != 0.0:
        pts = _rotate_points(pts, center, angle_rad)
    return pts, spacing


def _largest_contour(edges: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[0]


def _top_k_contours(edges: np.ndarray, k: int = 3, min_area: int = 300) -> List[np.ndarray]:
    # Use RETR_LIST to include internal contours (holes) instead of only external outlines
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    contours = [c for c in contours if cv2.contourArea(c) >= float(min_area)]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[:k]


def _simplify_contour(cnt: np.ndarray, epsilon_ratio: float = 0.01, resample_step: int = 4) -> np.ndarray:
    peri = cv2.arcLength(cnt, True)
    eps = max(1.0, epsilon_ratio * peri)
    approx = cv2.approxPolyDP(cnt, eps, True)
    pts = approx[:, 0, :]
    if resample_step > 1 and len(pts) > resample_step:
        pts = pts[::resample_step]
    return pts.astype(np.float32)


def _snap_polyline_to_dots(polyline: np.ndarray, dots: np.ndarray) -> np.ndarray:
    """Map each polyline point to the nearest dot; deduplicate to create a dot path."""
    if len(polyline) == 0 or len(dots) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    # compute nearest dot index for each point
    # (broadcast distances in chunks for memory safety if needed)
    # simple approach for moderate sizes
    idxs = []
    for p in polyline:
        d2 = ((dots[:, 0] - p[0])**2 + (dots[:, 1] - p[1])**2)
        idxs.append(int(np.argmin(d2)))
    idxs = np.array(idxs, dtype=np.int32)
    # deduplicate consecutive duplicates
    keep = np.ones_like(idxs, dtype=bool)
    keep[1:] = idxs[1:] != idxs[:-1]
    uniq = idxs[keep]
    # optionally prune small backtracks: remove immediate A->B->A
    if len(uniq) >= 3:
        pruned = [uniq[0]]
        for k in range(1, len(uniq)-1):
            if not (uniq[k-1] == uniq[k+1] == uniq[k]):
                pruned.append(uniq[k])
        pruned.append(uniq[-1])
        uniq = np.array(pruned, dtype=np.int32)
    # map to coordinates
    return dots[uniq]


def _svg_header(width: int, height: int, pad: int = 10) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width+2*pad}" height="{height+2*pad}" '
        f'viewBox="{-pad} {-pad} {width+2*pad} {height+2*pad}">'
    )


def _svg_footer() -> str:
    return '</svg>'


def _svg_path_from_points(pts: np.ndarray, stroke: str = '#000', stroke_width: int = 3) -> str:
    if len(pts) == 0:
        return ''
    d = f"M {pts[0,0]:.2f} {pts[0,1]:.2f} " + " ".join([f"L {x:.2f} {y:.2f}" for x, y in pts[1:]])
    return f'<path d="{d}" stroke="{stroke}" stroke-width="{stroke_width}" fill="none" stroke-linecap="round" stroke-linejoin="round" />'


def _svg_dots(dots: np.ndarray, r: float = 3.0, fill: str = '#111') -> str:
    if len(dots) == 0:
        return ''
    circles = [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" />' for x, y in dots]
    return "\n".join(circles)


# ------------------------------
# Public APIs
# ------------------------------

def convert_rangoli_to_kolam(
    input_source: Union[str, bytes, bytearray],
    output_path: Optional[str] = None,
    return_base64: bool = True,
    use_deep: bool = False,
    model_path: Optional[str] = None,
    blur_ksize: int = 5,
    canny1: int = 100,
    canny2: int = 200,
    stroke_thickness: int = 3,
    stroke_color: Tuple[int, int, int] = (0, 0, 0),
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
) -> Dict[str, Union[str, np.ndarray, Dict]]:
    """Legacy/demo raster pipeline kept for compatibility."""
    # 1) Load & preprocess
    img_bgr = _read_image(input_source)
    h, w = img_bgr.shape[:2]
    gray = _preprocess(img_bgr, blur_ksize=blur_ksize)

    # 2) Edges (DL optional)
    model = _load_torch_model(model_path) if use_deep else None
    edges = _detect_edges(gray, use_deep=use_deep, model=model, canny1=canny1, canny2=canny2)

    # 3) Symmetry for metadata
    symmetry = _estimate_symmetry(edges)

    # 4) Build simple serpentine grid path for demo
    complexity = symmetry["complexity"]
    c = float(np.clip(complexity, 0.0, 0.25)) / 0.25
    size = int(round(4 + (12 - 4) * (0.4 + 0.6 * c)))
    rows = int(np.clip(grid_rows or size, 4, 24))
    cols = int(np.clip(grid_cols or size, 4, 24))
    dots, _ = _generate_grid((w*0.5, h*0.5), (h, w), rows, cols, grid_type='square', angle_rad=0.0)

    # serpentine path
    path = []
    for r in range(rows):
        rowpts = dots[r*cols:(r+1)*cols]
        path.append(rowpts if (r % 2 == 0) else rowpts[::-1])
    polyline = np.concatenate(path, axis=0)

    # render raster
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    pts = polyline.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(canvas, [pts], isClosed=False, color=stroke_color, thickness=stroke_thickness, lineType=cv2.LINE_AA)
    image_b64 = _encode_png_base64(canvas) if return_base64 else None
    if output_path:
        cv2.imwrite(output_path, canvas)

    res: Dict[str, Union[str, np.ndarray, Dict]] = {
        "image": canvas,
        "metadata": {
            "symmetry": symmetry,
            "grid_shape": (rows, cols),
            "params": {"use_deep": use_deep}
        }
    }
    if image_b64 is not None:
        res["image_base64"] = image_b64
    return res


def image_to_kolam_dotgrid_svg(
    image_bytes: bytes,
    grid_type: str = 'square',        # 'square' | 'hex'
    grid_rows: Optional[int] = None,  # e.g., 5 or 7
    grid_cols: Optional[int] = None,
    simplicity: str = 'simple',       # 'simple' | 'medium' | 'complex'
    symmetry_mode: str = 'auto',      # 'auto' | 'rotational' | 'mirrored' | 'free'
    blur_ksize: int = 5,
    canny1: int = 100,
    canny2: int = 200,
    snap_epsilon_ratio: float = 0.01,
    resample_step: int = 4,
    dot_radius_px: Optional[float] = None,
    enhance_contrast: bool = True,
    adaptive_canny: bool = True,
    multi_component: bool = True,
    min_component_area: int = 300,
) -> Tuple[str, Dict]:
    """Full dot-grid snapping pipeline. Returns (svg, metadata)."""
    # Decode image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image bytes")
    h, w = img.shape[:2]

    # Preprocess + Edges
    gray = _preprocess(img, blur_ksize=blur_ksize, enhance_contrast=enhance_contrast)
    edges = _detect_edges(gray, use_deep=False, model=None, canny1=canny1, canny2=canny2, adaptive=adaptive_canny)

    # Symmetry + center + angle
    sym = _estimate_symmetry(edges)
    cx, cy, angle = sym["cx"], sym["cy"], sym["angle"]

    # Decide grid size if not given
    if grid_rows is None or grid_cols is None:
        if simplicity.lower() in ('simple', 'low'):
            rows = cols = 5
        elif simplicity.lower() in ('complex', 'high'):
            rows = cols = 11
        else:
            rows = cols = 7
    else:
        rows = int(grid_rows)
        cols = int(grid_cols)

    # Determine grid orientation
    if symmetry_mode in ('rotational', 'mirrored'):
        angle_rad = angle
    elif symmetry_mode == 'free':
        angle_rad = 0.0
    else:  # auto: pick by stronger symmetry axis
        angle_rad = angle

    # Generate dots
    dots, spacing = _generate_grid((cx, cy), (h, w), rows, cols, grid_type=grid_type, angle_rad=angle_rad)
    if dot_radius_px is None:
        dot_r = max(2.0, spacing * 0.08)
    else:
        dot_r = float(dot_radius_px)

    # Select shape(s)
    snapped_components: List[np.ndarray] = []
    if multi_component:
        cnts = _top_k_contours(edges, k=5, min_area=min_component_area)
        if not cnts:
            cnts = []
        for c in cnts:
            poly = _simplify_contour(c, epsilon_ratio=snap_epsilon_ratio, resample_step=resample_step)
            snapped = _snap_polyline_to_dots(poly, dots)
            if len(snapped) >= 2:
                snapped_components.append(snapped)
        if not snapped_components:
            # Fallback serpentine
            track = []
            for r in range(rows):
                rowpts = dots[r*cols:(r+1)*cols]
                track.append(rowpts if (r % 2 == 0) else rowpts[::-1])
            snapped_components = [np.concatenate(track, axis=0)]
    else:
        cnt = _largest_contour(edges)
        if cnt is None:
            track = []
            for r in range(rows):
                rowpts = dots[r*cols:(r+1)*cols]
                track.append(rowpts if (r % 2 == 0) else rowpts[::-1])
            snapped_components = [np.concatenate(track, axis=0)]
        else:
            poly = _simplify_contour(cnt, epsilon_ratio=snap_epsilon_ratio, resample_step=resample_step)
            snapped = _snap_polyline_to_dots(poly, dots)
            if len(snapped) < 2:
                track = []
                for r in range(rows):
                    rowpts = dots[r*cols:(r+1)*cols]
                    track.append(rowpts if (r % 2 == 0) else rowpts[::-1])
                snapped = np.concatenate(track, axis=0)
            snapped_components = [snapped]

    # Build SVG with dots + path(s)
    paths_svg = []
    for snapped in snapped_components:
        paths_svg.append(_svg_path_from_points(snapped, stroke="#000", stroke_width=3))

    svg_parts = [
        _svg_header(w, h),
        '<rect x="0" y="0" width="{w}" height="{h}" fill="#fff" opacity="0" />'.format(w=w, h=h),
        '<g id="dot-grid" fill="#111">',
        _svg_dots(dots, r=dot_r, fill="#111"),
        '</g>',
        '<g id="kolam">',
        *paths_svg,
        '</g>',
        _svg_footer(),
    ]

    metadata = {
        'source': 'image',
        'image_size': {'width': w, 'height': h},
        'grid': {
            'type': grid_type,
            'rows': rows,
            'cols': cols,
            'spacing': float(spacing),
            'dot_radius': float(dot_r),
        },
        'symmetry': sym,
        'processing': {
            'simplicity': simplicity,
            'symmetry_mode': symmetry_mode,
            'snap_epsilon_ratio': float(snap_epsilon_ratio),
            'resample_step': int(resample_step),
            'canny': {'t1': int(canny1), 't2': int(canny2)}
        }
    }

    return "\n".join(svg_parts), metadata


def image_to_kolam_dotgrid_matrix(
    image_bytes: bytes,
    grid_type: str = 'square',        # 'square' | 'hex' (matrix mapping supports square only)
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
    simplicity: str = 'simple',
    symmetry_mode: str = 'auto',
    blur_ksize: int = 5,
    canny1: int = 100,
    canny2: int = 200,
    snap_epsilon_ratio: float = 0.01,
    resample_step: int = 4,
    enhance_contrast: bool = True,
    adaptive_canny: bool = True,
    multi_component: bool = True,
    min_component_area: int = 150,
    top_k: int = 12,
    auto_circle_crop: bool = False,
) -> Tuple[List[List[int]], Dict]:
    """Convert image to a MATLAB-style matrix M by snapping contours to a dot grid
    and inferring prototype ids at each dot. Returns (M, metadata)."""
    # Decode image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image bytes")

    # Optional: auto-crop circular rangoli to increase edge density
    if auto_circle_crop:
        try:
            ih, iw = img.shape[:2]
            g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            g0 = cv2.GaussianBlur(g0, (5,5), 0)
            circles = cv2.HoughCircles(g0, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(ih, iw)//3,
                                       param1=120, param2=40,
                                       minRadius=int(0.25*min(ih,iw)), maxRadius=int(0.6*min(ih,iw)))
            if circles is not None and len(circles[0]) > 0:
                cx, cy, r = circles[0][0]
                cx, cy, r = int(cx), int(cy), int(r)
                x0, y0 = max(cx - r, 0), max(cy - r, 0)
                x1, y1 = min(cx + r, iw), min(cy + r, ih)
                crop = img[y0:y1, x0:x1]
                if crop.size > 0:
                    img = crop
        except Exception:
            pass

    h, w = img.shape[:2]

    # Preprocess + Edges
    gray = _preprocess(img, blur_ksize=blur_ksize, enhance_contrast=enhance_contrast)

    # 1) Base single-scale Canny (stable default)
    edges_base = _detect_edges(
        gray,
        use_deep=False,
        model=None,
        canny1=canny1,
        canny2=canny2,
        adaptive=adaptive_canny,
        multi_scale=False
    )

    # Post-process to connect thin gaps and remove tiny specks
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        edges_base = cv2.morphologyEx(edges_base, cv2.MORPH_CLOSE, kernel, iterations=1)
        edges_base = cv2.morphologyEx(edges_base, cv2.MORPH_OPEN, kernel, iterations=1)
    except Exception:
        pass

    # 2) If edges look too sparse, auto-escalate with multi-scale + threshold unions
    def _edge_density(e):
        try:
            return float((e > 0).sum()) / float(max(1, e.size))
        except Exception:
            return 0.0

    density = _edge_density(edges_base)

    if density < 0.003:
        # Multi-scale Canny union (more sensitive to various line widths)
        try:
            edges_ms = _detect_edges(
                gray,
                use_deep=False,
                model=None,
                canny1=canny1,
                canny2=canny2,
                adaptive=True,
                multi_scale=True,
                scales=(1.0, 0.75, 0.5)
            )
        except Exception:
            edges_ms = np.zeros_like(edges_base)

        # Adaptive thresholds (both polarities) to pick faint/low-contrast strokes
        try:
            # Use two block sizes to capture different stroke scales
            b1, c1 = 31, 5
            b2, c2 = 51, 7
            th1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, b1, c1)
            th1i = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, b1, c1)
            th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, b2, c2)
            th2i = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, b2, c2)
            # Morphological gradient to convert masks to thin edges
            kgrad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            e_th = cv2.morphologyEx(th1, cv2.MORPH_GRADIENT, kgrad)
            e_thi = cv2.morphologyEx(th1i, cv2.MORPH_GRADIENT, kgrad)
            e_th2 = cv2.morphologyEx(th2, cv2.MORPH_GRADIENT, kgrad)
            e_th2i = cv2.morphologyEx(th2i, cv2.MORPH_GRADIENT, kgrad)
        except Exception:
            e_th = e_thi = e_th2 = e_th2i = np.zeros_like(edges_base)

        # Union all candidates
        edges = np.maximum.reduce([edges_base, edges_ms, e_th, e_thi, e_th2, e_th2i])

        # Final tidy-up: close small gaps; if still sparse, lightly dilate
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
            if _edge_density(edges) < 0.003:
                edges = cv2.dilate(edges, kernel, iterations=1)
        except Exception:
            pass
    else:
        edges = edges_base

    # Symmetry + center + angle (with robust centroid fallback)
    sym = _estimate_symmetry(edges)
    cx, cy, angle = sym["cx"], sym["cy"], sym["angle"]

    # If edges are sparse or off-center, prefer largest contour centroid
    try:
        cnt_tmp = _largest_contour(edges)
        if cnt_tmp is not None:
            m = cv2.moments(cnt_tmp)
            if m.get('m00', 0.0) > 1e-3:
                cx_cnt = float(m['m10'] / m['m00'])
                cy_cnt = float(m['m01'] / m['m00'])
                dens = _edge_density(edges)
                # Re-center more aggressively when density is very low
                w_cnt = 0.8 if dens < 0.005 else (0.6 if dens < 0.015 else 0.4)
                cx = w_cnt * cx_cnt + (1.0 - w_cnt) * cx
                cy = w_cnt * cy_cnt + (1.0 - w_cnt) * cy
    except Exception:
        pass

    # Decide grid size if not given
    if grid_rows is None or grid_cols is None:
        if simplicity.lower() in ('simple', 'low'):
            rows = cols = 5
        elif simplicity.lower() in ('complex', 'high'):
            rows = cols = 11
        else:
            rows = cols = 7
    else:
        rows = int(grid_rows)
        cols = int(grid_cols)

    # Determine grid orientation
    if symmetry_mode in ('rotational', 'mirrored'):
        angle_rad = angle
    elif symmetry_mode == 'free':
        angle_rad = 0.0
    else:  # auto
        angle_rad = angle

    # Generate dots
    dots, spacing = _generate_grid((cx, cy), (h, w), rows, cols, grid_type=grid_type, angle_rad=angle_rad)

    # Extract snapped components (reuse logic)
    snapped_components: List[np.ndarray] = []
    if multi_component:
        cnts = _top_k_contours(edges, k=top_k, min_area=min_component_area)
        for c in cnts:
            poly = _simplify_contour(c, epsilon_ratio=snap_epsilon_ratio, resample_step=resample_step)
            snapped = _snap_polyline_to_dots(poly, dots)
            if len(snapped) >= 2:
                snapped_components.append(snapped)
        if not snapped_components:
            track = []
            for r in range(rows):
                rowpts = dots[r*cols:(r+1)*cols]
                track.append(rowpts if (r % 2 == 0) else rowpts[::-1])
            snapped_components = [np.concatenate(track, axis=0)]
    else:
        cnt = _largest_contour(edges)
        if cnt is None:
            track = []
            for r in range(rows):
                rowpts = dots[r*cols:(r+1)*cols]
                track.append(rowpts if (r % 2 == 0) else rowpts[::-1])
            snapped_components = [np.concatenate(track, axis=0)]
        else:
            poly = _simplify_contour(cnt, epsilon_ratio=snap_epsilon_ratio, resample_step=resample_step)
            snapped = _snap_polyline_to_dots(poly, dots)
            if len(snapped) < 2:
                track = []
                for r in range(rows):
                    rowpts = dots[r*cols:(r+1)*cols]
                    track.append(rowpts if (r % 2 == 0) else rowpts[::-1])
                snapped = np.concatenate(track, axis=0)
            snapped_components = [snapped]

    # Build connectivity graph
    def nearest_idx(pt: np.ndarray) -> int:
        d2 = ((dots[:, 0] - pt[0])**2 + (dots[:, 1] - pt[1])**2)
        return int(np.argmin(d2))

    adjacency: Dict[int, set] = {}
    for comp in snapped_components:
        idxs = [nearest_idx(p) for p in comp]
        for a, b in zip(idxs[:-1], idxs[1:]):
            if a == b:
                continue
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

    # Initialize M with 1s (blank)
    M = [[1 for _ in range(cols)] for _ in range(rows)]

    # Safety: if no adjacency was built (e.g., no edges), mark center with a simple motif
    if len(adjacency) == 0 and rows > 0 and cols > 0:
        M[rows // 2][cols // 2] = 9  # round loop

    def rc_from_idx(idx: int) -> Tuple[int, int]:
        r = idx // cols
        c = idx % cols
        return r, c

    def neighbor_indices(idx: int) -> Dict[str, int]:
        r, c = rc_from_idx(idx)
        n = {}
        if r > 0: n['U'] = (r - 1) * cols + c
        if r < rows - 1: n['D'] = (r + 1) * cols + c
        if c > 0: n['L'] = r * cols + (c - 1)
        if c < cols - 1: n['R'] = r * cols + (c + 1)
        if r > 0 and c < cols - 1: n['UR'] = (r - 1) * cols + (c + 1)
        if r > 0 and c > 0: n['UL'] = (r - 1) * cols + (c - 1)
        if r < rows - 1 and c > 0: n['DL'] = (r + 1) * cols + (c - 1)
        if r < rows - 1 and c < cols - 1: n['DR'] = (r + 1) * cols + (c + 1)
        return n

    for idx, neighs in adjacency.items():
        r, c = rc_from_idx(idx)
        nmap = neighbor_indices(idx)
        has = {k: (nmap.get(k) in neighs) for k in ['L','R','U','D','UL','UR','DL','DR']}
        deg = sum(1 for v in has.values() if v)

        code = 1
        if has['L'] and has['R'] and not (has['U'] or has['D'] or has['UL'] or has['UR'] or has['DL'] or has['DR']):
            code = 2
        elif has['U'] and has['D'] and not (has['L'] or has['R'] or has['UL'] or has['UR'] or has['DL'] or has['DR']):
            code = 3
        elif has['R'] and has['D'] and not (has['L'] or has['U']):
            code = 4
        elif has['D'] and has['L'] and not (has['R'] or has['U']):
            code = 5
        elif has['L'] and has['U'] and not (has['R'] or has['D']):
            code = 6
        elif has['U'] and has['R'] and not (has['L'] or has['D']):
            code = 7
        elif (has['UR'] and not (has['U'] or has['R'])) and deg <= 2:
            code = 10
        elif (has['UL'] and not (has['U'] or has['L'])) and deg <= 2:
            code = 11
        elif (has['DL'] and not (has['D'] or has['L'])) and deg <= 2:
            code = 12
        elif (has['DR'] and not (has['D'] or has['R'])) and deg <= 2:
            code = 13
        elif deg >= 3 or ((has['L'] and has['R']) and (has['U'] or has['D'])) or ((has['U'] and has['D']) and (has['L'] or has['R'])):
            code = 8
        else:
            if deg == 1:
                if has['R']: code = 4
                elif has['D']: code = 5
                elif has['L']: code = 6
                elif has['U']: code = 7
                elif has['UR']: code = 10
                elif has['UL']: code = 11
                elif has['DL']: code = 12
                elif has['DR']: code = 13
            else:
                code = 1

        M[r][c] = int(code)

    meta = {
        'source': 'image',
        'image_size': {'width': w, 'height': h},
        'grid': {
            'type': grid_type,
            'rows': rows,
            'cols': cols,
            'spacing': float(spacing),
        },
        'symmetry': sym,
        'processing': {
            'simplicity': simplicity,
            'symmetry_mode': symmetry_mode,
            'snap_epsilon_ratio': float(snap_epsilon_ratio),
            'resample_step': int(resample_step),
            'canny': {'t1': int(canny1), 't2': int(canny2)}
        },
        'matrix_inferred': True
    }

    return M, meta


if __name__ == "__main__":
    print("Module ready. Use image_to_kolam_dotgrid_svg() for SVG output, image_to_kolam_dotgrid_matrix() for matrix + renderer, or convert_rangoli_to_kolam() for raster demo.")
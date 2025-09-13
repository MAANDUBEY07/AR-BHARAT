from typing import Tuple, Dict, List, Sequence
import numpy as np
import cv2

# Image -> Kolam-like SVG converter with two modes:
# - 'contour' (default): edges + contours + polyline simplify
# - 'vector': adaptive threshold + morphology + smoothing (Catmull-Rom -> Bezier)


def _svg_header(width: int, height: int, pad: int = 10) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width+2*pad}" height="{height+2*pad}" '
        f'viewBox="{-pad} {-pad} {width+2*pad} {height+2*pad}">'
    )


def _svg_footer() -> str:
    return '</svg>'


def _svg_path_poly(points: np.ndarray, stroke: str = '#0D9488', stroke_width: int = 2) -> str:
    # points: Nx1x2 from approxPolyDP or Nx2
    if points.ndim == 3:
        pts = points[:, 0, :]
    else:
        pts = points
    if len(pts) == 0:
        return ''
    d = f"M {pts[0][0]} {pts[0][1]} " + " ".join([f"L {float(x)} {float(y)}" for (x, y) in pts[1:]])
    return (
        f'<path d="{d}" stroke="{stroke}" stroke-width="{stroke_width}" '
        f'fill="none" stroke-linecap="round" stroke-linejoin="round" />'
    )


def _svg_path_beziers(beziers: Sequence[Sequence[float]], stroke: str = '#0D9488', stroke_width: int = 3) -> str:
    # beziers: list of (x0,y0,x1,y1,x2,y2,x3,y3) for cubic segments
    if not beziers:
        return ''
    x0, y0, *_ = beziers[0]
    d_cmds = [f"M {x0:.2f} {y0:.2f}"]
    for seg in beziers:
        _, _, x1, y1, x2, y2, x3, y3 = seg
        d_cmds.append(f"C {x1:.2f} {y1:.2f}, {x2:.2f} {y2:.2f}, {x3:.2f} {y3:.2f}")
    d = " ".join(d_cmds)
    return (
        f'<path d="{d}" stroke="{stroke}" stroke-width="{stroke_width}" '
        f'fill="none" stroke-linecap="round" stroke-linejoin="round" />'
    )


def _catmull_rom_to_bezier(points: np.ndarray, closed: bool = True, alpha: float = 0.5) -> List[List[float]]:
    """Convert a sequence of points into cubic Bezier segments via centripetal Catmull-Rom.
    Returns list of segments: [x0,y0,x1,y1,x2,y2,x3,y3].
    """
    pts = points.astype(np.float32)
    n = len(pts)
    if n < 2:
        return []
    def tj(ti, pi, pj):
        return ((np.linalg.norm(pj - pi)) ** alpha) + ti
    beziers: List[List[float]] = []
    # Build extended list for closed/open handling
    if closed:
        p = np.vstack([pts[-2], pts, pts[1]])
    else:
        # duplicate end points for endpoints
        p = np.vstack([pts[0], pts, pts[-1]])
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        # Parametrization
        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        # Derivatives as tangent vectors
        m1 = (p2 - p0) / max(t2 - t0, 1e-6)
        m2 = (p3 - p1) / max(t3 - t1, 1e-6)
        # Bezier control points
        b0 = p1
        b3 = p2
        b1 = b0 + (m1 * (t2 - t1) / 3.0)
        b2 = b3 - (m2 * (t2 - t1) / 3.0)
        beziers.append([float(b0[0]), float(b0[1]), float(b1[0]), float(b1[1]), float(b2[0]), float(b2[1]), float(b3[0]), float(b3[1])])
    return beziers


def _preprocess_vector(img: np.ndarray, thresh_block: int = 21, thresh_C: int = 5) -> np.ndarray:
    # Grayscale + contrast + denoise + adaptive threshold to binary
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    den = cv2.bilateralFilter(eq, 7, 75, 75)
    bin_img = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, thresh_block | 1, thresh_C)
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bin_img


def _find_main_contours(bin_img: np.ndarray, mode: str = 'external') -> List[np.ndarray]:
    retrieval = cv2.RETR_EXTERNAL if mode == 'external' else cv2.RETR_TREE
    contours, hierarchy = cv2.findContours(bin_img, retrieval, cv2.CHAIN_APPROX_NONE)
    return contours


def _resample(points: np.ndarray, step: int = 4) -> np.ndarray:
    if len(points.shape) == 3:
        pts = points[:, 0, :]
    else:
        pts = points
    if step <= 1 or len(pts) <= step:
        return pts
    return pts[::step]


def image_to_kolam_svg(
    image_bytes: bytes,
    max_contours: int = 3,
    canny1: int = 100,
    canny2: int = 200,
    min_area: int = 100,
    simplify_epsilon_ratio: float = 0.01,
    mode: str = 'contour',
    # vector mode params
    thresh_block: int = 21,
    thresh_C: int = 5,
    retrieve_mode: str = 'external',  # 'external' | 'tree'
    resample_step: int = 4,
    smooth_alpha: float = 0.5,
) -> Tuple[str, Dict]:
    """
    Convert an input image (bytes) into a Kolam-like SVG.

    Modes:
    - contour: Canny -> contours -> simplify (polyline)
    - vector: Adaptive threshold -> morphology -> contours -> Catmull-Rom smoothing (cubic Beziers)
    """
    # Decode image bytes using OpenCV
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image bytes")

    h, w = img.shape[:2]

    svg_paths: List[str] = []
    path_stats: List[Dict] = []

    if mode == 'vector':
        bin_img = _preprocess_vector(img, thresh_block=thresh_block, thresh_C=thresh_C)
        contours = _find_main_contours(bin_img, mode=retrieve_mode)
        # Sort by perimeter length desc
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)
        kept: List[np.ndarray] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            kept.append(cnt)
            if len(kept) >= max_contours:
                break
        # Build smoothed Bezier paths
        for idx, cnt in enumerate(kept):
            pts = _resample(cnt, step=max(1, resample_step))
            if len(pts) < 4:
                continue
            beziers = _catmull_rom_to_bezier(pts, closed=True, alpha=float(smooth_alpha))
            color = ['#0D9488', '#C026D3', '#EA580C'][idx % 3]
            svg_paths.append(_svg_path_beziers(beziers, stroke=color, stroke_width=3))
            peri = float(cv2.arcLength(cnt, True))
            path_stats.append({'perimeter': peri, 'points': int(len(pts)), 'beziers': int(len(beziers))})
    else:
        # contour mode (original)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, canny1, canny2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        kept: List[np.ndarray] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            kept.append(cnt)
            if len(kept) >= max_contours:
                break
        for idx, cnt in enumerate(kept):
            peri = cv2.arcLength(cnt, True)
            eps = max(1.0, simplify_epsilon_ratio * peri)
            approx = cv2.approxPolyDP(cnt, eps, True)
            svg_paths.append(_svg_path_poly(approx, stroke=['#0D9488', '#C026D3', '#EA580C'][idx % 3], stroke_width=3))
            path_stats.append({
                'area': float(cv2.contourArea(cnt)),
                'perimeter': float(peri),
                'points': int(len(approx)),
            })

    # Compose SVG
    svg_parts = [
        _svg_header(w, h),
        '<g id="pattern">',
        *svg_paths,
        '</g>',
        _svg_footer(),
    ]

    metadata = {
        'source': 'image',
        'image_size': {'width': w, 'height': h},
        'contours_used': len(svg_paths),
        'paths': path_stats,
        'processing': {
            'mode': mode,
            'canny': {'t1': canny1, 't2': canny2},
            'simplify_epsilon_ratio': simplify_epsilon_ratio,
            'min_area': min_area,
            'thresh_block': thresh_block,
            'thresh_C': thresh_C,
            'retrieve_mode': retrieve_mode,
            'resample_step': resample_step,
            'smooth_alpha': smooth_alpha,
        },
        'region_hint': 'N/A (image-derived)'
    }

    return "\n".join(svg_parts), metadata
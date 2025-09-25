# backend/kolam/rangoli_to_kolam.py
# -*- coding: utf-8 -*-
"""
A simple feature to convert a colorful rangoli image into a traditional South Indian
kolam-style design using OpenCV and NumPy.

Pipeline:
1) Load image and preprocess (grayscale + optional contrast enhance + blur)
2) Edge detection (adaptive Canny) and light morphology to clean edges
3) Overlay a visible dot-grid pattern
4) Reconstruct as white strokes on black background simulating kolam lines
5) Return PNG bytes; optionally save to file and/or display with Matplotlib

This is a standalone utility and does not affect existing API routes.
"""
from __future__ import annotations
from typing import Optional, Tuple
import io

import cv2
import numpy as np


def _read_image(source: str | bytes | bytearray) -> np.ndarray:
    if isinstance(source, str):
        img = cv2.imread(source, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image at path: {source}")
        return img
    if isinstance(source, (bytes, bytearray)):
        buf = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from bytes")
        return img
    raise TypeError("source must be a filepath (str) or raw image bytes")


def _adaptive_canny(gray: np.ndarray) -> np.ndarray:
    # Compute thresholds from image median for robustness
    med = float(np.median(gray))
    sigma = 0.33
    low = int(max(0, (1.0 - sigma) * med))
    high = int(min(255, (1.0 + sigma) * med))
    edges = cv2.Canny(gray, low, high)
    return edges


def _overlay_dot_grid(canvas: np.ndarray, rows: int, cols: int, radius: int = 2) -> None:
    h, w = canvas.shape[:2]
    # Fit grid to 84% of the shorter side, centered
    span = 0.84 * min(h, w)
    xs = np.linspace((w - span) / 2, (w + span) / 2, cols)
    ys = np.linspace((h - span) / 2, (h + span) / 2, rows)
    for y in ys:
        for x in xs:
            cv2.circle(canvas, (int(x), int(y)), radius, (80, 80, 80), thickness=-1, lineType=cv2.LINE_AA)


def rangoli_to_kolam(
    input_image: str | bytes | bytearray,
    grid_rows: int = 9,
    grid_cols: int = 9,
    blur_ksize: int = 5,
    enhance_contrast: bool = True,
    stroke_thickness: int = 2,
    dot_radius: int = 2,
    return_png: bool = True,
    save_path: Optional[str] = None,
    show_matplotlib: bool = False,
) -> bytes | np.ndarray:
    """
    Convert a colorful rangoli image into a kolam-style PNG.

    - Processes the image (grayscale, optional CLAHE, blur)
    - Extracts edges (adaptive Canny) and cleans with morphology
    - Overlays a dot grid
    - Draws white strokes on black background

    Returns PNG bytes by default; if return_png=False, returns BGR ndarray.
    Optionally saves to disk and/or shows via Matplotlib.
    """
    img = _read_image(input_image)
    h, w = img.shape[:2]

    # Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if enhance_contrast:
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        except Exception:
            pass
    if blur_ksize > 1 and blur_ksize % 2 == 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Edges + morphology
    edges = _adaptive_canny(gray)
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    except Exception:
        pass

    # Render on black canvas
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw detected edges as white strokes (simulate kolam lines)
    # Thin edges if ximgproc is available; else draw directly
    edges_draw = edges.copy()
    if edges_draw.ndim == 2:
        edges_draw = cv2.cvtColor(edges_draw, cv2.COLOR_GRAY2BGR)
    canvas[edges > 0] = (255, 255, 255)

    # Slight dilation + blur to make strokes look hand-drawn
    try:
        k = max(1, stroke_thickness)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        canvas = cv2.dilate(canvas, ker, iterations=1)
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=0.8)
    except Exception:
        pass

    # Overlay subtle dot grid
    _overlay_dot_grid(canvas, grid_rows, grid_cols, radius=dot_radius)

    # Optional save
    if save_path:
        cv2.imwrite(save_path, canvas)

    # Optional show via Matplotlib (avoids blocking if not available)
    if show_matplotlib:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title('Kolam-style Reconstruction')
            plt.show()
        except Exception:
            pass

    if return_png:
        ok, enc = cv2.imencode('.png', canvas)
        if not ok:
            raise RuntimeError('PNG encoding failed')
        return enc.tobytes()
    return canvas


if __name__ == "__main__":
    # Quick manual test (adjust path accordingly)
    # png = rangoli_to_kolam("../../data/kolams/demo.svg", save_path="out.png", show_matplotlib=True)
    pass
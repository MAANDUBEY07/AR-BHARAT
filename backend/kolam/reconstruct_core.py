"""
kolam_reconstruct_core.py

Complete, self-contained Kolam reconstruction core.

Features:
 - load image, preprocess
 - detect dots (HoughCircles fallback to blob centroids)
 - estimate spacing, group into rows, build rectangular grid
 - detect connections by sampling along straight lines
 - produce SVG and PNG bytes and metadata

Usage:
  from kolam.reconstruct_core import reconstruct_and_export
  png_bytes, svg_bytes, meta = reconstruct_and_export("/path/to/image.jpg")
"""

import cv2
import numpy as np
import math
import svgwrite
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image
import base64
import os

# ---------------- Utility functions ----------------
def resize_for_processing(img, max_dim=900):
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img, scale

def detect_dots_hough(gray, dp=1.2, min_dist=12, param1=100, param2=15, min_radius=3, max_radius=30):
    img_blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                               param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    if circles is None:
        return []
    circles = np.uint16(np.around(circles[0]))
    pts = [(int(x), int(y)) for x, y, r in circles]
    return pts

def detect_dots_blobs(binary, min_area=8):
    bin_u = (binary > 0).astype('uint8') * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opened = cv2.morphologyEx(bin_u, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened)
    pts = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cx, cy = centroids[i]
            pts.append((int(cx), int(cy)))
    return pts

def estimate_spacing(pts):
    if len(pts) < 2:
        return 30.0
    pts_arr = np.array(pts)
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2).fit(pts_arr)
        dists, _ = nn.kneighbors(pts_arr)
        median = float(np.median(dists[:,1]))
        return max(8.0, median)
    except Exception:
        dists = []
        for i in range(len(pts_arr)):
            d = np.sqrt(np.sum((pts_arr - pts_arr[i])**2, axis=1))
            d = np.sort(d)
            if len(d) > 1:
                dists.append(d[1])
        if len(dists) == 0:
            return 30.0
        return float(np.median(dists))

def group_into_rows(pts, spacing_est):
    if len(pts) == 0:
        return []
    pts_sorted = sorted(pts, key=lambda p: p[1])
    rows = []
    current_row = [pts_sorted[0]]
    last_y = pts_sorted[0][1]
    thresh = spacing_est * 0.6
    for x,y in pts_sorted[1:]:
        if abs(y - last_y) <= thresh:
            current_row.append((x,y))
            last_y = (last_y * (len(current_row)-1) + y) / len(current_row)
        else:
            rows.append(current_row)
            current_row = [(x,y)]
            last_y = y
    rows.append(current_row)
    rows = [sorted(r, key=lambda p: p[0]) for r in rows]
    return rows

def build_rectangular_grid(rows):
    if len(rows) == 0:
        return []
    from collections import Counter
    lengths = [len(r) for r in rows]
    mode_len = Counter(lengths).most_common(1)[0][0]
    grid = []
    for r in rows:
        if len(r) >= mode_len:
            start = (len(r) - mode_len)//2
            grid.append(r[start:start+mode_len])
        else:
            if len(r) == 0:
                grid.append([(0,0)]*mode_len)
            else:
                xs = [p[0] for p in r]
                ys = [p[1] for p in r]
                new_xs = np.linspace(min(xs), max(xs), mode_len)
                new_ys = np.interp(new_xs, xs, ys)
                grid.append([(int(nx), int(ny)) for nx, ny in zip(new_xs, new_ys)])
    return grid

def sample_line_fraction_dark(p1, p2, binary_img, samples=60):
    h, w = binary_img.shape
    x1,y1 = p1; x2,y2 = p2
    xs = np.linspace(x1, x2, samples).astype(int)
    ys = np.linspace(y1, y2, samples).astype(int)
    xs = np.clip(xs, 0, w-1)
    ys = np.clip(ys, 0, h-1)
    vals = binary_img[ys, xs]
    # We expect in our pipeline line pixels are white (255) because of THRESH_BINARY_INV
    ink = (vals > 128).astype(int)
    return float(np.sum(ink)) / samples

def quadratic_bezier(p1, p2, control, steps=80):
    t = np.linspace(0,1,steps)
    x1,y1 = p1; x2,y2 = p2; cx,cy = control
    bx = (1-t)**2 * x1 + 2*(1-t)*t * cx + t**2 * x2
    by = (1-t)**2 * y1 + 2*(1-t)*t * cy + t**2 * y2
    return bx, by

# ---------------- Control point logic ----------------
def control_point_for(p1, p2, spacing):
    x1,y1 = p1; x2,y2 = p2
    midx, midy = (x1+x2)/2.0, (y1+y2)/2.0
    dx, dy = x2-x1, y2-y1
    length = math.hypot(dx, dy)
    if length == 0:
        return (midx, midy)
    nx, ny = dx/length, dy/length
    px, py = -ny, nx
    mag = max(6.0, spacing * 0.28 * (length/spacing))
    dir_sign = 1 if (int(round(x1 + y1)) % 2 == 0) else -1
    return (midx + dir_sign*px*mag, midy + dir_sign*py*mag)

# ---------------- Main inference pipeline (core) ----------------
def reconstruct_kolam_from_image_core(image_path, params=None, max_dim=900):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open {image_path}")
    img_bgr, scale = resize_for_processing(img_bgr, max_dim=max_dim)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, blockSize=31, C=10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    th_closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    pts = detect_dots_hough(gray, dp=1.2, min_dist=12, param1=100, param2=15, min_radius=3, max_radius=30)
    if len(pts) < 6:
        pts = detect_dots_blobs(th_closed, min_area=8)
    if len(pts) == 0:
        pts = detect_dots_blobs(th_closed, min_area=4)
    if len(pts) == 0:
        raise RuntimeError("No dots detected. Try adjusting input quality or parameters.")

    spacing = estimate_spacing(pts)
    rows = group_into_rows(pts, spacing_est=spacing)
    grid = build_rectangular_grid(rows)
    if len(grid) == 0:
        raise RuntimeError("Could not infer grid layout from detected dots.")

    rows_count = len(grid)
    cols_count = len(grid[0])

    connections = []
    binary_for_sampling = th_closed
    for r in range(rows_count):
        for c in range(cols_count):
            p1 = grid[r][c]
            neighbors = []
            if c < cols_count - 1:
                neighbors.append((r, c+1))
            if r < rows_count - 1:
                neighbors.append((r+1, c))
            if (c < cols_count - 1) and (r < rows_count - 1):
                neighbors.append((r+1, c+1))
            if (c > 0) and (r < rows_count - 1):
                neighbors.append((r+1, c-1))
            for nr, nc in neighbors:
                p2 = grid[nr][nc]
                frac = sample_line_fraction_dark(p1, p2, binary_for_sampling, samples=60)
                if frac > 0.25:
                    connections.append(((r,c),(nr,nc)))

    if len(connections) == 0:
        for r in range(rows_count):
            for c in range(cols_count):
                if c < cols_count-1:
                    connections.append(((r,c),(r,c+1)))
                if r < rows_count-1:
                    connections.append(((r,c),(r+1,c)))
                if (r < rows_count-1) and (c < cols_count-1):
                    connections.append(((r,c),(r+1,c+1)))

    meta = {
        "rows": rows_count,
        "cols": cols_count,
        "num_dots": sum(len(r) for r in grid),
        "num_connections_est": len(connections),
        "spacing": spacing,
        "scale": scale
    }

    return {
        "grid": grid,
        "connections": connections,
        "spacing": spacing,
        "meta": meta,
        "thresholded": th_closed,
        "image_size": (img_bgr.shape[1], img_bgr.shape[0])
    }

# ---------------- Export / Draw functions ----------------
def compute_control_point_svg(p1, p2, spacing):
    return control_point_for(p1, p2, spacing)

def sample_quad_bezier_points(p1, p2, ctrl, steps=100):
    bx, by = quadratic_bezier(p1, p2, ctrl, steps=steps)
    return bx, by

def reconstruct_and_export(image_path, params=None, max_dim=900, out_size=(3840,2160)):
    """
    Reconstruct Kolam design from image.
    - out_size: tuple (W,H) for PNG export. Default = 3840x2160 (4K).
    """
    core = reconstruct_kolam_from_image_core(image_path, params=params, max_dim=max_dim)
    grid = core['grid']
    connections = core['connections']
    spacing = core['spacing']
    meta = core['meta']

    rows = len(grid)
    cols = len(grid[0])

    xs = [p[0] for row in grid for p in row]; ys = [p[1] for row in grid for p in row]
    minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
    margin = max(20, int(spacing*0.5))
    w = int(maxx - minx + margin*2)
    h = int(maxy - miny + margin*2)

    # ---------- SVG (vector) ----------
    dwg = svgwrite.Drawing(size=(w, h))
    def t(p): return (p[0]-minx+margin, p[1]-miny+margin)

    for r in range(rows):
        for c in range(cols):
            x,y = t(grid[r][c])
            dwg.add(dwg.circle(center=(x,y), r=2, fill='black'))

    seen = set()
    for conn in connections:
        a,b = conn
        key = tuple(sorted([a,b]))
        if key in seen: continue
        seen.add(key)
        (r1,c1),(r2,c2) = key
        p1 = t(grid[r1][c1]); p2 = t(grid[r2][c2])
        ctrl = control_point_for(p1, p2, spacing)
        path = f"M {p1[0]},{p1[1]} Q {ctrl[0]},{ctrl[1]} {p2[0]},{p2[1]}"
        dwg.add(dwg.path(d=path, stroke='black', fill='none', stroke_width=1.8, stroke_linecap='round'))

    svg_bytes = dwg.tostring().encode('utf-8')

    # ---------- PNG (4K) ----------
    fig, ax = plt.subplots(figsize=(out_size[0]/300, out_size[1]/300), dpi=300)  
    # 300 DPI ensures sharp 4K output
    ax.set_axis_off()
    ax.set_xlim(0, w); ax.set_ylim(h, 0)

    for r in range(rows):
        for c in range(cols):
            x,y = t(grid[r][c])
            ax.plot(x, y, 'ko', markersize=3)

    for key in seen:
        (r1,c1),(r2,c2) = key
        p1 = t(grid[r1][c1]); p2 = t(grid[r2][c2])
        ctrl = control_point_for(p1, p2, spacing)
        bx, by = quadratic_bezier(p1, p2, ctrl, steps=400)
        ax.plot(bx, by, linewidth=1.8, color='black')

    plt.tight_layout(pad=0)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    png_bytes = buf.getvalue()

    return png_bytes, svg_bytes, meta

# ---------------- CLI test ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True)
    parser.add_argument("--out", "-o", default="out.png")
    parser.add_argument("--svg-out", default="out.svg")
    parser.add_argument("--max-dim", type=int, default=900)
    args = parser.parse_args()
    png, svg, meta = reconstruct_and_export(args.image, max_dim=args.max_dim)
    with open(args.svg_out, "wb") as f:
        f.write(svg)
    with open(args.out, "wb") as f:
        f.write(png)
    print("Saved:", args.out, args.svg_out)
    print("Meta:", meta)
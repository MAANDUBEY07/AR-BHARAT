from typing import List, Tuple, Dict, Any
import math

# MATLAB-style kolam renderer from a matrix M using built-in, dot-centered prototypes
# We approximate pt{k} lookup with a richer set of normalized curve generators.
# Key differences vs earlier version:
# - Interpret M indices as DOT locations (i,j), not cell centers
# - Flip rows like MATLAB (M = M(end:-1:1,:)) so the visual matches their coordinate system
# - Provide multiple prototype ids including rotated variants for corner turns

SVG_NS = 'http://www.w3.org/2000/svg'


def _svg_header(width: float, height: float) -> str:
    pad = 20
    return (
        f'<svg xmlns="{SVG_NS}" width="{width + 2*pad}" height="{height + 2*pad}" '
        f'viewBox="{-pad} {-pad} {width + 2*pad} {height + 2*pad}">'
    )


def _svg_footer() -> str:
    return '</svg>'


def _svg_dots(points, r=2, color="#55695a") -> str:
    circles = [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{color}" />' for (x, y) in points]
    return "\n".join(circles)


# --- Prototype primitives (relative to a DOT at (cx, cy)) ---

def _stroke(attrs: Dict[str, Any]) -> str:
    stroke = attrs.get('stroke', '#2458ff')
    sw = attrs.get('stroke_width', 2.0)
    return f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" fill="none"'


def _lemniscate_h(cx: float, cy: float, R: float, stroke: str, stroke_width: float) -> str:
    """Horizontal figure-eight around the dot, extending towards left/right neighbors.
    R ~ spacing*0.48 looks good for continuity.
    """
    r = R
    k = r * 0.56
    # Left lobe start
    x0, y0 = cx - r * 0.95, cy
    # Upper-left curve to top
    x1, y1 = cx - r * 0.95, cy - k
    x2, y2 = cx - r * 0.30, cy - r
    x3, y3 = cx, cy - r
    # Across top to right lobe
    x4, y4 = cx + r * 0.30, cy - r
    x5, y5 = cx + r * 0.95, cy - k
    x6, y6 = cx + r * 0.95, cy
    # Lower-right return
    x7, y7 = cx + r * 0.95, cy + k
    x8, y8 = cx + r * 0.30, cy + r
    x9, y9 = cx, cy + r
    # Lower-left return to start
    x10, y10 = cx - r * 0.30, cy + r
    x11, y11 = cx - r * 0.95, cy + k
    x12, y12 = cx - r * 0.95, cy

    d = (
        f"M {x0:.2f} {y0:.2f} "
        f"C {x1:.2f} {y1:.2f}, {x2:.2f} {y2:.2f}, {x3:.2f} {y3:.2f} "
        f"C {x4:.2f} {y4:.2f}, {x5:.2f} {y5:.2f}, {x6:.2f} {y6:.2f} "
        f"C {x7:.2f} {y7:.2f}, {x8:.2f} {y8:.2f}, {x9:.2f} {y9:.2f} "
        f"C {x10:.2f} {y10:.2f}, {x11:.2f} {y11:.2f}, {x12:.2f} {y12:.2f} Z"
    )
    return f'<path d="{d}" {_stroke({"stroke": stroke, "stroke_width": stroke_width})} />'


def _lemniscate_v(cx: float, cy: float, R: float, stroke: str, stroke_width: float) -> str:
    """Vertical figure-eight (rotate 90°)."""
    # Build using the horizontal but swap x/y around center
    r = R
    k = r * 0.56
    # Top lobe start
    x0, y0 = cx, cy - r * 0.95
    x1, y1 = cx + k, cy - r * 0.95
    x2, y2 = cx + r, cy - r * 0.30
    x3, y3 = cx + r, cy

    x4, y4 = cx + r, cy + r * 0.30
    x5, y5 = cx + k, cy + r * 0.95
    x6, y6 = cx, cy + r * 0.95

    x7, y7 = cx - k, cy + r * 0.95
    x8, y8 = cx - r, cy + r * 0.30
    x9, y9 = cx - r, cy

    x10, y10 = cx - r, cy - r * 0.30
    x11, y11 = cx - k, cy - r * 0.95
    x12, y12 = cx, cy - r * 0.95

    d = (
        f"M {x0:.2f} {y0:.2f} "
        f"C {x1:.2f} {y1:.2f}, {x2:.2f} {y2:.2f}, {x3:.2f} {y3:.2f} "
        f"C {x4:.2f} {y4:.2f}, {x5:.2f} {y5:.2f}, {x6:.2f} {y6:.2f} "
        f"C {x7:.2f} {y7:.2f}, {x8:.2f} {y8:.2f}, {x9:.2f} {y9:.2f} "
        f"C {x10:.2f} {y10:.2f}, {x11:.2f} {y11:.2f}, {x12:.2f} {y12:.2f} Z"
    )
    return f'<path d="{d}" {_stroke({"stroke": stroke, "stroke_width": stroke_width})} />'


def _teardrop(cx: float, cy: float, R: float, theta: float, stroke: str, stroke_width: float) -> str:
    """Corner turn: a rounded teardrop occupying one quadrant around (cx,cy).
    theta is the facing angle in radians (0=+x, pi/2=+y, etc.).
    """
    r = R * 0.9
    # Build a teardrop pointing along +x, then rotate by theta around (cx,cy)
    # Base control points (pointing right)
    pts = [
        (0.0, -0.35*r),
        (0.55*r, -0.35*r), (0.95*r, -0.15*r), (r, 0.0),
        (0.95*r, 0.15*r), (0.55*r, 0.35*r), (0.0, 0.35*r),
        (-0.15*r, 0.20*r), (-0.20*r, 0.10*r), (-0.20*r, 0.0),
        (-0.20*r, -0.10*r), (-0.15*r, -0.20*r), (0.0, -0.35*r),
    ]

    def rot(p):
        x, y = p
        ct, st = math.cos(theta), math.sin(theta)
        return (cx + ct*x - st*y, cy + st*x + ct*y)

    # Map to path commands (M then two C segments + closing)
    P = [rot(pts[0])] + [rot(p) for p in pts[1:]]
    d = (
        f"M {P[0][0]:.2f} {P[0][1]:.2f} "
        f"C {P[1][0]:.2f} {P[1][1]:.2f}, {P[2][0]:.2f} {P[2][1]:.2f}, {P[3][0]:.2f} {P[3][1]:.2f} "
        f"C {P[4][0]:.2f} {P[4][1]:.2f}, {P[5][0]:.2f} {P[5][1]:.2f}, {P[6][0]:.2f} {P[6][1]:.2f} "
        f"C {P[7][0]:.2f} {P[7][1]:.2f}, {P[8][0]:.2f} {P[8][1]:.2f}, {P[9][0]:.2f} {P[9][1]:.2f} "
        f"C {P[10][0]:.2f} {P[10][1]:.2f}, {P[11][0]:.2f} {P[11][1]:.2f}, {P[12][0]:.2f} {P[12][1]:.2f} Z"
    )
    return f'<path d="{d}" {_stroke({"stroke": stroke, "stroke_width": stroke_width})} />'


def _four_petal(cx: float, cy: float, R: float, stroke: str, stroke_width: float) -> str:
    """A cross shape made by overlaying horizontal and vertical lemniscates."""
    return (
        _lemniscate_h(cx, cy, R*0.92, stroke, stroke_width)
        + "\n" +
        _lemniscate_v(cx, cy, R*0.92, stroke, stroke_width)
    )


def _round_loop(cx: float, cy: float, R: float, stroke: str, stroke_width: float) -> str:
    """Rounded diamond loop (kept from previous version)."""
    r = R * 0.82
    d = (
        f"M {cx:.2f} {cy - r:.2f} "
        f"A {r:.2f} {r:.2f} 0 0 1 {cx + r:.2f} {cy:.2f} "
        f"A {r:.2f} {r:.2f} 0 0 1 {cx:.2f} {cy + r:.2f} "
        f"A {r:.2f} {r:.2f} 0 0 1 {cx - r:.2f} {cy:.2f} "
        f"A {r:.2f} {r:.2f} 0 0 1 {cx:.2f} {cy - r:.2f} Z"
    )
    return f'<path d="{d}" {_stroke({"stroke": stroke, "stroke_width": stroke_width})} />'


# Map prototype id -> renderer function (dot-centered)
# Note: id=1 is reserved (blank/no stroke) to match MATLAB behavior where many cells contain 1.
# 2: horizontal lemniscate, 3: vertical lemniscate
# 4/5/6/7: teardrop turns facing 0/90/180/270 degrees respectively
# 8: four-petal (overlay), 9: round loop
# 10/11/12/13: teardrop diagonals 45/135/225/315 degrees
# 14/15/16: extra motifs (reuse smooth curves for variety and continuity)
_PROTOTYPES = {
    2: lambda cx, cy, r, stroke, sw: _lemniscate_h(cx, cy, r, stroke, sw),
    3: lambda cx, cy, r, stroke, sw: _lemniscate_v(cx, cy, r, stroke, sw),
    4: lambda cx, cy, r, stroke, sw: _teardrop(cx, cy, r, 0.0, stroke, sw),
    5: lambda cx, cy, r, stroke, sw: _teardrop(cx, cy, r, math.pi/2, stroke, sw),
    6: lambda cx, cy, r, stroke, sw: _teardrop(cx, cy, r, math.pi, stroke, sw),
    7: lambda cx, cy, r, stroke, sw: _teardrop(cx, cy, r, 3*math.pi/2, stroke, sw),
    8: lambda cx, cy, r, stroke, sw: _four_petal(cx, cy, r, stroke, sw),
    9: lambda cx, cy, r, stroke, sw: _round_loop(cx, cy, r, stroke, sw),
    10: lambda cx, cy, r, stroke, sw: _teardrop(cx, cy, r, math.pi/4, stroke, sw),
    11: lambda cx, cy, r, stroke, sw: _teardrop(cx, cy, r, 3*math.pi/4, stroke, sw),
    12: lambda cx, cy, r, stroke, sw: _teardrop(cx, cy, r, 5*math.pi/4, stroke, sw),
    13: lambda cx, cy, r, stroke, sw: _teardrop(cx, cy, r, 7*math.pi/4, stroke, sw),
    14: lambda cx, cy, r, stroke, sw: _round_loop(cx, cy, r*0.9, stroke, sw),
    15: lambda cx, cy, r, stroke, sw: _lemniscate_h(cx, cy, r*0.9, stroke, sw),
    16: lambda cx, cy, r, stroke, sw: _lemniscate_v(cx, cy, r*0.9, stroke, sw),
}


def draw_kolam_svg_from_matrix(
    M: List[List[int]],
    spacing: float = 40.0,
    color: str = "#2458ff",
    linewidth: float = 2.0,
    show_dots: bool = True,
    mode: str = 'matlab',  # 'matlab' (dot-centered like screenshot) | 'cell' (legacy)
) -> Tuple[str, Dict[str, Any]]:
    """Render an SVG kolam from matrix M.

    matlab mode:
      - Treat M[i][j] as a code for a prototype placed at dot (j, i)
      - Flip rows to match MATLAB visual (top row is highest i)
      - Draw dots at integer lattice (1..n, 1..m)

    cell mode (legacy):
      - Treat M elements as cells and draw at cell centers
    """
    if not M or not isinstance(M, list) or not isinstance(M[0], list):
        raise ValueError("M must be a 2D list")

    m = len(M)
    n = len(M[0])
    for row in M:
        if len(row) != n:
            raise ValueError("M must be a rectangular 2D list")

    if mode not in ('matlab', 'cell'):
        mode = 'matlab'

    # Canvas size — leave one extra spacing for nicer margins
    width = (n + 1) * spacing
    height = (m + 1) * spacing

    parts: List[str] = [_svg_header(width, height)]

    if mode == 'matlab':
        # Dots at integer lattice positions (1..n, 1..m)
        if show_dots:
            dots = [(j * spacing, i * spacing) for i in range(1, m + 1) for j in range(1, n + 1)]
            parts.append('<g id="grid">')
            parts.append(_svg_dots(dots, r=max(1.2, spacing * 0.035)))
            parts.append('</g>')

        # Draw prototypes dot-centered with vertical flip of M
        parts.append('<g id="pattern">')
        R = spacing * 0.48
        for i in range(m):  # screen row
            Mi = M[m - 1 - i]  # flip vertically
            for j in range(n):
                k = Mi[j]
                if isinstance(k, bool):
                    k = 1 if k else 0
                if isinstance(k, (int, float)) and int(k) > 1:
                    cx = (j + 1) * spacing
                    cy = (i + 1) * spacing
                    renderer = _PROTOTYPES.get(int(k))
                    if renderer:
                        parts.append(renderer(cx, cy, R, color, linewidth))
        parts.append('</g>')

    else:
        # Legacy cell-centered drawing
        if show_dots:
            dots = []
            for i in range(m + 1):
                for j in range(n + 1):
                    dots.append((j * spacing, i * spacing))
            parts.append('<g id="grid">')
            parts.append(_svg_dots(dots, r=max(1.4, spacing * 0.04)))
            parts.append('</g>')

        parts.append('<g id="pattern">')
        r = spacing * 0.38
        for i in range(m):
            for j in range(n):
                k = M[i][j]
                if isinstance(k, bool):
                    k = 1 if k else 0
                if isinstance(k, (int, float)) and k > 0:
                    cx = (j + 0.5) * spacing
                    cy = (i + 0.5) * spacing
                    renderer = _PROTOTYPES.get(int(k), _PROTOTYPES[8])
                    parts.append(renderer(cx, cy, r, color, linewidth))
        parts.append('</g>')

    parts.append(_svg_footer())

    metadata: Dict[str, Any] = {
        'matrix_size': {'rows': m, 'cols': n},
        'spacing': spacing,
        'mode': mode,
        'nonzero_cells': sum(1 for row in M for v in row if isinstance(v, (int, float)) and v > 0),
        'prototype_ids': sorted(set(int(v) for row in M for v in row if isinstance(v, (int, float)) and v > 0)),
        'note': 'Rendered from matrix with built-in prototypes (approximate, dot-centered).'
    }

    return "\n".join(parts), metadata
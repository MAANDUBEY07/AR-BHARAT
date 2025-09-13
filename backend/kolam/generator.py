from typing import Tuple, Dict

# Simple square grid dot generator and an exemplar kolam path using symmetry

def _generate_square_grid(rows: int, cols: int, spacing: float):
    dots = []
    for r in range(rows):
        for c in range(cols):
            x = c * spacing
            y = r * spacing
            dots.append((x, y))
    return dots


def _svg_loop_cell(cx: float, cy: float, r: float, stroke="#C2185B", stroke_width=3) -> str:
    # Rounded diamond loop around a cell center using 4 quarter-arc segments
    # Start at top point, go clockwise with arc commands
    d = (
        f"M {cx:.2f} {cy - r:.2f} "
        f"A {r:.2f} {r:.2f} 0 0 1 {cx + r:.2f} {cy:.2f} "
        f"A {r:.2f} {r:.2f} 0 0 1 {cx:.2f} {cy + r:.2f} "
        f"A {r:.2f} {r:.2f} 0 0 1 {cx - r:.2f} {cy:.2f} "
        f"A {r:.2f} {r:.2f} 0 0 1 {cx:.2f} {cy - r:.2f} Z"
    )
    return f'<path d="{d}" stroke="{stroke}" stroke-width="{stroke_width}" fill="none" stroke-linecap="round" stroke-linejoin="round" />'


def _generate_cell_loops(rows: int, cols: int, spacing: float):
    # Create a loop centered in each 2x2 dot cell
    loops = []
    r = spacing * 0.35  # loop radius relative to spacing
    for rr in range(rows - 1):
        for cc in range(cols - 1):
            cx = (cc + 0.5) * spacing
            cy = (rr + 0.5) * spacing
            loops.append(_svg_loop_cell(cx, cy, r))
    return loops


def _svg_header(width: float, height: float) -> str:
    pad = 20
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width+2*pad}" height="{height+2*pad}" viewBox="{-pad} {-pad} {width+2*pad} {height+2*pad}">'


def _svg_footer() -> str:
    return '</svg>'


def _svg_dots(dots, r=2, color="#666") -> str:
    circles = []
    for (x, y) in dots:
        circles.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" />')
    return "\n".join(circles)


def _svg_path(points, stroke="#000", stroke_width=2, fill="none") -> str:
    if not points:
        return ''
    d = f"M {points[0][0]} {points[0][1]} " + " ".join([f"L {x} {y}" for (x, y) in points[1:]])
    return f'<path d="{d}" stroke="{stroke}" stroke-width="{stroke_width}" fill="{fill}" stroke-linecap="round" stroke-linejoin="round" />'


def generate_kolam_svg(rows: int, cols: int, spacing: float, grid_type: str, style: str) -> Tuple[str, Dict]:
    # Only square grid currently
    dots = _generate_square_grid(rows, cols, spacing)
    width = (cols - 1) * spacing
    height = (rows - 1) * spacing

    # Pattern: loop around each cell (more kolam-like)
    loops = _generate_cell_loops(rows, cols, spacing)

    svg_parts = [
        _svg_header(width, height),
        '<g id="grid">',
        _svg_dots(dots),
        '</g>',
        '<g id="pattern">',
        *loops,
        '</g>',
        _svg_footer()
    ]

    metadata = {
        'grid': {
            'type': 'square',
            'rows': rows,
            'cols': cols,
            'spacing': spacing
        },
        'symmetry': {
            'rotational': 4,
            'reflection_axes': 2
        },
        'pattern': {
            'style': style,
            'complexity': 'cell-loops'
        },
        'region_hint': 'Tamil Nadu (demo)'
    }

    return "\n".join(svg_parts), metadata
def explain_kolam(meta: dict) -> str:
    grid = meta.get('grid', {})
    sym = meta.get('symmetry', {})
    pattern = meta.get('pattern', {})

    rows = grid.get('rows')
    cols = grid.get('cols')
    spacing = grid.get('spacing')

    rot = sym.get('rotational')
    axes = sym.get('reflection_axes')

    style = pattern.get('style')
    complexity = pattern.get('complexity')

    parts = []
    parts.append(f"This Kolam uses a {grid.get('type','square')} dot grid with {rows} rows and {cols} columns, spacing {spacing} units.")
    parts.append(f"It exhibits approximately {rot}-fold rotational symmetry and {axes} reflection axes.")
    parts.append(f"Pattern style: {style}; complexity: {complexity}.")
    region = meta.get('region_hint', 'South India')
    parts.append(f"Culturally, similar motifs are common in {region}, often associated with daily thresholds and festive occasions.")
    parts.append("Mathematically, Kolams demonstrate geometry, symmetry, and recursion through repeated motifs around dot lattices.")
    return " ".join(parts)


def chat_about_kolam(meta: dict, message: str) -> str:
    """
    Simple rule-based Q&A about the kolam described by `meta`.
    """
    msg = (message or "").lower().strip()

    grid = meta.get('grid', {})
    sym = meta.get('symmetry', {})
    pattern = meta.get('pattern', {})
    region = meta.get('region_hint', 'South India')

    rows = grid.get('rows')
    cols = grid.get('cols')
    spacing = grid.get('spacing')
    gtype = grid.get('type', 'square')

    rot = sym.get('rotational')
    axes = sym.get('reflection_axes')

    style = pattern.get('style')
    complexity = pattern.get('complexity')

    # Intent: grid details
    if any(k in msg for k in ["grid", "rows", "columns", "cols", "dots", "spacing", "type"]):
        return (
            f"Grid: {gtype} lattice with {rows} rows × {cols} cols, spacing {spacing} units between dots. "
            f"Total dots ≈ {rows*cols if rows and cols else 'N/A'}."
        )

    # Intent: symmetry
    if "symmetry" in msg or "symmetries" in msg or "rotational" in msg or "reflection" in msg or "axis" in msg:
        return (
            f"Symmetry: ~{rot}-fold rotational symmetry with {axes} reflection axes. "
            f"This means it looks similar after {360/max(rot,1):.0f}° rotations and mirror flips across key axes."
        )

    # Intent: style/complexity
    if "style" in msg or "complexity" in msg or "type of kolam" in msg:
        return f"Style: {style}. Complexity: {complexity}. This is a basic demo loop around the outer boundary."

    # Intent: cultural/region meaning
    if any(k in msg for k in ["culture", "cultural", "meaning", "festival", "origin", "region", "where"]):
        return (
            f"Cultural note: Similar motifs are common in {region}. "
            f"Kolams are drawn daily at thresholds and also during festivals like Pongal; they express auspiciousness, symmetry, and skill."
        )

    # Intent: how to draw
    if "how" in msg and ("draw" in msg or "steps" in msg or "make" in msg):
        return (
            "Steps: 1) Mark a square grid of dots. 2) Trace a smooth line around the outer boundary to form a loop. "
            "3) Keep line joints rounded; avoid crossings. 4) Optionally thicken the stroke and add color accents."
        )

    # Intent: export/save
    if any(k in msg for k in ["export", "save", "png", "svg", "download", "raster"]):
        return (
            "You can POST the SVG to /api/export/png (placeholder now) or save the returned SVG as a file. "
            "Frontends can rasterize the SVG to PNG for download."
        )

    # Intent: who/what
    if any(k in msg for k in ["who are you", "what are you", "help", "guide"]):
        return (
            "I'm a rule-based Kolam explainer. Ask about the grid, symmetry, style, region, meaning, or how to draw it."
        )

    # Fallback: provide a concise explanation
    return explain_kolam(meta)
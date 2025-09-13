import sys
import base64
from pathlib import Path
from kolam.image_converter import image_to_kolam_svg

# Usage: python backend/convert_local_image.py "C:\\path\\to\\image.png" [output.svg]
# Converts the image to kolam-style SVG with tuned defaults.

def main():
    if len(sys.argv) < 2:
        print("Usage: python backend/convert_local_image.py <input_image> [output_svg]", file=sys.stderr)
        sys.exit(1)
    in_path = Path(sys.argv[1])
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else in_path.with_suffix('.svg')

    image_bytes = in_path.read_bytes()

    svg, meta = image_to_kolam_svg(
        image_bytes,
        max_contours=10,
        min_area=800,
        mode='vector',
        thresh_block=31,
        thresh_C=3,
        retrieve_mode='tree',
        resample_step=8,
        smooth_alpha=0.8,
    )

    out_path.write_text(svg, encoding='utf-8')
    print(f"Wrote: {out_path}")

if __name__ == '__main__':
    main()
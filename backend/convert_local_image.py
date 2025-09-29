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
        max_contours=4,           # Fewer contours for Kolam clarity
        min_area=200,             # Lower area to capture smaller details
        mode='vector',
        thresh_block=17,          # More sensitive thresholding
        thresh_C=7,               # Higher C for noisy images
        retrieve_mode='external', # Use external contours for Kolam
        resample_step=3,          # More detail in curves
        smooth_alpha=0.5,         # Less smoothing for sharper lines
    )

    out_path.write_text(svg, encoding='utf-8')
    print(f"Wrote: {out_path}")

if __name__ == '__main__':
    main()
"""
Generate a demo Kolam SVG and an explanation, saved to the data/kolams folder.
"""
from pathlib import Path
from kolam.generator import generate_kolam_svg
from chatbot.explainer import explain_kolam

# Parameters
ROWS = 6
COLS = 6
SPACING = 40
GRID_TYPE = "square"
STYLE = "traditional"

# Absolute output paths
SVG_PATH = Path(r"c:\Users\MAAN DUBEY\Desktop\SIH project\data\kolams\demo.svg")
TXT_PATH = Path(r"c:\Users\MAAN DUBEY\Desktop\SIH project\data\kolams\demo_explanation.txt")


def main():
    svg_str, metadata = generate_kolam_svg(ROWS, COLS, SPACING, GRID_TYPE, STYLE)
    explanation = explain_kolam(metadata)

    # Ensure directories exist
    SVG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save SVG
    SVG_PATH.write_text(svg_str, encoding="utf-8")

    # Save explanation
    TXT_PATH.write_text(explanation, encoding="utf-8")

    print(f"Saved SVG to: {SVG_PATH}")
    print(f"Saved explanation to: {TXT_PATH}")


if __name__ == "__main__":
    main()
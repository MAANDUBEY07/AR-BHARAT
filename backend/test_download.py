#!/usr/bin/env python3
"""Test script to debug pattern download"""

from app import app, db, Pattern
from pathlib import Path

def test_pattern_download():
    with app.app_context():
        pattern = Pattern.query.get(14)
        if not pattern:
            print("Pattern 14 not found")
            return
        
        print(f"Testing pattern: {pattern.name}")
        print(f"SVG content length: {len(pattern.svg_content) if pattern.svg_content else 0}")
        
        # Test cairosvg import and conversion
        try:
            import cairosvg
            print("cairosvg imported successfully")
            
            # Test conversion
            try:
                png_data = cairosvg.svg2png(bytestring=pattern.svg_content.encode('utf-8'))
                print(f"PNG conversion successful, size: {len(png_data)} bytes")
            except Exception as e:
                print(f"PNG conversion failed: {e}")
                import traceback
                traceback.print_exc()
                
        except ImportError as e:
            print(f"cairosvg import failed: {e}")
        except OSError as e:
            print(f"cairosvg OS error: {e}")
        except Exception as e:
            print(f"Other error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    test_pattern_download()
#!/usr/bin/env python3
"""Debug script to check pattern 14"""

from app import app, db, Pattern

def check_pattern():
    with app.app_context():
        patterns = Pattern.query.all()
        print(f'Total patterns: {len(patterns)}')
        print(f'First 10 pattern IDs: {[p.id for p in patterns[:10]]}')
        
        pattern_14 = Pattern.query.get(14)
        print(f'Pattern 14 exists: {pattern_14 is not None}')
        
        if pattern_14:
            print(f'Pattern 14 data:')
            print(f'  ID: {pattern_14.id}')
            print(f'  Name: {pattern_14.name}')
            print(f'  Has SVG content: {len(pattern_14.svg_content) > 0 if pattern_14.svg_content else False}')
            print(f'  Filename PNG: {pattern_14.filename_png}')
            print(f'  Filename SVG: {pattern_14.filename_svg}')

if __name__ == '__main__':
    check_pattern()
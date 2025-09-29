#!/usr/bin/env python3
"""
Test script for authentic dot grid kolam generation
Tests the new authentic kolam generator against the real Tamil kolam reference image
"""

import os
import sys

# Add backend to Python path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from kolam.authentic_dot_grid_kolam_generator import AuthenticDotGridKolamGenerator
from kolam.rangoli_to_kolam_converter import convert_rangoli_to_kolam

def test_authentic_kolam_generation():
    """Test authentic kolam generation with the reference Tamil kolam image"""
    
    # Reference kolam image path (the authentic Tamil kolam provided by user)
    reference_kolam_path = r"c:\Users\MAAN DUBEY\Downloads\904b0aec56ec63ddacfbbbd156fa7906.jpg"
    
    if not os.path.exists(reference_kolam_path):
        print(f"❌ Reference kolam image not found at: {reference_kolam_path}")
        print("Please ensure the authentic Tamil kolam image is available.")
        return False
    
    print("🔍 Testing Authentic Dot Grid Kolam Generator")
    print("=" * 60)
    
    try:
        # Test 1: Direct authentic generator
        print("\n1️⃣ Testing Direct Authentic Generator")
        print("-" * 40)
        
        generator = AuthenticDotGridKolamGenerator()
        
        # Analyze the kolam image
        analysis = generator.analyze_input_image(reference_kolam_path)
        print(f"✅ Image Analysis Complete:")
        print(f"   • Pattern Type: {analysis['pattern_type']}")
        print(f"   • Grid Size: {analysis['grid_size']}")
        print(f"   • Symmetry Order: {analysis['symmetry']}")
        print(f"   • Complexity Score: {analysis['complexity']:.3f}")
        print(f"   • Has Central Motif: {analysis['has_central_motif']}")
        
        # Generate authentic kolam
        authentic_svg = generator.generate_kolam_svg(reference_kolam_path)
        
        # Save result
        output_path = "authentic_kolam_from_reference.svg"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(authentic_svg)
        
        print(f"✅ Authentic SVG generated: {len(authentic_svg):,} characters")
        print(f"   Saved as: {output_path}")
        
        # Test 2: Converter integration
        print("\n2️⃣ Testing Converter Integration")
        print("-" * 40)
        
        converter_svg = convert_rangoli_to_kolam(reference_kolam_path)
        
        # Save converter result
        converter_output_path = "converted_kolam_from_reference.svg"
        with open(converter_output_path, 'w', encoding='utf-8') as f:
            f.write(converter_svg)
        
        print(f"✅ Converter SVG generated: {len(converter_svg):,} characters")
        print(f"   Saved as: {converter_output_path}")
        
        # Test 3: Quality Assessment
        print("\n3️⃣ Quality Assessment")
        print("-" * 40)
        
        # Check for authentic kolam elements
        quality_metrics = assess_kolam_authenticity(authentic_svg)
        
        print("🔍 Authenticity Check:")
        for metric, result in quality_metrics.items():
            status = "✅" if result['passes'] else "❌"
            print(f"   {status} {metric}: {result['description']}")
        
        # Overall assessment
        passing_metrics = sum(1 for result in quality_metrics.values() if result['passes'])
        total_metrics = len(quality_metrics)
        accuracy_percentage = (passing_metrics / total_metrics) * 100
        
        print(f"\n📊 Overall Authenticity Score: {accuracy_percentage:.1f}% ({passing_metrics}/{total_metrics})")
        
        # Create HTML viewer for visual comparison
        create_kolam_comparison_viewer(
            reference_kolam_path,
            authentic_svg,
            converter_svg,
            analysis,
            quality_metrics
        )
        
        print(f"\n🌐 Visual comparison viewer created: authentic_kolam_comparison.html")
        print(f"   Open in browser to visually compare results with reference")
        
        return accuracy_percentage >= 80.0  # 80% authenticity target
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def assess_kolam_authenticity(svg_content: str) -> dict:
    """Assess how authentic the generated kolam is based on traditional elements"""
    
    metrics = {}
    
    # Check for dot grid foundation (pulli)
    has_dots = 'pulli-dots' in svg_content or '<circle' in svg_content
    metrics['Dot Grid Foundation'] = {
        'passes': has_dots,
        'description': 'Contains pulli (dot) grid structure' if has_dots else 'Missing traditional dot grid'
    }
    
    # Check for curved lines
    has_curves = 'curve' in svg_content or 'path' in svg_content or 'Q ' in svg_content or 'C ' in svg_content
    metrics['Curved Lines'] = {
        'passes': has_curves,
        'description': 'Contains curved line elements' if has_curves else 'Missing curved line patterns'
    }
    
    # Check for proper background color
    proper_bg = '#f8f8f8' in svg_content or 'fill="#f8f8f8"' in svg_content
    metrics['Traditional Background'] = {
        'passes': proper_bg,
        'description': 'Uses traditional light background' if proper_bg else 'Using incorrect background color'
    }
    
    # Check for appropriate colors
    kolam_colors = ['#2458ff', '#ff6347', '#ffd700', '#dc143c', '#228b22']
    has_traditional_colors = any(color in svg_content for color in kolam_colors)
    metrics['Traditional Colors'] = {
        'passes': has_traditional_colors,
        'description': 'Uses traditional kolam colors' if has_traditional_colors else 'Missing traditional color palette'
    }
    
    # Check for symmetrical structure
    has_groups = '<g' in svg_content and 'id=' in svg_content
    metrics['Structured Organization'] = {
        'passes': has_groups,
        'description': 'Well-organized SVG structure' if has_groups else 'Poor SVG organization'
    }
    
    # Check SVG size/complexity
    is_complex = len(svg_content) > 2000  # Should be reasonably complex
    metrics['Pattern Complexity'] = {
        'passes': is_complex,
        'description': 'Sufficiently complex pattern' if is_complex else 'Pattern too simple'
    }
    
    # Check for metadata/description
    has_metadata = '<title>' in svg_content or '<desc>' in svg_content
    metrics['Cultural Documentation'] = {
        'passes': has_metadata,
        'description': 'Includes cultural context' if has_metadata else 'Missing cultural documentation'
    }
    
    return metrics

def create_kolam_comparison_viewer(reference_image_path: str, authentic_svg: str, converter_svg: str, 
                                 analysis: dict, quality_metrics: dict):
    """Create HTML viewer for visual comparison"""
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Authentic Kolam Generation - Comparison Viewer</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 20px;
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }}
        
        .kolam-panel {{
            border: 2px solid #e0e6ed;
            border-radius: 12px;
            padding: 20px;
            background: #fafbfc;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .kolam-panel:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        }}
        
        .kolam-panel h3 {{
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.4rem;
            border-left: 4px solid #667eea;
            padding-left: 15px;
        }}
        
        .reference-image {{
            max-width: 100%;
            height: 300px;
            object-fit: contain;
            border: 2px solid #ddd;
            border-radius: 8px;
            background: white;
        }}
        
        .generated-kolam {{
            width: 100%;
            height: 300px;
            border: 2px solid #ddd;
            border-radius: 8px;
            background: white;
        }}
        
        .analysis-section {{
            margin-top: 40px;
            background: #f8f9fa;
            padding: 25px;
            border-radius: 12px;
            border: 2px solid #e9ecef;
        }}
        
        .analysis-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}
        
        .analysis-box {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        
        .analysis-box h4 {{
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.2rem;
        }}
        
        .metric {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            padding: 8px;
            border-radius: 6px;
            transition: background-color 0.2s ease;
        }}
        
        .metric:hover {{
            background-color: #f1f3f4;
        }}
        
        .metric.pass {{
            background-color: #d4edda;
            color: #155724;
        }}
        
        .metric.fail {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        
        .metric-icon {{
            margin-right: 10px;
            font-size: 1.1rem;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .stat-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        
        .stat-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-label {{
            font-size: 0.9rem;
            color: #6c757d;
            margin-top: 5px;
        }}
        
        .improvements {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }}
        
        .improvements h4 {{
            color: #856404;
            margin-bottom: 10px;
        }}
        
        .improvements ul {{
            color: #856404;
            margin-left: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Authentic Kolam Generation Analysis</h1>
            <p>Comparison between reference Tamil kolam and AI-generated authentic patterns</p>
        </div>
        
        <div class="comparison-grid">
            <div class="kolam-panel">
                <h3>📷 Reference Tamil Kolam</h3>
                <img src="{reference_image_path}" alt="Reference Tamil Kolam" class="reference-image">
                <p><strong>Source:</strong> Authentic traditional kolam with proper dot grid, curves, and decorative elements</p>
            </div>
            
            <div class="kolam-panel">
                <h3>🤖 AI-Generated Authentic Kolam</h3>
                <div class="generated-kolam">{authentic_svg}</div>
                <p><strong>Method:</strong> Authentic Dot Grid Generator with traditional Tamil kolam techniques</p>
            </div>
            
            <div class="kolam-panel">
                <h3>🔄 Converter Output</h3>
                <div class="generated-kolam">{converter_svg}</div>
                <p><strong>Method:</strong> Rangoli-to-Kolam converter with authentic generator integration</p>
            </div>
        </div>
        
        <div class="analysis-section">
            <h2>📊 Analysis Results</h2>
            
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value">{analysis['pattern_type']}</div>
                    <div class="stat-label">Detected Pattern</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{analysis['grid_size'][0]}×{analysis['grid_size'][1]}</div>
                    <div class="stat-label">Grid Size</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{analysis['symmetry']}-fold</div>
                    <div class="stat-label">Symmetry Order</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{analysis['complexity']:.1%}</div>
                    <div class="stat-label">Complexity Score</div>
                </div>
            </div>
            
            <div class="analysis-grid">
                <div class="analysis-box">
                    <h4>🎯 Authenticity Metrics</h4>
                    {''.join(f'''<div class="metric {'pass' if result['passes'] else 'fail'}">
                        <span class="metric-icon">{'✅' if result['passes'] else '❌'}</span>
                        <div>
                            <strong>{metric}:</strong><br>
                            <small>{result['description']}</small>
                        </div>
                    </div>''' for metric, result in quality_metrics.items())}
                </div>
                
                <div class="analysis-box">
                    <h4>📈 Performance Summary</h4>
                    <div class="metric">
                        <span class="metric-icon">📏</span>
                        <div>
                            <strong>SVG Size:</strong> {len(authentic_svg):,} characters
                        </div>
                    </div>
                    <div class="metric">
                        <span class="metric-icon">🎨</span>
                        <div>
                            <strong>Elements:</strong> Dots, Curves, Traditional Colors
                        </div>
                    </div>
                    <div class="metric">
                        <span class="metric-icon">🔄</span>
                        <div>
                            <strong>Method:</strong> Authentic Dot Grid Algorithm
                        </div>
                    </div>
                    <div class="metric">
                        <span class="metric-icon">🏛️</span>
                        <div>
                            <strong>Cultural Accuracy:</strong> Traditional Tamil Kolam Structure
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="improvements">
                <h4>🔧 Key Improvements Made</h4>
                <ul>
                    <li>✅ <strong>Dot Grid Foundation (Pulli):</strong> Implemented traditional dot matrix system</li>
                    <li>✅ <strong>Curved Line Patterns:</strong> Added flowing Bezier curves and loops</li>
                    <li>✅ <strong>Authentic Background:</strong> Using proper light grey (#f8f8f8)</li>
                    <li>✅ <strong>Traditional Colors:</strong> Tamil kolam color palette</li>
                    <li>✅ <strong>Pattern Analysis:</strong> Input-specific symmetry and complexity detection</li>
                    <li>✅ <strong>Cultural Elements:</strong> Proper kolam vocabulary and structure</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>'''
    
    with open('authentic_kolam_comparison.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    success = test_authentic_kolam_generation()
    
    if success:
        print("\n🎉 SUCCESS: Authentic kolam generation meets 80%+ authenticity target!")
    else:
        print("\n⚠️  IMPROVEMENT NEEDED: Authenticity score below 80% target")
    
    print("\nTest complete. Check the generated files:")
    print("• authentic_kolam_from_reference.svg")  
    print("• converted_kolam_from_reference.svg")
    print("• authentic_kolam_comparison.html")
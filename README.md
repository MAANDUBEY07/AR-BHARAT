# Kolam Intelligence 2.0 (AI + AR + Culture)

## Overview
A modular system to generate, explain, and showcase Kolam designs with AI, AR, and cultural knowledge.

## Modules (MVP)
1. Dot Grid Module: basic square grid generator.
2. Kolam Generator: simple rect loop + dots; SVG export.
3. AI Chatbot: rule-based explanation from metadata.
4. Gallery Dashboard: placeholder (to be built with React + Tailwind).
5. AR Experience: placeholder (Unity/ARCore/ARKit).
6. Generative Recommender + Dataset: scaffolding only.

## Quick Start
1. Create venv and install deps:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Run backend:
   ```powershell
   python .\backend\app.py
   ```
3. Test endpoints:
   - Health: `GET http://localhost:5000/api/health`
   - Generate SVG: `POST http://localhost:5000/api/generate` JSON body
     ```json
     {"rows":6, "cols":6, "spacing":40, "grid_type":"square", "style":"traditional"}
     ```
   - Explain: `POST http://localhost:5000/api/explain` with `metadata` from generate.

## Next Steps
- Add triangular/circular grids and real kolam path algorithms.
- Implement PNG export using `cairosvg`.
- Build React + Tailwind frontend for gallery and interactive grid.
- Start dataset schema (SVG + JSON annotations by region, festival, symmetry).
- Plan AR app prototype (Unity): render SVG->texture onto plane using ARFoundation.

## License
MIT (add as needed)
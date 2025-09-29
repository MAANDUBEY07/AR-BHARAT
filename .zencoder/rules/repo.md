# Repository Overview

Project: Kolam Intelligence 2.0 (AI + AR + Cultural Knowledge System)

Structure:
- backend/: Flask API for kolam generation, chatbot endpoints, model services
  - kolam/: generators and grid tools
  - chatbot/: rule-based explainer for math + culture
  - models/: recommender and future training code
- frontend/: HTML/CSS/JS app for gallery dashboard with heritage-themed UI
- data/kolams/: dataset (SVG/JSON) and annotations
- ar/: AR prototypes/placeholders (Unity/ARCore/ARKit export targets)
- docs/: specs, README, API docs

Run order:
1) Backend Flask app -> serves API
2) Frontend -> consumes API for gallery and generator UI
3) AR (later) -> mobile experience consuming generated SVG/PNG

targetFramework: Playwright

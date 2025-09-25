# Comprehensive UX/UI Prompt for “Kolam (Rangoli) AI Agent” — Homepage & Discovery Redesign

Use this prompt to redesign the web app into a visually rich, highly engaging platform with a Netflix/YouTube-like discovery experience while preserving cultural authenticity and clarity.

## Objective
Transform the current functional UI into a cinematic, discovery-first experience that:
- Showcases Kolam designs in a dark, elegant interface.
- Prioritizes exploration, personalization, and retention.
- Streamlines conversion into a focused, delightful flow.
- Scales across devices with excellent readability and performance.

## Visual Tone and Aesthetic
- Dark, immersive theme to make white Kolam lines pop (near-black background).
- Culturally reverent: natural pigment accents (turmeric, indigo) used sparingly.
- Clean typography: readable body (16px+), generous spacing, high contrast.
- Motion as storytelling: seamless transitions and living thumbnails.

## Homepage & Discovery Engine
Design the homepage as a dynamic discovery surface with large, scrollable modules:
- Sections (stacked vertically, full-width):
  - Trending Kolams
  - Featured Artists
  - Newly Converted Designs
  - Kolams of the Day
- Layout:
  - Grid-based, edge-to-edge rows with large, high-quality thumbnails (2–6 columns responsive).
  - Each tile shows: thumbnail, title, creator/artist, likes.
  - Hover/Focus: subtle zoom-in (1.03x), soft glow border, animated line trace around the tile.
  - Infinite scroll with skeleton loaders; lazy-load images; prefetch next row.
- Personal rails:
  - “Because you liked Floral Kolams”
  - “More Hexagonal Grid Designs”
  - “For You” (based on symmetry/complexity preferences)

## Visual Design & Animation
- Palette:
  - Background: #0B0C0E (near-black)
  - Line: #F8F8F6 (rice white)
  - Accents: #D97706 (turmeric), #4F46E5 (indigo)
- Thumbnails:
  - Elevation on hover; outline tracing animation (SVG stroke-dashoffset)
- Transitions:
  - Cinematic card-to-detail: scale up clicked tile into detail view; keep Kolam focal.
- Micro-animations:
  - Hover dots appear/fade; line “breathes” subtly
  - Buttons: loop ripple; progress bars with line shimmer

## Enhanced User Flow & Readability
- Conversion Experience:
  - Full-screen, focused modal/page replacing parameter-heavy UI.
  - Hero CTA: “Upload Rangoli to Convert” + minimal creative inputs (Symmetry, Complexity, Grid Type, Grid Size).
  - Progress: dot-grid → line-formation animation.
- Post-Conversion Details:
  - Full-screen Kolam (SVG), prominent actions: Save, Share, Like, View in AR, Download.
  - Metadata: symmetry, complexity, grid, creation date, artist/user.
  - Below fold: “Similar Kolams” recommendations.
- Readability & Zoom:
  - Body 16px+, headings 24–36px, line-height 1.5–1.7.
  - Viewport meta for consistent scaling; ensure inputs are 16px+ to avoid iOS auto-zoom.
  - Prominent descriptive text block near top of Details page.

## User Engagement & Retention
- Gamification:
  - Daily Kolam Challenge, streaks, badges, leaderboards.
- Social Features:
  - Save to My Gallery, Like, Comment, Share (deep links + previews), Follow Artists.
- Personalization:
  - Track preferences (symmetry, complexity, theme, grid type).
  - Re-rank homepage rails for “For You”; interactive thumbs up/down to refine.

## Information Architecture
- Top Nav: Home, Discover, Create, My Gallery, Learn, Profile.
- Home: Personalized and curated rails.
- Discover: Filterable catalog by symmetry/theme/complexity/grid.
- Create: Conversion and Generative Recommender.
- Details: Full-screen Kolam with actions and recommendations.
- Learn: Cultural context, tutorials, community.

## Accessibility & Performance
- WCAG AA+, keyboard navigable, focus-visible styles.
- Respect reduced motion; provide alt text; semantic structure.
- Responsive images, lazy-load, skeletons, SVG caching.

## Acceptance Criteria
- Discovery-first homepage with multiple animated rails.
- Focused conversion flow with progress animation.
- Immersive Details page with clear actions.
- Social, gamification, personalization visible at MVP.
- Smooth, cinematic transitions; excellent readability; no auto-zoom issues.

## Deliverables
- High-fidelity mockups (Home, Discover, Convert, Details, My Gallery).
- Motion specs for hover and route transitions.
- Component library (React + Tailwind): Cards, Rails, Modals, Buttons, Progress, Thumbnail, Detail Header, Social Actions.
- Prototype: homepage interactions, conversion flow, details transition.
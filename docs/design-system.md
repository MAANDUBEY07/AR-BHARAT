# Kolam (Rangoli) AI Agent — Frontend Design System

Technology: React + Tailwind CSS
Core principle: Express the mathematical beauty, symmetry, and cultural elegance of Kolam.

---

## Grid System
Inspired by Kolam dot grids, the layout grid uses a symmetric, responsive system that “snaps” components to harmonious positions.

- Structure
  - Root container: centered column with max width and generous breathing room.
  - Symmetric lanes: 12-column CSS grid, with mirrored areas for left/right balance.
  - Dot rhythm: spacing scale (2, 4, 6, 8, 12, 16, 24) used consistently to echo dot spacing.

- Tailwind utilities
  - Container: `mx-auto max-w-7xl px-4 sm:px-6 lg:px-8`
  - Grid: `grid grid-cols-12 gap-6 md:gap-8`
  - Symmetric blocks: use `col-span-*` with mirrored pairs (e.g., 3-6-3)

- Example usage (React + Tailwind)
```tsx
export function SymmetricLayout({ left, center, right }: { left: React.ReactNode; center: React.ReactNode; right: React.ReactNode }) {
  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
      <div className="grid grid-cols-12 gap-6 md:gap-8 items-start">
        <aside className="col-span-12 md:col-span-3 order-2 md:order-1">{left}</aside>
        <main className="col-span-12 md:col-span-6 order-1 md:order-2">{center}</main>
        <aside className="col-span-12 md:col-span-3 order-3">{right}</aside>
      </div>
    </div>
  );
}
```

- Kolam Dot Grid helper (for decorative symmetrical scaffolding)
```tsx
export function DotBackdrop() {
  // subtle dot pattern as background grid
  return (
    <div className="pointer-events-none fixed inset-0 z-0"
      style={{ backgroundImage: `radial-gradient(currentColor 1px, transparent 1px)`,
               backgroundSize: '24px 24px', color: 'rgba(0,0,0,0.06)'}} />
  );
}
```

---

## Color Palette
Reflects white rice flour, natural pigments, and a minimal black-and-white base.

- Primary palette
  - **Rice White**: #FAFAF9 (backgrounds)
  - **Kolam Black**: #111827 (primary text, Kolam line art)
  - **Slate Ink**: #374151 (secondary text)

- Accent palette (natural pigments)
  - **Turmeric Gold**: #D97706 (primary CTA / highlights)
  - **Indigo Dye**: #4F46E5 (secondary CTA / links)
  - **Henna Red**: #B91C1C (warnings / destructive)
  - **Leaf Green**: #16A34A (success / positive)

- Usage
  - Background: Rice White
  - Components: white panels with subtle shadows
  - CTAs: Turmeric Gold solid; Secondary: Indigo outline
  - Accents: use sparingly to preserve focus on the Kolam art

---

## Typography
- Primary (UI/tech): **Inter** or **IBM Plex Sans**
  - Clean, modern, excellent for forms and technical labels
- Secondary (brand/heading): **Fraunces** (serif with elegance) or **Marcellus**
  - Suggests tradition without losing readability

- Tailwind config suggestion
```ts
// tailwind.config.js (excerpt)
module.exports = {
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        display: ['Fraunces', 'ui-serif', 'Georgia', 'serif'],
      },
      colors: {
        rice: '#FAFAF9',
        kolam: '#111827',
        turmeric: '#D97706',
        indigoDye: '#4F46E5',
        henna: '#B91C1C',
        leaf: '#16A34A',
      }
    }
  }
}
```

---

## Component Design
Kolam motifs: continuous lines, loops, circles, and symmetry.

- Buttons
  - Soft rounded corners (loop feel), inner glow on focus
  - Primary “Convert” in Turmeric; secondary outline in Indigo
```tsx
export function KButton({ children, variant = 'primary', ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary'|'secondary'|'ghost' }) {
  const base = 'inline-flex items-center justify-center rounded-full px-4 py-2 font-medium transition-all duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2';
  const styles = {
    primary: 'bg-turmeric text-white hover:brightness-110 focus-visible:ring-turmeric',
    secondary: 'border border-indigoDye text-indigoDye hover:bg-indigoDye hover:text-white focus-visible:ring-indigoDye',
    ghost: 'text-kolam hover:bg-black/5 focus-visible:ring-slate-400',
  }[variant];
  return <button className={`${base} ${styles}`} {...props}>{children}</button>;
}
```

- Inputs
  - Subtle rounded corners, faint inner stroke (like etched sand)
```tsx
export function KInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input {...props} className="w-full rounded-lg border border-black/10 bg-white px-3 py-2 shadow-[inset_0_1px_2px_rgba(0,0,0,0.03)] focus:outline-none focus:ring-2 focus:ring-indigoDye/30" />
  );
}
```

- Cards
  - Symmetrical padding, faint border, optional top/bottom line motif
```tsx
export function KCard({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-black/10 bg-white p-5 shadow-sm">
      <div className="relative">
        {/* Top loop motif */}
        <div className="absolute -top-3 left-1/2 h-6 w-6 -translate-x-1/2 rounded-full border border-black/10 bg-white" />
        {children}
      </div>
    </div>
  );
}
```

- Dividers with dot motif
```tsx
export function KDivider() {
  return (
    <div className="flex items-center justify-center gap-2 py-3 text-black/20">
      <span className="h-1 w-1 rounded-full bg-current" />
      <span className="h-1 w-1 rounded-full bg-current" />
      <span className="h-1 w-1 rounded-full bg-current" />
    </div>
  );
}
```

---

## Micro-animations
- Convert action
  - Dots emerge (scale/opacity) in a radial pattern
  - A single line animates along a path to form the Kolam (stroke-dashoffset animation)

- Button hover
  - Subtle loop ripple (box-shadow pulse)

- Progress
  - Linear bar with easing; optional “flowing line” shimmer

- Example line draw animation (CSS on SVG path)
```css
.kolam-path {
  stroke-dasharray: 1200; /* Adjust to path length */
  stroke-dashoffset: 1200;
  animation: draw 2.8s ease-in-out forwards;
}
@keyframes draw {
  to { stroke-dashoffset: 0; }
}
```

---

## Code Structure
Organize for reuse and symmetry at every layer.

```
src/
  components/
    primitives/      // Button, Input, Card, Divider, Modal, Tooltip
    layout/          // SymmetricLayout, GridSection, DotBackdrop
    kolam/           // DotGridPreview, KolamCanvas, KolamControls, ProgressBar
    feedback/        // Toasts, Loaders, EmptyStates
  pages/
    Home.tsx         // Gallery, Current, Controls
    Create.tsx       // Generative Recommender UI
    AR.tsx           // WebXR viewer
  hooks/
    useKolamApi.ts   // API calls, SWR/React Query (fetch convert/generate)
    useProgress.ts   // determinate/indeterminate progress
  styles/
    globals.css
    animations.css   // keyframes for kolam draw, dot pop
  lib/
    svg.ts           // helpers to style/animate SVG paths
    symmetry.ts      // helpers to mirror layouts if needed
```

- Principles
  - Primitives are Kolam-styled, highly reusable
  - Layout components ensure symmetry and predictable spacing
  - Kolam-specific components own visualization and animations
  - Hooks isolate data fetching and state; pages compose features

---

## Example: Generative Recommender Panel
```tsx
export function RecommenderPanel() {
  return (
    <KCard>
      <h2 className="font-display text-2xl text-kolam">Generative Recommender</h2>
      <KDivider />
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-sm text-slate-600 mb-1">Symmetry</label>
          <select className="w-full rounded-lg border border-black/10 px-3 py-2">
            <option>Rotational</option>
            <option>Bilateral</option>
            <option>Asymmetric</option>
          </select>
        </div>
        <div>
          <label className="block text-sm text-slate-600 mb-1">Complexity</label>
          <select className="w-full rounded-lg border border-black/10 px-3 py-2">
            <option>Simple</option>
            <option>Medium</option>
            <option>Complex</option>
          </select>
        </div>
        <div>
          <label className="block text-sm text-slate-600 mb-1">Theme</label>
          <select className="w-full rounded-lg border border-black/10 px-3 py-2">
            <option>Floral</option>
            <option>Geometric</option>
            <option>Animal</option>
            <option>Mythological</option>
          </select>
        </div>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-4">
        <KInput placeholder="Rows (e.g., 7)" />
        <KInput placeholder="Cols (e.g., 7)" />
      </div>
      <div className="mt-6 flex items-center justify-between">
        <KButton>Suggest Designs</KButton>
        <KButton variant="secondary">Surprise Me</KButton>
      </div>
    </KCard>
  );
}
```

---

## Accessibility & Internationalization
- High contrast (WCAG AA)
- Respect reduced motion preference (disable line-draw animation)
- Semantic HTML, ARIA where needed
- i18n-ready via message catalogs

---

## Theming Notes
- Light default: Rice White background; Dark mode optional with softer contrast (off-black bg, warm white lines)
- Keep color accents minimal to let Kolam art remain focal
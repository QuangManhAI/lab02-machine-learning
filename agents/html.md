# Agent Guide: Building Standalone HTML Analysis Reports

## Goal

Generate a self-contained, responsive, professional-looking HTML report that presents analysis results — EDA visualizations, model metrics, comparisons, and conclusions — to stakeholders. Every report must be **stylable without external dependencies** (no CDN, no Bootstrap), **print-friendly**, and **consistent** across projects.

## Core Principles

1. **One file, zero dependencies.** All CSS is inline in `<style>`. No external fonts, no CDN, no JavaScript (unless interactive features are explicitly requested).
2. **Responsive by default.** The layout collapses gracefully on viewports < 700px.
3. **Section-based structure.** Every distinct analysis is a `.section` block with a unique `id` and a consistent inner layout.
4. **Visual hierarchy.** Header → TOC → sections → footer. Each section has a colored header bar, body content, figures, and optional insight boxes.
5. **Data-driven, not decorative.** Every visual element serves a purpose. Insight boxes highlight actionable takeaways. Tables present precise numbers.

## HTML Structure

### Document Skeleton

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Project Name — Analysis Type</title>
  <style>/* all CSS here */</style>
</head>
<body>

<div class="header">
  <h1>Title</h1>
  <p>Subtitle / description paragraph.</p>
  <div class="meta">Optional metadata line.</div>
</div>

<div class="container">

<div class="toc">
  <h2>Table of Contents</h2>
  <ol>
    <li><a href="#s1">Section 1</a></li>
    <li><a href="#s2">Section 2</a></li>
    <!-- ... -->
  </ol>
</div>

<!-- Section N -->
<div class="section" id="sN">
  <div class="section-header">
    <h2><span class="step">N.</span> Section Title</h2>
  </div>
  <div class="section-body">
    <p>Description paragraphs.</p>
    <div class="figure">
      <img src="reports/.../image.png" alt="Description">
      <div class="caption"><strong>Figure N:</strong> Caption text.</div>
    </div>
    <div class="insight"><strong>Key takeaway:</strong> Actionable guidance.</div>
  </div>
</div>

<div class="footer">
  Generated from ... &bull; N sections &bull; tags
</div>

</div>
</body>
</html>
```

### Required Components

| Component | CSS Class / ID | Purpose |
|-----------|---------------|---------|
| Page header | `.header` | Title, subtitle, metadata. Dark gradient background. |
| Table of Contents | `.toc` | Linked list of all sections. 2-column layout on wide screens. |
| Section | `.section` | Wrapper for one analysis block. White card with shadow. |
| Section header | `.section-header` | Colored left border + light background. Contains `<h2>`. |
| Section body | `.section-body` | Inner padding container for text, figures, tables. |
| Figure block | `.figure` | Centers an image with a caption below. |
| Insight box | `.insight` | Yellow left-border box for key findings or decisions. |
| Two-column grid | `.two-col` | CSS grid, 2 equal columns, collapses to 1 on mobile. |
| Footer | `.footer` | Centered muted text. |

## CSS Conventions

### CSS Variables (Theme)

```css
:root {
  --primary: #2c3e50;        /* dark headings */
  --accent: #3498db;         /* link color, active elements */
  --accent-light: #ebf5fb;   /* section header background */
  --bg: #f8f9fa;             /* page background */
  --card: #ffffff;            /* section background */
  --text: #2c3e50;           /* body text */
  --muted: #7f8c8d;          /* secondary text, captions */
  --border: #dee2e6;         /* light borders */
}
```

- Use `var(--primary)` consistently. Never hardcode colors.
- The accent color scheme should match the project (use blue `#3498db` as default; override `--accent` for different themes).

### Typography

- Font stack: `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif`
- Body: `font-size: 0.95rem`, `line-height: 1.6`
- H1: `2.2rem`, H2 in sections: `1.25rem`
- Captions: `0.85rem`, italic, muted color
- Badges / small labels: `0.7rem`

### Section Card

```css
.section {
  background: var(--card);
  border-radius: 10px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  margin-bottom: 1.8rem;
  overflow: hidden;
}
.section-header {
  background: var(--accent-light);
  padding: 1.2rem 2rem;
  border-bottom: 2px solid var(--accent);
}
.section-body { padding: 1.5rem 2rem 2rem; }
```

- No border-radius on the header bottom (overflow hidden on parent clips it).
- The `border-bottom: 2px solid var(--accent)` creates the accent line under each section header.

### Image Handling

```css
.figure {
  margin: 1.5rem 0 0.5rem;
  text-align: center;
}
.figure img {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  border: 1px solid var(--border);
  box-shadow: 0 1px 6px rgba(0,0,0,0.08);
}
```

- All images are responsive (`max-width: 100%`).
- Every `.figure` block must contain exactly one `<img>` and one `.caption`.
- Figure captions are numbered sequentially across the report ("Figure 1", "Figure 2", etc.).

### Insight Box

```css
.insight {
  background: #fff3cd;
  border-left: 4px solid #ffc107;
  padding: 0.8rem 1.2rem;
  border-radius: 0 6px 6px 0;
  margin: 0.8rem 0;
  font-size: 0.92rem;
}
.insight strong { color: #856404; }
```

- Use for: decision rules, key findings, actionable takeaways, warnings.
- The yellow color draws attention without being alarmist.
- Keep insight text concise — 1–3 sentences.

### Header

```css
.header {
  background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
  color: #fff;
  padding: 3rem 2rem;
  text-align: center;
}
```

- Gradient from `--primary` to `--accent`.
- Large padding top/bottom. Centered text.

### Table of Contents

```css
.toc ol { columns: 2 280px; column-gap: 2rem; }
@media (max-width: 700px) { .toc ol { columns: 1; } }
```

- 2-column layout with `columns` CSS property (not flex/grid).
- Collapses to single column on mobile.

### Two-Column Grid

```css
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
@media (max-width: 700px) { .two-col { grid-template-columns: 1fr; } }
```

- Use for side-by-side images (e.g., two related plots).
- Gap of 1.5rem for visual breathing room.

### Tables

```css
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin: 0.8rem 0; }
th, td { padding: 0.5rem 0.8rem; text-align: left; border-bottom: 1px solid var(--border); }
th { background: var(--accent-light); font-weight: 600; }
```

- Full-width tables with subtle horizontal borders.
- Header row gets the accent-light background.
- No vertical borders — keep it clean.

## Content Guidelines per Section

### Section Body
1. **Opening paragraph** — what this analysis shows and why it matters.
2. **Figure(s)** — one or two images with captions.
3. **Interpretation** — how to read the chart/table. What patterns are expected.
4. **Insight box** — specific decision rule, threshold, or actionable finding.

### Naming and Numbering
- Section IDs: `s1`, `s2`, ..., `sN`.
- Step labels in headers: `1.`, `2.`, etc. using `<span class="step">`.
- Figures: incrementing "Figure N:" at the start of each caption.
- TOC entries must match section titles exactly.

### Image Paths
- All images stored under `reports/EDA/`.
- Path format: `reports/EDA/<filename>.png`.
- Alt text should be descriptive (not just the filename).

## Mobile Breakpoint

```css
@media (max-width: 700px) {
  .two-col { grid-template-columns: 1fr; }
  .toc ol { columns: 1; }
  .header h1 { font-size: 1.6rem; }
}
```

The single breakpoint at 700px handles the common tablet/mobile case. Only three things change: grid collapses, TOC collapses, and H1 shrinks.

## Checklist for Quality

Before finishing, verify:

- [ ] **Zero external dependencies** — no CDN, no JS, no external fonts?
- [ ] All `<style>` is in the `<head>` (not inline on elements)?
- [ ] Every `.section` has a unique `id` and matching TOC entry?
- [ ] All image paths use `reports/EDA/` and files actually exist?
- [ ] Every `.figure` has exactly one `<img>` + one `.caption`?
- [ ] Figure numbers are sequential (1, 2, 3, ...) without gaps?
- [ ] At least one `.insight` box per section (or per analysis)?
- [ ] Table headers are styled with `background: var(--accent-light)`?
- [ ] The responsive `<meta>` viewport tag is present?
- [ ] The footer includes project name, section count, and tags?

## Example: Section with Figure + Insight

```html
<!-- 3 -->
<div class="section" id="s3">
  <div class="section-header">
    <h2><span class="step">3.</span> Feature Distributions</h2>
  </div>
  <div class="section-body">
    <p>Histograms of all numeric features reveal their underlying distributions — several are <strong>right-skewed</strong> (<code>total_rooms</code>, <code>total_bedrooms</code>, <code>population</code>).</p>
    <div class="figure">
      <img src="reports/EDA/eda_02_feature_distributions.png" alt="Feature Distributions">
      <div class="caption"><strong>Figure 2:</strong> Histogram grid of all numeric features.</div>
    </div>
    <div class="insight"><strong>What to watch:</strong> Skewed features may benefit from log transformation for linear models. Tree-based models are invariant to monotonic transformations.</div>
  </div>
</div>
```

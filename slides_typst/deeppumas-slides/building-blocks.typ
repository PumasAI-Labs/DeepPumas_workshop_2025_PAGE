// ──────────────────────────────────────────────────────────────────────────
//  deeppumas-slides — building blocks
//  Small reusable visual pieces that compose inside `content-slide` bodies.
//  Sizes are tuned for Touying's default 16:9 paper (841.89pt × 473.56pt);
//  on a larger explicit-pixel page they'll appear proportionally smaller.
// ──────────────────────────────────────────────────────────────────────────

// ─── Palette (mirrors brand book) ────────────────────────────────────────

#let dp-purple       = rgb("#6B00C7")
#let dp-purple-deep  = rgb("#250044")
#let dp-purple-mid   = rgb("#AE7FD6")
#let dp-purple-tint  = rgb("#F1EAF9")
#let dp-purple-50    = rgb("#EEEDF0")
#let dp-white        = white
#let dp-gray-100     = rgb("#E3E3E3")
#let dp-gray-500     = rgb("#9B97A5")
#let dp-gray-700     = rgb("#6C6874")

// ─── Eyebrow labels ──────────────────────────────────────────────────────

#let eyebrow(body, color: dp-purple) = text(
  size: 9pt,
  weight: 700,
  tracking: 0.18em,
  fill: color,
  upper(body),
)

// `sub-eyebrow` controls its OWN trailing gap via `block(below: ..., spacing: 0pt)`
// — so callers should NOT add a `#v(...)` after it. Variants:
//   variant: "text"  (default) → 8pt gap, suited for paragraph or math
//   variant: "list"            → 14pt gap, suited for bullet / numbered / tagged lists
// Override either with `below: 12pt` (or any length) for exact control.
//
// Size: bumped from 8pt → 11pt (2026-05-24) for projector legibility — the
// 8pt default was hard to read past the front row. Override with `size:`
// if you want the old behaviour locally.
#let sub-eyebrow(
  body,
  color: dp-purple-mid,
  variant: "text",
  size: 11pt,
  below: auto,
) = {
  let gap = if below != auto {
    below
  } else if variant == "list" {
    14pt
  } else {
    8pt
  }
  // Render the text in a 0-spacing block, then append an explicit
  // vertical gap with `v(...)`. Why not `block(below: gap)`? In Typst,
  // a block's `below:` is between-block margin; it is honoured in
  // normal vertical flow but *not* counted toward a grid cell's height
  // (the cell wraps the block tightly). Using `v(gap, weak: false)`
  // means the gap shows up as part of the rendered content, so it
  // adds height inside a grid cell AND produces the same visible gap
  // in vertical flow.
  block(spacing: 0pt,
    text(
      size: size,
      weight: 700,
      tracking: 0.12em,
      fill: color,
      upper(body),
    ))
  v(gap, weak: false)
}

// Math-aware eyebrow. Use when the eyebrow text needs to embed math
// expressions that should *not* be uppercased (e.g. `$S_"arm"$` — the
// subscript "arm" must stay lowercase). Pass an alternating sequence of
// strings (tracked-uppercased) and content values (rendered as-is):
//
//   sub-eyebrow-segments(
//     "SURVIVAL DISTRIBUTION", $S_"arm"$, "OF A STUDY ARM",
//   )
//
// The string segments are uppercased internally — you can pass them in
// any case, but writing them ALL CAPS at the call site makes the visual
// result obvious in the source.
#let sub-eyebrow-segments(
  ..segments,
  color: dp-purple-mid,
  size: 11pt,
  below: 8pt,
  // 0.5em gives a comfortable visible word-space between text and math
  // segments at the standard 11pt eyebrow size — `tracking: 0.12em` on
  // the text segments doesn't contribute trailing kerning, so a smaller
  // inter-gap leaves the math segments touching the adjacent text.
  inter-gap: 0.5em,
) = {
  // See `sub-eyebrow` for why the trailing gap is rendered with
  // `v(...)` instead of `block(below: ...)` — grid-cell friendly.
  block(spacing: 0pt, {
    let segs = segments.pos()
    for (i, s) in segs.enumerate() {
      if type(s) == str {
        text(size: size, weight: 700, tracking: 0.12em, fill: color, upper(s))
      } else {
        text(size: size, weight: 700, fill: color, s)
      }
      if i + 1 < segs.len() { h(inter-gap) }
    }
  })
  v(below, weak: false)
}

// ─── Generic card ────────────────────────────────────────────────────────

#let card(
  body,
  fill: dp-white,
  stroke: 0.7pt + dp-gray-100,
  radius: 9pt,
  inset: 12pt,
  width: 100%,
) = block(
  width: width,
  fill: fill,
  stroke: stroke,
  radius: radius,
  inset: inset,
  body,
)

// ─── Outline card (transparent rounded container) ───────────────────────
//
// A rounded box with a header band and arbitrary body — like info-card but
// with NO background fill, so it reads as a visual "outline" used to group
// related items on a slide without competing visually with the body content.
//
// Variants of `stroke-weight` let you nest: use the heavier "outer" weight
// for the parent, the lighter "inner" weight for children — the hierarchy
// stays readable even when boxes are nested two levels deep.
//
//   #outline-card(header: [Why stop early])[
//     - clinical trials are expensive
//     - participants may fare worse than on SoC
//   ]

// header-style options:
//   "eyebrow"     — bold + small-caps + tracked, 11pt (matches sub-eyebrow)
//   "eyebrow-sm"  — same as "eyebrow" but at 9pt (for nested inner cards)
//   "bold"        — bold sentence-case, 11pt, no tracking
//   "italic-caps" — italic + small-caps + tracked, 11pt
//   "plain"       — bold body-size in body-color (for headers that should
//                   feel like a label, not an eyebrow)
//   "passthrough" — emit the `header` argument verbatim, no wrapping. Use
//                   when the caller has already formatted the header
//                   (e.g. with `sub-eyebrow-segments(...)` for math-aware
//                   eyebrows). The caller is responsible for the
//                   below-block spacing.

#let outline-card(
  body,
  header: none,
  header-color: dp-purple,
  header-style: "eyebrow",
  header-align: left,       // alignment of the header within the card
  header-position: "above", // "above" (default) | "left" (header sits
                            //   beside body, vertically centred)
  header-gap: auto,         // auto → 4pt vertical (above) /
                            //        14pt horizontal (left).
                            //   Pass a length to override.
  stroke-weight: "outer",   // "outer" | "inner" | a stroke value
  radius: 10pt,
  inset: (x: 14pt, y: 11pt),
  width: 100%,
  body-size: 12pt,
  body-color: dp-purple-deep,
  fill: none,
) = {
  let resolved-stroke = if stroke-weight == "outer" {
    1.0pt + dp-purple-mid
  } else if stroke-weight == "inner" {
    0.6pt + dp-gray-100
  } else {
    stroke-weight
  }
  // Header as a block (with built-in below-spacing from `sub-eyebrow`
  // and friends). Used when `header-position: "above"`.
  let render-header-above(h) = {
    let rendered = if header-style == "eyebrow" {
      sub-eyebrow(h, color: header-color)
    } else if header-style == "eyebrow-sm" {
      sub-eyebrow(h, color: header-color, size: 9pt)
    } else if header-style == "bold" {
      block(below: 6pt, spacing: 0pt,
        text(size: 11pt, weight: 700, fill: header-color, h))
    } else if header-style == "italic-caps" {
      block(below: 6pt, spacing: 0pt,
        text(size: 11pt, weight: 700, style: "italic",
             tracking: 0.10em, fill: header-color, upper(h)))
    } else if header-style == "plain" {
      block(below: 6pt, spacing: 0pt,
        text(size: body-size, weight: 700, fill: body-color, h))
    } else if header-style == "passthrough" {
      h
    } else {
      sub-eyebrow(h, color: header-color)
    }
    // Wrap in `align(...)` so the header's alignment cascades into its
    // (full-width) inner block. Default `left` preserves prior behaviour.
    if header-align == left { rendered }
    else { align(header-align, rendered) }
  }
  // Header as bare inline content (no block, no below-spacing). Used
  // when `header-position: "left"` — sits alongside body content in
  // a grid cell, so block-flow spacing would shift the alignment.
  let render-header-inline(h) = {
    if header-style == "eyebrow" {
      text(size: 11pt, weight: 700, tracking: 0.12em,
           fill: header-color, upper(h))
    } else if header-style == "eyebrow-sm" {
      text(size: 9pt, weight: 700, tracking: 0.12em,
           fill: header-color, upper(h))
    } else if header-style == "bold" {
      text(size: 11pt, weight: 700, fill: header-color, h)
    } else if header-style == "italic-caps" {
      text(size: 11pt, weight: 700, style: "italic",
           tracking: 0.10em, fill: header-color, upper(h))
    } else if header-style == "plain" {
      text(size: body-size, weight: 700, fill: body-color, h)
    } else if header-style == "passthrough" {
      h
    } else {
      text(size: 11pt, weight: 700, tracking: 0.12em,
           fill: header-color, upper(h))
    }
  }
  // For `header-position: "left"`, the body grid column is `auto`
  // when the card itself is `width: auto` (so the whole card shrinks
  // to content) and `1fr` otherwise (body fills the remaining width
  // within the fixed card width).
  let left-body-col = if width == auto { auto } else { 1fr }
  block(
    width: width,
    fill: fill,
    stroke: resolved-stroke,
    radius: radius,
    inset: inset,
    {
      if header == none {
        set text(size: body-size, fill: body-color)
        set par(leading: 0.6em)
        body
      } else if header-position == "left" {
        let gap = if header-gap == auto { 14pt } else { header-gap }
        set text(size: body-size, fill: body-color)
        set par(leading: 0.6em)
        grid(
          columns: (auto, left-body-col),
          column-gutter: gap,
          align: (left + horizon, left + horizon),
          render-header-inline(header),
          body,
        )
      } else {
        // header-position == "above" (default — backward compatible)
        render-header-above(header)
        v(if header-gap == auto { 4pt } else { header-gap })
        set text(size: body-size, fill: body-color)
        set par(leading: 0.6em)
        body
      }
    },
  )
}

// ─── Glossary card (slide 4 "Where" panel) ───────────────────────────────

#let glossary-card(header, body) = block(
  width: 100%,
  fill: rgb("#FAFAFB"),
  stroke: 0.7pt + dp-gray-100,
  radius: 10pt,
  inset: (x: 14pt, y: 12pt),
  {
    sub-eyebrow(header, color: dp-gray-500)
    v(6pt)
    body
  },
)

#let glossary-row(symbol, description) = grid(
  columns: (34pt, 1fr),
  column-gutter: 8pt,
  row-gutter: 4pt,
  align: (right + horizon, left + horizon),
  symbol, description,
)

// ─── Filled-brand callout (slide 4 "Limit" box) ──────────────────────────

#let brand-callout(header, body) = block(
  width: 100%,
  fill: dp-purple,
  radius: 10pt,
  inset: (x: 14pt, y: 12pt),
  {
    sub-eyebrow(header, color: dp-purple-mid)
    v(6pt)
    text(fill: dp-white, size: 10pt, body)
  },
)

// ─── Deep-purple callout (slide 6 "Population" / slide 9 footer style) ──

#let dark-callout(header, body) = block(
  width: 100%,
  fill: dp-purple-deep,
  radius: 10pt,
  inset: (x: 14pt, y: 10pt),
  {
    sub-eyebrow(header, color: dp-purple-mid)
    v(4pt)
    text(fill: dp-white, size: 9pt, body)
  },
)

// ─── Pill chip (slide 3 "Same likelihood. Same workflow." bar) ──────────
// Inline content: a small filled circle + label, both vertically centred
// on a single row. The dot uses a real `circle()` (not a glyph) wrapped
// in a grid so `align: horizon + left` puts its centre exactly on the
// text's optical midline — independent of font baseline metrics.
//
// Tunable knobs:
//   `dot-size: auto`   — circle radius. `auto` → `text-size * 0.3` so the dot
//                        scales with the text. Pass a length to override.
//   `radius: 999pt`    — corner roundness. 999pt = fully rounded pill;
//                        pass `8pt` for soft-rectangle, `0pt` for sharp, etc.
//   `gap`              — space between dot and text.

#let pill(
  body,
  dot: true,
  fill: dp-purple-tint,
  dot-color: dp-purple,
  text-color: dp-purple-deep,
  text-size: 10pt,
  dot-size: auto,
  radius: 999pt,
  inset: (x: 16pt, y: 8pt),
  gap: 8pt,
) = {
  let r = if dot-size == auto { text-size * 0.3 } else { dot-size }
  box(
    fill: fill,
    radius: radius,
    inset: inset,
    if dot {
      grid(
        columns: (auto, auto),
        column-gutter: gap,
        align: horizon + left,
        circle(radius: r, fill: dot-color, stroke: none),
        text(size: text-size, weight: 600, fill: text-color, body),
      )
    } else {
      text(size: text-size, weight: 600, fill: text-color, body)
    },
  )
}

// ─── Metric card (slide 6 single + slide 7 row of four) ─────────────────

#let metric-card(
  label: none,
  value: none,
  sublabel: none,
  variant: "default",
  size: "lg",
) = {
  let is-brand = variant == "brand"
  let card-fill   = if is-brand { dp-purple } else { dp-white }
  let card-stroke = if is-brand { none } else { 0.7pt + dp-gray-100 }
  let label-color = if is-brand { dp-purple-mid } else { dp-gray-500 }
  let value-color = if is-brand { dp-white } else { dp-purple-deep }
  let sub-color   = if is-brand { rgb("#E2D2F0") } else { dp-gray-700 }

  let value-size = if size == "lg" { 30pt } else { 20pt }

  block(
    width: 100%,
    fill: card-fill,
    stroke: card-stroke,
    radius: 12pt,
    inset: 12pt,
    {
      if label != none { sub-eyebrow(label, color: label-color) ; v(6pt) }
      text(
        size: value-size,
        weight: 700,
        fill: value-color,
        number-width: "tabular",
        value,
      )
      if sublabel != none {
        v(4pt)
        text(size: 8pt, fill: sub-color, sublabel)
      }
    },
  )
}

// ─── Chart card (slide 6 left column) ────────────────────────────────────

#let chart-card(title: none, legend: none, body) = block(
  width: 100%,
  fill: dp-white,
  stroke: 0.7pt + dp-gray-100,
  radius: 12pt,
  inset: (x: 14pt, y: 12pt),
  {
    if title != none or legend != none {
      grid(
        columns: (1fr, auto),
        column-gutter: 8pt,
        align: (left + bottom, right + bottom),
        if title != none { text(size: 9pt, weight: 700, fill: dp-purple-deep, title) } else { [] },
        if legend != none { text(size: 8pt, fill: dp-gray-700, legend) } else { [] },
      )
      v(6pt)
    }
    body
  },
)

// ─── Info card (slide 6 middle right) ───────────────────────────────────

#let info-card(body) = block(
  width: 100%,
  fill: rgb("#FAFAFB"),
  stroke: 0.7pt + dp-gray-100,
  radius: 10pt,
  inset: (x: 12pt, y: 10pt),
  text(size: 9pt, fill: dp-purple-deep, body),
)

// ─── Progress bar (slide 7 footer) ───────────────────────────────────────

#let progress-bar(label: none, value: 0%, accent: dp-purple) = {
  if label != none {
    grid(
      columns: (1fr, auto),
      column-gutter: 6pt,
      text(size: 9pt, fill: dp-gray-700, label),
      text(size: 9pt, weight: 700, fill: dp-purple-deep, number-width: "tabular",
           str(int(value / 1%)) + "%"),
    )
    v(4pt)
  }
  block(
    width: 100%,
    height: 6pt,
    fill: dp-purple-50,
    radius: 999pt,
    {
      place(
        left + horizon,
        rect(
          width: value,
          height: 6pt,
          radius: 999pt,
          fill: gradient.linear(dp-purple, dp-purple-mid),
        ),
      )
    },
  )
}

// ─── Numbered list (slide 9 takeaways) ───────────────────────────────────

// The badge is an 18pt circle, taller than a single line of the 11pt body
// text. With `left + top` on the text column, the badge's visual centre
// (row middle = 9pt) sat well below the text's first-line centre
// (~6pt) — a ~3pt vertical mismatch. The single-list `numbered-list`
// fixes this by aligning both cells with `horizon`, which puts the
// badge centre on the body's geometric centre — correct for the 1-
// and 2-line items the deck uses. (For very tall multi-line items the
// badge centres on the whole text block; acceptable trade-off given
// the common case.)
//
// `parallel-numbered-list` cannot use `horizon` (rows have shared
// heights driven by the tallest item; horizon would centre 1-line
// items in tall rows and break cross-column alignment). It solves the
// same alignment problem with a different mechanism — top-anchoring
// plus a logical-height box matching the body's first-line bounding
// box. See `parallel-numbered-list` below for the worked example.
#let _numbered-badge(n) = block(
  width: 18pt,
  height: 18pt,
  fill: dp-purple-tint,
  radius: 50%,
  align(center + horizon,
    text(size: 9pt, weight: 700, fill: dp-purple, str(n))),
)

// `horizontal: false` (default) — the original vertical stack: one item
// per row, badge in the left column.
//
// `horizontal: true` — items laid out left-to-right. Each item is a
// (badge, text) pair separated by `inter-pair-gap` (default 18pt, larger
// than the within-pair gap so the pairs read as discrete groups).
// Useful for short, glanceable lists where vertical stacking would waste
// space (e.g. a tight 2-step recipe in a side box).
#let numbered-list(
  ..items,
  horizontal: false,
  spacing: 10pt,           // gap between items in vertical mode
  inter-pair-gap: 18pt,    // gap between pairs in horizontal mode
) = {
  let pairs = items.pos().enumerate().map(((i, item)) => (
    _numbered-badge(i + 1),
    text(size: 11pt, fill: dp-purple-deep, item),
  ))
  if horizontal {
    stack(
      dir: ltr,
      spacing: inter-pair-gap,
      ..pairs.map(((badge, item)) => grid(
        columns: (18pt, auto),
        column-gutter: 8pt,
        align: (center + horizon, left + horizon),
        badge,
        item,
      )),
    )
  } else {
    stack(
      spacing: spacing,
      ..pairs.map(((badge, item)) => grid(
        columns: (18pt, 1fr),
        column-gutter: 10pt,
        align: (center + horizon, left + horizon),
        badge,
        item,
      )),
    )
  }
}

// Parallel numbered lists with badges aligned across columns.
//
// When two (or more) numbered lists need to sit side-by-side and the
// badges in row k must line up across columns, a `grid(..., a, b)`
// wrapping two `numbered-list(...)`s is not enough — each list has its
// own per-row heights, which generally differ, so the badges drift
// past row 1. `parallel-numbered-list` renders every column into a
// single unified grid so row k has a single shared height across all
// columns and the badges in row k sit on the same baseline.
//
// Usage:
//
//   #parallel-numbered-list(
//     ([Take 1], [Take 2], [Take 3]),    // left column items (array)
//     ([Outlook 1], [Outlook 2]),         // right column items (shorter ok)
//   )
//
// Columns are passed positionally as arrays. Unequal-length columns are
// supported — the longest column drives the row count; shorter columns
// leave their trailing rows empty.
//
// Options:
//   align-numbers: true   (default) — badges align across columns AND
//                                     centre against the body's first
//                                     line. Both effects are achieved
//                                     by anchoring badge cells to the
//                                     row top and wrapping each badge
//                                     in a logical-height box equal to
//                                     the body's first-line height
//                                     (measured at compile time). The
//                                     visible 18pt circle overflows
//                                     the box by ~3pt above and below
//                                     — handled by the row-gutter and
//                                     the v-gap above the list. Set
//                                     false to fall back to per-cell
//                                     `horizon` (badge centred on body
//                                     geometric centre, no box wrap)
//                                     — only useful for uniformly
//                                     1-line items where the mid-line
//                                     look is preferred.
//   spacing:       14pt              — gap between rows
//   column-gutter: 36pt              — gap between adjacent list columns
//   badge-gutter:  10pt              — gap between a badge and its body
//   badge-width:   18pt              — badge column width (match
//                                     `_numbered-badge`'s width)
#let parallel-numbered-list(
  ..columns,
  align-numbers: true,
  spacing: 14pt,
  column-gutter: 36pt,
  badge-gutter: 10pt,
  badge-width: 18pt,
) = {
  let cols = columns.pos()
  if cols.len() == 0 { return [] }
  let n-cols = cols.len()
  let n-rows = calc.max(..cols.map(c => c.len()))

  // 2N grid columns interleaved: badge, body, badge, body, …
  let grid-cols = ()
  for _ in range(n-cols) {
    grid-cols.push(badge-width)
    grid-cols.push(1fr)
  }

  // 2N-1 gutters: badge-gutter (badge↔body), column-gutter (body↔next
  // badge), badge-gutter, column-gutter, …, badge-gutter.
  let gutters = ()
  for c in range(n-cols) {
    gutters.push(badge-gutter)
    if c < n-cols - 1 { gutters.push(column-gutter) }
  }

  // Per-column alignment. Vertical anchor = top when align-numbers is
  // true (badges sit at row top so they line up across columns even
  // when items have different heights — the badge-box wrapping
  // applied below then re-centres the badge against the body's first
  // line). horizon otherwise (badge centres on body geometric centre).
  let vert = if align-numbers { top } else { horizon }
  let cell-aligns = ()
  for _ in range(n-cols) {
    cell-aligns.push(center + vert)
    cell-aligns.push(left + vert)
  }

  context {
    // Align the badge's vertical centre with the body's first-line
    // vertical centre when align-numbers: true.
    //
    // The badge is an 18pt-tall block with the number centred in it
    // (number centre at y=9pt below the badge top). A line of 11pt
    // body text has a bounding box of ~11.5pt (font ascent + descent),
    // so its first-line centre sits at y=~5.75pt below the cell top.
    // With both cells anchored to row top, the badge centre would
    // therefore sit ~3pt below the body's first-line centre — a
    // visible mismatch (especially on 1-line items like the conclusion
    // slide's outlook column, where there's no second line to anchor
    // the eye on a 2-line geometric mean).
    //
    // Fix: wrap each badge in a `box` whose logical height equals the
    // body's first-line bounding box, with the badge centred inside.
    // The visible 18pt circle then overflows the box by ~3pt above
    // and below; the row's natural height is otherwise unchanged
    // (still driven by the tallest body cell), and both cells'
    // top-anchoring puts their centres at the same y. The single-list
    // `numbered-list` sidesteps this with per-cell `horizon`
    // anchoring, but here we cannot — rows have shared heights driven
    // by the tallest item, so `horizon` would centre 1-line items in
    // tall rows and break cross-column alignment.
    //
    // We measure the body line height instead of hard-coding it so
    // the alignment stays exact across font changes (Source Sans 3
    // metrics ≠ generic).
    let body-line-h = measure(text(size: 11pt)[Xj]).height

    // Build cells row-major so the grid's natural row heights are the
    // per-row maxima across all columns (= shared row heights).
    let cells = ()
    for r in range(n-rows) {
      for c in range(n-cols) {
        let col = cols.at(c)
        if r < col.len() {
          let badge = if align-numbers {
            box(width: badge-width, height: body-line-h,
              align(center + horizon, _numbered-badge(r + 1)))
          } else {
            _numbered-badge(r + 1)
          }
          cells.push(badge)
          cells.push(text(size: 11pt, fill: dp-purple-deep, col.at(r)))
        } else {
          cells.push([])
          cells.push([])
        }
      }
    }
    grid(
      columns: grid-cols,
      column-gutter: gutters,
      row-gutter: spacing,
      align: cell-aligns,
      ..cells,
    )
  }
}

// ─── Tagged list (slide 9 "What's next") ─────────────────────────────────

#let _tag-pill(label, variant: "dark") = {
  let bg = if variant == "brand" { dp-purple } else { dp-purple-deep }
  // Generous y-padding so the pill's height roughly matches the body
  // text's line height (matches the source design's tag proportions).
  block(
    width: 42pt,
    fill: bg,
    radius: 4pt,
    inset: (x: 6pt, y: 5pt),
    align(center,
      text(
        size: 9pt,
        weight: 700,
        fill: dp-white,
        number-width: "tabular",
        label,
      )),
  )
}

#let tagged-list(..entries) = stack(
  spacing: 10pt,
  ..entries.pos().map(e => grid(
    columns: (42pt, 1fr),
    column-gutter: 10pt,
    align: (left + top, left + top),
    _tag-pill(e.at("tag"), variant: e.at("variant", default: "dark")),
    text(size: 11pt, fill: dp-purple-deep, e.at("body")),
  )),
)

// ─── Thanks card (slide 9 footer pill) ───────────────────────────────────

// Thanks-card: white pill with a "● THANKS FOR LISTENING" header followed
// by any number of `(label, value)` link pairs, separated by small gray
// centre-dots (in keeping with the rest of the deck's visual language).
//
// Each visual element — the leading bullet, the header text, every
// separator dot, every (label, value) pair — is its own grid cell. The
// grid's `align: horizon` then puts every cell's centre on a single row
// baseline, so all elements visually line up no matter how many entries
// are passed. The theme places this card on the same y-line as the
// slide-counter and the brand-mark.
#let thanks-card(
  ..items,
  header: [Thanks for listening],
  bullet-color: dp-purple,
  separator-color: dp-gray-500,
) = {
  let entries = items.pos()
  let cells = ()

  // Leading bullet (own cell) + header text (own cell), so each cell's
  // centre is computed independently and aligned by the grid.
  cells.push(circle(radius: 2.5pt, fill: bullet-color, stroke: none))
  cells.push(text(size: 8pt, weight: 700, tracking: 0.12em, fill: dp-purple,
                  upper(header)))

  for entry in entries {
    let (label, value) = entry
    // Centered gray dot as separator (replaces the vertical bar).
    cells.push(circle(radius: 1.4pt, fill: separator-color, stroke: none))
    cells.push({
      text(size: 9pt, fill: dp-gray-500, label)
      h(4pt)
      text(size: 9pt, fill: dp-purple-deep, value)
    })
  }

  block(
    fill: dp-white,
    stroke: 0.7pt + dp-gray-100,
    radius: 8pt,
    inset: (x: 14pt, y: 8pt),
    grid(
      columns: (auto,) * cells.len(),
      column-gutter: 10pt,
      align: horizon + center,
      ..cells,
    ),
  )
}

// ─── Thanks panel (closing-slide card) ───────────────────────────────────

// `thanks-panel` is the larger sibling of `thanks-card`. Where
// `thanks-card` is a single-row pill designed to sit in a content-slide's
// footer slot, `thanks-panel` is a body-level card with a greeting on
// the left and a stack of clickable URL links on the right — meant for
// the closing slide where the "thanks" moment deserves real estate.
//
// The greeting is passed in as content so the caller controls font /
// size / weight / colour / casing. The helper imposes only the layout
// (greeting left, links right) and the card's visual frame.
//
// Link rendering: each link is a `(label, url)` pair. The full URL
// (including the scheme) is the navigation target; the displayed text
// is the URL with `https://` (or `http://`) stripped, so e.g.
// `https://pumas.ai` reads as `pumas.ai` on the slide but clicking it
// still opens the right page. The whole label-and-url row is one click
// target.
//
// Usage:
//
//   #thanks-panel(
//     ([Web],         "https://pumas.ai"),
//     ([Publications], "https://pumas.ai/resources/publications"),
//     greeting: text(
//       size: 18pt, weight: 700, tracking: 0.12em,
//       fill: dp-purple-deep, upper[Thank you for listening],
//     ),
//   )
//
// Options:
//   greeting:        content    — required for the 2-column form; pass
//                                 `none` to render the link stack alone.
//   fill:            dp-purple-tint
//   stroke:          1.2pt + dp-purple
//   radius:          16pt
//   inset:           (x: 28pt, y: 16pt)
//   column-gutter:   36pt       — gap between greeting and link stack
//   link-spacing:    8pt        — gap between consecutive link rows
//   strip-schemes:   true       — strip `https://` / `http://` from the
//                                 *displayed* URL (target keeps the
//                                 scheme so the link still works)
//   label-color, url-color, label-size, url-size  — styling knobs for
//                                 the link rows.
//   label-gap:       8pt        — horizontal gap between a row's label
//                                 and its URL. Inline on a shared
//                                 baseline so different label/url
//                                 sizes don't desync.
//   label-align:     "right"    — how labels and URLs lay out across
//                                 rows. Three modes:
//                                 - "right" (default): every label is
//                                   placed in a fixed-width box
//                                   (= longest label's measured
//                                   width), right-justified inside the
//                                   box, so labels' *right* edges line
//                                   up across rows AND every URL
//                                   starts at the same x. Reads as a
//                                   tidy key/value table — the most
//                                   structured of the three modes.
//                                 - "left":  same fixed-width box,
//                                   but labels left-justified. URLs
//                                   still share their start x; the
//                                   label column has a ragged right
//                                   edge.
//                                 - "inline": no shared column —
//                                   label and URL sit immediately
//                                   next to each other on a single
//                                   inline line, so URLs start at a
//                                   row-dependent x (longer labels
//                                   push their URL further right).
//                                   Compact, prose-like.
//                                 In all three modes label and URL on
//                                 the same row share a baseline
//                                 (inline placement).
#let thanks-panel(
  ..links,
  greeting: none,
  fill: dp-purple-tint,
  stroke: 1.2pt + dp-purple,
  radius: 16pt,
  inset: (x: 28pt, y: 16pt),
  column-gutter: 36pt,
  link-spacing: 8pt,
  strip-schemes: true,
  label-color: dp-purple,
  url-color: dp-purple-deep,
  label-size: 11pt,
  url-size: 13pt,
  label-gap: 8pt,
  label-align: "right",
) = context {
  // Display value for a URL: optionally strip the scheme so the link
  // text reads as a clean hostname/path.
  let _display(url) = {
    if not strip-schemes { return url }
    if url.starts-with("https://") { url.slice(8) }
    else if url.starts-with("http://") { url.slice(7) }
    else { url }
  }

  // Styled label content (used both for measurement and rendering).
  let _styled-label(label) = text(
    size: label-size, weight: 700, tracking: 0.12em,
    fill: label-color, upper(label),
  )

  // Shared label-column width across all rows. Only meaningful when
  // label-align is "left" or "right" — for "inline" mode each row
  // sizes its own label naturally. Measured here under a `context`
  // wrapper (see the outer `context` on the helper) so the rendered
  // width matches the layout-time width.
  let pairs = links.pos()
  let label-col-width = if label-align == "inline" or pairs.len() == 0 {
    auto
  } else {
    calc.max(..pairs.map(((l, _)) => measure(_styled-label(l)).width))
  }

  // Render a single label. In "inline" mode the label is plain text;
  // in "left"/"right" mode it is placed inside a fixed-width box with
  // the requested internal alignment, so the URLs that follow line up
  // across rows.
  let _render-label(label) = {
    if label-col-width == auto {
      _styled-label(label)
    } else {
      let side = if label-align == "right" { right } else { left }
      box(width: label-col-width, align(side, _styled-label(label)))
    }
  }

  // One link row: label + URL on a single inline line, the whole row
  // wrapped in `link(...)` so both pieces are clickable targets
  // pointing at the full URL. Inline (not a 2-cell grid) so the label
  // and URL share a baseline: Typst lays mixed-size spans in the same
  // line on a common baseline, whereas `grid(align: horizon)` would
  // centre the cell *bounding boxes* — and a 11pt label and a 13pt URL
  // have different bounding-box heights, so horizon-aligned cells
  // leave the baselines visibly offset. `box(...)` keeps the pair
  // non-breaking inside the line.
  let _link-line(label, url) = link(url, box({
    _render-label(label)
    h(label-gap)
    text(size: url-size, fill: url-color, _display(url))
  }))

  let link-stack = stack(
    spacing: link-spacing,
    ..pairs.map(((label, url)) => _link-line(label, url)),
  )

  let inner = if greeting == none {
    link-stack
  } else {
    grid(
      columns: (auto, auto),
      column-gutter: column-gutter,
      align: (left + horizon, left + horizon),
      greeting,
      link-stack,
    )
  }

  block(
    fill: fill,
    stroke: stroke,
    radius: radius,
    inset: inset,
    inner,
  )
}

// ─── Author-role wrappers ────────────────────────────────────────────────

#let first-author(body) = strong(body)

// Presenter mark: black underline by default (matches the source deck's
// understated style). Pass `color:` to override per call.
#let presenter(body, color: black, thickness: 0.7pt, offset: 2pt) = underline(
  stroke: thickness + color,
  offset: offset,
  body,
)

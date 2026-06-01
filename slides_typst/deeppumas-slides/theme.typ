// ──────────────────────────────────────────────────────────────────────────
//  deeppumas-slides — Touying custom theme
//
//  Pinned dependency:
//    @preview/touying:0.7.3
//
//  The pin is required by Typst Universe (no semver-range syntax exists).
//  To take a Touying bugfix release, bump the version string just below.
//  Tested ranges are documented in README.md.
// ──────────────────────────────────────────────────────────────────────────

#import "@preview/touying:0.7.3": *
#import "logos.typ": logo, cobrand-row, pumas-centered, deeppumas-centered
#import "building-blocks.typ": *

// ─── Layout proportions ─────────────────────────────────────────────────
// All positioning is expressed as a fraction of the page so the layout
// reflows correctly across aspect ratios *and* across whatever absolute
// page size Touying picks for the chosen ratio. Text sizes are tuned for
// Touying's default 16:9 paper (841.89pt × 473.56pt).

#let _PAD-X     = 4.7%   // horizontal page padding
#let _PAD-TOP   = 7.8%   // top page padding (title eyebrow row)
#let _PAD-BOT   = 7.5%   // bottom page padding (clears the slide-num row)
#let _FOOTER-Y  = 6pt    // visual centre line for slide-num + brand-mark
// Brand-mark distance from the right page edge. Sits a little OUTSIDE the
// content margin so the logo doesn't crowd whatever's in the right column.
// Override per slide via the `corner-x:` keyword on content/section/quote
// slide functions.
#let _CORNER-X  = 14pt
#let _BRAND-H   = 30pt   // brand-mark row-box height (= slide-num row height; shared centreline)
// Default height of the brand-mark *logo image* inside that row-box. Smaller
// than the row-box so that the visible wordmark glyph height matches what it
// was before the bundled `*-primary-*` SVGs were retrimmed to tight viewBoxes
// (content used to occupy ~66.5 % of the viewBox; now occupies ~91 %).
// Override per slide via the `brand-logo-height:` keyword on
// `content-slide` / `section-slide` / `quote-slide`.
#let _BRAND-LOGO-H = 22pt
#let _CONTENT-W = 100% - 2 * _PAD-X

// ─── Internal: decorative corner arcs ────────────────────────────────────
// All slides use `margin: 0pt`, so the `place(...)` calls below resolve
// against the full page. Anything outside the page bounds is clipped by
// the page boundary automatically (no manual clip mask needed).

// Radii are in `pt` because `circle(radius: ...)` only accepts absolute
// lengths. Tuned for Touying's default 16:9 paper (841.89pt wide); on a
// larger custom page they'll appear proportionally smaller — pass scaled
// values via the `arc-scale:` argument to override.
#let _arcs(variant: "title", scale: 1.0) = {
  let r = scale * 1pt
  if variant == "title" {
    place(top + right, dx: 22%,  dy: -25%, circle(radius: 210pt * scale, fill: dp-purple-tint))
    place(top + right, dx: 8%,   dy: 10%,  circle(radius: 125pt * scale, fill: dp-purple-mid.transparentize(70%)))
  } else if variant == "section" {
    place(bottom + left,  dx: -26%, dy: 30%,  circle(radius: 255pt * scale, fill: dp-purple-mid.transparentize(82%)))
    place(top    + right, dx: -12%, dy: 10%,  circle(radius: 110pt * scale, fill: dp-purple-mid.transparentize(88%)))
  } else if variant == "quote" {
    place(top + right,    dx: 18%,  dy: -20%, circle(radius: 170pt * scale, fill: dp-purple-mid.transparentize(90%)))
    place(bottom + right, dx: -8%,  dy: -14%, circle(radius: 92pt  * scale, fill: dp-purple-mid.transparentize(90%)))
  } else if variant == "thanks" {
    place(bottom + right, dx: 22%,  dy: 26%,  circle(radius: 225pt * scale, fill: dp-purple-mid.transparentize(82%)))
    place(top    + left,  dx: -10%, dy: -12%, circle(radius: 92pt  * scale, fill: rgb(255, 255, 255, 16)))
  }
}

// ─── Internal: brand-mark in the bottom-right corner ────────────────────
// The brand mark and the slide-number footer share a single visual
// centre-line. Both elements are wrapped in a fixed-height row-box and
// `align(horizon, ...)` is used to vertically centre their contents
// inside that row — so a tall logo and a short "N / N" string end up
// vertically centred on the same Y.

#let _brand-mark(
  name,
  height: _BRAND-H,            // outer row-box height (shared centreline with slide-num)
  logo-height: _BRAND-LOGO-H,  // inner logo image height — the visible glyph size
  corner-x: _CORNER-X,
  footer-y: _FOOTER-Y,
) = {
  // `corner-x: auto` → fall back to the layout constant.
  let cx = if corner-x == auto { _CORNER-X } else { corner-x }
  place(
    bottom + right,
    dx: -1 * cx,
    dy: -1 * footer-y,
    box(height: height, align(horizon + right, logo(name, height: logo-height))),
  )
}

// ─── Internal: slide-number footer ──────────────────────────────────────
// Plain digits (no leading zeros) aligned with tabular figures so the
// column doesn't jitter as the counter rolls past 9. Emitted from the
// slide body (not `page.background:`) so the page-preamble's counter-step
// has already happened by the time it renders — Touying steps the counter
// during `page-preamble`, which runs AFTER `set page(...)` chrome but
// before the body. Reading it from `page.background` / `page.foreground`
// (or any other `set page(...)` slot) gives you the *previous* slide's
// number.
//
// Verification:
//   Compile the example deck and open the LAST physical PDF page. The
//   footer-left must read "N / N" — not "N-1 / N". (For the example deck
//   with one #pause that produces 10 physical pages from 9 logical slides,
//   the last content-slide should read "9 / 9".)

#let _slide-num(
  color: dp-gray-500,
  row-height: _BRAND-H,
  pad-x: _PAD-X,
  footer-y: _FOOTER-Y,
) = place(
  bottom + left,
  dx: pad-x,
  dy: -1 * footer-y,
  box(height: row-height, align(horizon + left,
    context {
      let cur  = utils.slide-counter.get().first()
      let last = utils.last-slide-counter.final().first()
      text(
        size: 9pt,
        weight: 700,
        tracking: 0.14em,
        fill: color,
        number-width: "tabular",
        upper(str(cur) + " / " + str(last)),
      )
    },
  )),
)

// ─── Internal: standard slide chrome ────────────────────────────────────

#let _chrome(
  arcs: none,
  brand-logo: none,
  brand-logo-height: _BRAND-LOGO-H,
  brand-corner-x: auto,
  slide-num-color: dp-gray-500,
  show-num: true,
  show-brand: true,
  footer-card: none,
) = {
  if arcs != none { _arcs(variant: arcs) }
  if show-num { _slide-num(color: slide-num-color) }
  if show-brand and brand-logo != none {
    _brand-mark(brand-logo, logo-height: brand-logo-height, corner-x: brand-corner-x)
  }
  if footer-card != none {
    // The footer-card lives on the same baseline as the slide-num and
    // brand-mark: same dy from the page bottom, wrapped in a row-box of
    // the same height so its contents centre-align with the other two.
    place(bottom + center, dx: 0pt, dy: -1 * _FOOTER-Y,
      box(height: _BRAND-H, align(horizon + center, footer-card)))
  }
}

// Wrap inner content in the standard page padding (left/right/top/bottom).

#let _page-pad(body) = pad(
  top: _PAD-TOP, bottom: _PAD-BOT, x: _PAD-X,
  body,
)

// ─── Affiliation grid (title slide) ─────────────────────────────────────
// Renders the affiliations row (or stack) on the title slide. The layout
// keeps all eyebrows visually aligned even when entries mix single-line
// and multi-line text — see `_affiliations-grid` below for the per-layout
// alignment details.

// `_affil-eyebrow` returns the eyebrow content for one entry — a text
// label, a brand lockup, a bundled logo, or a user-supplied image.
//
// Entry forms (use exactly one):
//   `label: "TEXT"`                            — plain uppercased eyebrow
//   `brand: "pumasai"` | `"deeppumas"`         — logomark + Amplitude
//                                                 wordmark (cleanest
//                                                 baseline alignment with
//                                                 adjacent text eyebrows)
//   `logo: "shortname"`                        — bundled SVG (wordmark
//                                                 baseline is approximate)
//   `logo-path: "./your-logo.svg"`             — external SVG
//
// Knobs (any form):
//   `logo-height: 16pt`  — absolute height
//   `logo-scale:  1.2`    — multiplier on the default
//
// Text labels render at the same point size as the `brand:` Amplitude
// wordmark (12pt by default — the theme's body text size), so all four
// affiliation columns share one visual line height even when some columns
// use a brand lockup and others a plain text label.
#let _affil-eyebrow(entry, default-logo-height: 14pt) = {
  if "label" in entry {
    // No `upper()` and only mild tracking, so the rendered text matches the
    // optical weight of the adjacent Amplitude wordmarks rather than reading
    // as a small-caps eyebrow. Keep `dp-purple-deep` (very dark, on-brand)
    // so the colour sits next to the `pumas-centered` lockup (which defaults
    // to black) without an obvious tilt.
    //
    // `bottom-edge: "baseline"` makes the text's frame end exactly at the
    // baseline (no descender space below). Paired with `align: bottom` in
    // the affiliations grid, this baseline-aligns the institution with its
    // author row — the same convention HTML tables use for `<td>`s with
    // mixed font sizes.
    text(
      size: 12pt,
      weight: 600,
      tracking: 0.02em,
      fill: dp-purple-deep,
      // Tight metrics: line-box = cap-line region (top at cap-top,
      // bottom at baseline). `_affiliations-grid` uses this together
      // with the matching setting on `_affil-authors` and a per-cell
      // inset to baseline-align institution and authors first lines —
      // HTML-table style. See the long note in `_affiliations-grid`.
      top-edge: "cap-height",
      bottom-edge: "baseline",
      entry.label,
    )
  } else if "brand" in entry {
    // Centered variants: the icon extends above and below the text so the
    // lockup reads like the original SVG wordmark, while the text baseline
    // still aligns with the adjacent text-eyebrow rows.
    if entry.brand == "pumasai" {
      pumas-centered()
    } else if entry.brand == "deeppumas" {
      deeppumas-centered()
    } else {
      panic("unknown brand `" + entry.brand + "` — use \"pumasai\" or \"deeppumas\"")
    }
  } else {
    let path = if "logo" in entry { entry.logo } else if "logo-path" in entry { entry.logo-path } else {
      panic("affiliation entry must have `label`, `brand`, `logo`, or `logo-path`")
    }
    let scale = entry.at("logo-scale", default: 1.0)
    let h = entry.at("logo-height", default: default-logo-height * scale)
    logo(path, height: h)
  }
}

// `size` is a required keyword — `_affiliations-grid` always passes the
// caller's `authors-size`, and there is no sensible global default that
// won't go stale relative to `title-slide`'s default.
//
// `top-edge: "cap-height"` + `bottom-edge: "baseline"` mirrors the same
// settings in `_affil-eyebrow` (label branch): both texts use the same
// tight line-box (cap-top to baseline) so the per-cell inset in the
// grid can baseline-align the institution with the first author line —
// HTML-table-style.
#let _affil-authors(entry, size: none) = block({
  assert(size != none, message: "_affil-authors: `size` is required")
  set par(leading: 0.55em)
  text(
    size: size,
    fill: dp-purple-deep,
    top-edge: "cap-height",
    bottom-edge: "baseline",
    entry.at("authors"),
  )
})

// Classify each entry — "text" (`label:`) vs "image" (`brand:` / `logo:` /
// `logo-path:`). Used to enforce that every affiliation in a single deck is
// the same kind, which keeps the row visually uniform.
#let _affil-entry-kind(entry) = {
  if "label" in entry {
    "text"
  } else if "brand" in entry or "logo" in entry or "logo-path" in entry {
    "image"
  } else {
    panic("affiliation entry must have `label`, `brand`, `logo`, or `logo-path`")
  }
}

// Layouts:
//   "stacked"  — (default) institutions stacked vertically, authors to the
//                right of each one. Left-aligned eyebrow column + left-aligned
//                authors column. Reads cleanly with many institutions, long
//                institution names, or long author lists.
//   "row"      — one column per affiliation side-by-side. Eyebrows
//                vertically centred so multi-line entries align with
//                single-line ones. Authors block sits beneath each
//                eyebrow. Use for the classic poster-style 2-4 affiliation
//                row.
//
// `institutions-align` (stacked layout only):
//   "right" — (default) institution names right-justify within the auto-sized
//             first column, so the names visually pull toward the authors
//             column.  Reads as a tight institution-author pairing.
//   "left"  — institution names left-justify within the same column. Short
//             names like "MIT" then leave whitespace before the authors.
//             The authors column always stays left-justified.
//   In the "row" layout this parameter is ignored.
//
// Entry-kind check: every entry must be the same kind (all `label:` or all
// logo-based) so the row reads as one consistent visual block.
//
// Default (`"stacked"`) matches `title-slide`'s public default — keeping
// the two defaults in sync so a maintainer doesn't have to remember which
// layer wins.
#let _affiliations-grid(
  entries,
  layout: "stacked",
  authors-size: 10pt,
  institutions-align: "right",
) = {
  if entries.len() == 0 { return }

  let kinds = entries.map(_affil-entry-kind)
  if kinds.dedup().len() > 1 {
    panic(
      "title-slide: all `affiliations` entries must be the same kind — " +
      "either all `label:` (text) or all logo-based (`brand:` / `logo:` / `logo-path:`). " +
      "Got a mix: " + repr(kinds),
    )
  }

  if layout == "row" {
    // 4-column hard cap: at the default 16:9 page size (≈ 762pt content
    // width) five or more columns crush each institution name into a
    // narrow ribbon that wraps awkwardly. Caller should switch to
    // `"stacked"` if they have more than four affiliations.
    let cols = calc.min(entries.len(), 4)
    grid(
      columns: (1fr,) * cols,
      rows: (auto, auto),
      column-gutter: 18pt,
      row-gutter: 10pt,
      // Eyebrows row: vertically centred. A multi-line entry (e.g. "Uppsala
      // University · Pharmetheus") makes the row taller; single-line entries
      // in the same row centre vertically inside that height so the whole
      // line of institution names reads as one visual band.
      ..entries.map(e => grid.cell(align: left + horizon, _affil-eyebrow(e))),
      ..entries.map(e => grid.cell(align: left + top, _affil-authors(e, size: authors-size))),
    )
  } else if layout == "stacked" {
    // One row per affiliation; institution in column 1 (auto-width),
    // authors in column 2 (1fr).
    //
    // `institutions-align` only changes the horizontal alignment of text
    // inside the institution column — the column itself stays as the left
    // grid column (auto-sized to its widest entry). At "right", the names
    // visually pull toward the authors, which is the typically-desired
    // effect.
    let inst-h-align = if institutions-align == "right" {
      right
    } else if institutions-align == "left" {
      left
    } else {
      panic(
        "title-slide: `institutions-align` must be \"left\" or \"right\", got "
          + repr(institutions-align),
      )
    }
    // Vertical alignment: baseline-align the institution row with the
    // FIRST line of the authors block — the same convention HTML tables
    // use for `<td>`s with mixed font sizes. Works for single-line
    // institution + single-line authors (the typical case) AND for
    // multi-line authors (subsequent author lines extend downward from
    // the row anchor).
    //
    // Mechanics:
    //   1. Both `_affil-eyebrow` (label branch) and `_affil-authors` set
    //      `top-edge: "cap-height"` and `bottom-edge: "baseline"` on their
    //      text, which makes the per-line layout frame the cap-line
    //      region: top at cap-top, bottom at baseline.
    //   2. `align: top` anchors each cell's first-line cap-top at the
    //      grid row top.
    //   3. The authors cell gets an extra top inset equal to the
    //      cap-height difference, so its first-line cap-top sits LOWER
    //      than the institution's cap-top by exactly that amount —
    //      enough that the two baselines coincide.
    //
    // Cap-height ≈ 0.66 × font-size for the bundled Source Sans 3 font
    // (and most humanist sans typefaces). If you switch to a font with a
    // markedly different ratio you may need to retune this constant or
    // expose it as a knob; for Source Sans 3 12pt vs 10pt the 1.32pt
    // shift is correct within sub-pixel tolerance.
    let cap-height-ratio = 0.66
    let inst-font-size = 12pt           // hard-coded in `_affil-eyebrow` (label branch)
    let baseline-shift = calc.max(
      0pt,
      (inst-font-size - authors-size) * cap-height-ratio,
    )
    let cells = ()
    for e in entries {
      cells.push(grid.cell(align: inst-h-align + top, _affil-eyebrow(e)))
      cells.push(grid.cell(
        align: left + top,
        inset: (top: baseline-shift),
        _affil-authors(e, size: authors-size),
      ))
    }
    grid(
      columns: (auto, 1fr),
      column-gutter: 24pt,
      row-gutter: 12pt,
      ..cells,
    )
  } else {
    panic(
      "title-slide: `affiliations-layout` must be \"row\" or \"stacked\", got " + repr(layout),
    )
  }
}

// ─── Slide functions ─────────────────────────────────────────────────────

// Title slide — cobrand logos (any number) at top-left, optional conference
// tag at top-right, big title + subtitle, multi-column affiliations row.
#let title-slide(
  config: (:),
  conference: none,
  cobrand: (),
  // Global default height for the cobrand logo row. Individual logos can
  // override via the `cobrand:` tuple's dict form — see `cobrand-row` for
  // the absolute (`length`) vs relative (`ratio`) semantics.
  cobrand-height: 42pt,
  title: none,
  subtitle: none,
  affiliations: (),
  // Default `"stacked"` — one institution per row with authors to the right;
  // reads cleanly even with long institution names or many entries. Pass
  // `"row"` for the classic side-by-side columns.
  affiliations-layout: "stacked",
  authors-size: 10pt,
  // Horizontal alignment of the institution names within their grid column
  // (stacked layout only — ignored when `affiliations-layout: "row"`).
  // `"right"` (default) right-justifies the names so they visually pull
  // toward the author lists. Pass `"left"` to left-justify instead.
  institutions-align: "right",
  legend: none,
  // Show the decorative brand-purple arcs/circles in the slide background.
  // Pass `false` for a fully clean white background (e.g. when the venue
  // demands a sober look or when an external co-brand asset clashes).
  show-arcs: true,
) = touying-slide-wrapper(self => {
  self = utils.merge-dicts(
    self,
    // Don't count the title slide in the slide counter — both the visible
    // counter on subsequent slides and the final denominator should treat
    // the deck as starting at "1" on the first content slide. Standard
    // Touying pattern, used by every first-party theme (Metropolis,
    // University, Simple, Stargazer, Dewdrop) on their title slides.
    config-common(freeze-slide-counter: true),
    config-page(fill: white, margin: 0pt, header: none, footer: none),
  )

  let body = {
    if show-arcs { _arcs(variant: "title") }

    // Conference tag — top-right. Sits halfway between the page top and
    // the deeper decorative arc (which has its top edge at dy: 10%), so
    // the tag visually centres in the empty band above the arc.
    if conference != none {
      place(top + right, dx: -1 * _PAD-X, dy: 5%,
        text(size: 10pt, weight: 700, tracking: 0.16em, fill: dp-gray-500,
             upper(conference)))
    }

    // Cobrand + title + subtitle stack — upper area
    place(top + left, dx: _PAD-X, dy: 21%,
      block(width: _CONTENT-W, {
        if cobrand.len() > 0 {
          // No divider between cobrand logos — the source deck uses them
          // bare. Users who want a divider in their own decks can pass
          // `cobrand: (..., ..., ...)` to a custom `cobrand-row(divider: true)`
          // call directly.
          cobrand-row(..cobrand, height: cobrand-height, divider: false)
          v(4pt)
        }
        if title != none {
          // 0.4em leading keeps the title tight without letting descenders
          // (g/y/p in the first line) collide with ascenders (h/d/b in the
          // second line). Smaller values overlap glyphs at this size.
          //
          // Verification: render the title slide with a multi-line title
          // that has descenders on the upper line and ascenders on the
          // lower (the example deck's "Hybrid modeling, / embedded in
          // pharmacometrics." does this — the "g" in "modeling" sits above
          // the "h" / "b" / "d" of the second line). Glyphs must not touch.
          set par(leading: 0.4em)
          text(size: 48pt, weight: 600, fill: dp-purple-deep, tracking: -0.025em,
               title)
        }
        if subtitle != none {
          set par(leading: 0.4em)
          text(size: 18pt, fill: dp-gray-700, subtitle)
        }
      }),
    )

    // Affiliations + legend — anchored just above the bottom edge.
    if affiliations.len() > 0 {
      place(bottom + left, dx: _PAD-X, dy: -7%,
        block(width: _CONTENT-W, {
          _affiliations-grid(
            affiliations,
            layout: affiliations-layout,
            authors-size: authors-size,
            institutions-align: institutions-align,
          )
          if legend != none {
            // More breathing room between the last institution row and the
            // legend in the stacked layout (the legend sits *below* a tall
            // multi-row block, so a tight 6pt gap makes it look glued on);
            // the row layout has a separate authors row already providing
            // some vertical separation, so a smaller gap is fine there.
            let legend-gap = if affiliations-layout == "stacked" { 18pt } else { 6pt }
            v(legend-gap)
            text(size: 8pt, fill: dp-gray-500, legend)
          }
        }),
      )
    }
  }

  touying-slide(self: self, config: config, body)
})


// Section divider — deep-purple background, eyebrow + huge title.
//
// Pass `part: 2` and the eyebrow renders as "Part 2" (no leading zero). Pass
// `eyebrow-text:` for arbitrary text, which takes precedence. Tuning knobs
// (`title-size`, `eyebrow-size`, `title-leading`) let you nudge typography
// without forking the function.
#let section-slide(
  config: (:),
  part: none,
  eyebrow-text: none,
  eyebrow-size: 12pt,
  // Vertical gap from the "PART N" eyebrow down to the hero title.
  // Applied via `block(below: eyebrow-gap, spacing: 0pt, ...)` which
  // bypasses the global paragraph spacing — so the rendered gap is
  // exactly this value, not paragraph spacing on top.
  eyebrow-gap: 10pt,
  title: none,
  title-size: 56pt,
  // Line-height between title lines at the section-slide's 56pt size.
  // Anything tighter than ~0.3em risks descender/ascender collision on
  // multi-line titles. See the verification note next to the title-slide
  // hero leading above.
  title-leading: 0.35em,
  // Optional one-line subtitle rendered in purple-mid below the title.
  // Used to give a section divider a short tagline — e.g.
  //   #section-slide(
  //     title: [The TGD-OS model],
  //     subtitle: [Predicting per-subject and per-arm survival distributions],
  //   )
  subtitle: none,
  subtitle-size: 22pt,
  // 22pt at the section-slide's 56pt title + 22pt subtitle pairing
  // gives a visibly open gap between the title's descenders (e.g. the
  // "g" of "augmentation") and the subtitle's ascenders / descenders
  // (e.g. the "p" of "plausible"). Lower values (the previous 12pt)
  // looked visually pinched because the font line boxes already eat
  // most of the gap as ascender + descender padding.
  subtitle-gap: 22pt,   // gap between title bottom and subtitle top
  brand-logo: "deeppumas-primary-white",
  brand-logo-height: _BRAND-LOGO-H, // tunes the corner brand-mark glyph height
  corner-x: auto,                   // brand-mark distance from page right edge
  // Show the decorative brand-purple arcs in the slide background. Pass
  // `false` for a flat fill (e.g. for an exec deck that should read sober).
  show-arcs: true,
) = touying-slide-wrapper(self => {
  self = utils.merge-dicts(
    self,
    // Section dividers are navigational markers, not content — don't count
    // them in the slide counter. See the matching note on `title-slide`.
    config-common(freeze-slide-counter: true),
    config-page(fill: dp-purple-deep, margin: 0pt, header: none, footer: none),
  )

  let resolved-eyebrow = if eyebrow-text != none {
    eyebrow-text
  } else if part != none {
    "Part " + str(part)
  } else {
    none
  }

  let body = {
    if show-arcs { _arcs(variant: "section") }
    _brand-mark(brand-logo, logo-height: brand-logo-height, corner-x: corner-x)

    place(left + horizon, dx: _PAD-X,
      block(width: _CONTENT-W, {
        if resolved-eyebrow != none {
          // `block(below:, spacing: 0pt)` so the gap is exactly `eyebrow-gap`
          // — bypasses the global paragraph spacing.
          block(below: eyebrow-gap, spacing: 0pt,
            text(size: eyebrow-size, weight: 700, tracking: 0.18em, fill: dp-purple-mid,
                 upper(resolved-eyebrow)))
        }
        if title != none {
          // Title in a fixed-gap block so the subtitle (if any) sits at
          // exactly `subtitle-gap` below the title — independent of global
          // paragraph spacing.
          let title-block = {
            set par(leading: title-leading)
            text(size: title-size, weight: 600, fill: white, tracking: -0.02em, title)
          }
          if subtitle != none {
            block(below: subtitle-gap, spacing: 0pt, title-block)
            text(size: subtitle-size, weight: 400, fill: dp-purple-mid, tracking: 0em, subtitle)
          } else {
            title-block
          }
        }
      }),
    )
  }

  touying-slide(self: self, config: config, body)
})


// Content slide — the workhorse. Optional eyebrow + title, body, optional
// inline footer-card, slide-number footer, brand-mark.
//
// Tuning knobs (any can be overridden per slide; live defaults below):
//   title-size      : 32pt    — h2 size matching the source deck
//   title-leading   : 0.35em  — line-height between title lines (descender floor)
//   title-weight    : 600     — bump to 700 for a heavier h2
//   eyebrow-size    : 9pt
//   eyebrow-gap     : 5pt     — vertical gap between eyebrow and title
//   subtitle-size   : 16pt    — body-style sentence below the title (used when
//                                `subtitle:` is set)
//   subtitle-gap    : 11pt    — vertical gap between title and subtitle
//                                (only applied when `subtitle:` is set)
//   title-gap       : 16pt    — vertical gap between title (or subtitle, if
//                                set) and body. Ignored when
//                                `center-body: true` (the centred body has
//                                equal top and bottom margins by definition).
//   center-body     : false   — when `true`, vertically centres the body in
//                                the area below the title/subtitle so the
//                                visual margins above and below the body are
//                                equal. Default is `false` (body flows from
//                                the top below `title-gap`); opt in per slide
//                                when a short body would otherwise leave a
//                                void at the bottom.
//   body-align      : auto    — explicit vertical alignment override for
//                                the body within the area below the
//                                title/subtitle. Pass `top`, `horizon` (=
//                                centred, same as `center-body: true`), or
//                                `bottom` to pin the body to that edge.
//                                When `auto` (default) the alignment is
//                                derived from `center-body`. Setting this
//                                to `bottom` is handy when you want extra
//                                breathing room under the title.
//   scale           : 1.0     — proportional multiplier on all above sizes
//                               (useful to ramp the whole slide up for a 1080p export)
//   corner-x        : auto    — brand-mark distance from page right edge
//                               (falls back to the `_CORNER-X` constant)
//
// Spacing uses `block(below: ..., spacing: 0pt)` to bypass the global
// paragraph spacing, so the eyebrow→title gap is *exactly* what you ask
// for, not paragraph spacing plus your value.
//
// (If you add or rename a knob, also update GUIDE.md §3.3's tuning-knobs
// table so the docs don't drift.)
#let content-slide(
  config: (:),
  eyebrow-text: none,
  title: none,
  title-size: 32pt,
  // 32pt h2: 0.35em line-height keeps the title compact while staying
  // above the descender/ascender collision floor. The same rationale as
  // the title-slide hero (which uses 0.4em at 48pt). Verify on multi-line
  // titles with descender→ascender pairs on consecutive lines.
  title-leading: 0.35em,
  title-weight: 600,                // bump to 700 for a heavier h2
  // Optional one-line subtitle sitting just below the title — a sentence-
  // style elaboration of the title. Pass content (`subtitle: [...]`) to
  // render; leave at `none` to skip and let the body sit directly under
  // the title at `title-gap` distance.
  subtitle: none,
  subtitle-size: 16pt,
  subtitle-gap: 11pt,               // title→subtitle gap (only applied when subtitle is set)
  eyebrow-size: 9pt,
  eyebrow-gap: 5pt,
  title-gap: 16pt,                  // title→body, OR subtitle→body when subtitle is set (ignored when center-body: true or body-align is set)
  center-body: false,               // opt in to vertically centre the body in the area below the title/subtitle
  body-align: auto,                 // explicit body alignment override (`top` | `horizon` | `bottom`); when `auto`, falls back to `center-body`
  scale: 1.0,
  brand-logo: "deeppumas-primary-purple",
  brand-logo-height: _BRAND-LOGO-H, // tunes the corner brand-mark glyph height
  corner-x: auto,                   // brand-mark distance from page right edge
  fill: white,
  footer-card: none,
  composer: auto,
  ..bodies,
) = touying-slide-wrapper(self => {
  self = utils.merge-dicts(
    self,
    config-page(fill: fill, margin: 0pt, header: none, footer: none),
  )

  // Theme-level toggle. When `true` (the default), every content-slide
  // reserves eyebrow space — so the title's top-y is identical whether or
  // not a particular slide uses an eyebrow. When `false`, the eyebrow region
  // is dropped entirely (the title floats up into the reclaimed space) and
  // any slide that still passes `eyebrow-text:` errors out, on the principle
  // that loud failures beat silent inconsistency.
  let eyebrows-enabled = self.store.at("eyebrows-enabled", default: true)
  if eyebrow-text != none and not eyebrows-enabled {
    panic(
      "content-slide was given `eyebrow-text: " + repr(eyebrow-text) + "` " +
      "but the theme has `eyebrows: false`. Either remove `eyebrow-text` from " +
      "this slide or set `eyebrows: true` on `deeppumas-theme(...)`.",
    )
  }

  // ── Fixed eyebrow / title geometry ────────────────────────────────────
  // The eyebrow lives inside a `block(height: eyebrow-reserve, …)` of
  // *exactly* the eyebrow-band height (one font line tall), with the
  // eyebrow text anchored at the top of the block. A `below:` setting of
  // exactly `eyebrow-gap` then sits between the block's bottom and the
  // title block's top.
  //
  // Because the eyebrow block has a rigorously fixed height (and the
  // `below:` gap is bypassed by `spacing: 0pt`), the title-block's top-y
  // is identical on every slide in the deck — **and** the distance from
  // the eyebrow's bottom to the title's top is rigorously `eyebrow-gap`,
  // not "approximately" — regardless of eyebrow text length or whether
  // the slide even has an eyebrow.
  //
  // When `eyebrows: false`, the whole eyebrow block is dropped and the
  // title moves up to the top padding, reclaiming the band.
  let eyebrow-reserve = eyebrow-size * scale * 1.2     // ~1 line height

  // The header stack — eyebrow band + title + (optional) subtitle. The gap
  // below the *last* element of the stack is parameterised because we want
  // it to differ between the two layout modes:
  //   • center-body: true  → no trailing gap; the centred body adds equal
  //     margins on both sides relative to the bottom of this stack.
  //   • center-body: false → `title-gap` is applied as the trailing gap so
  //     the body flows directly below at that distance.
  let header-stack(trailing-gap) = {
    if eyebrows-enabled {
      // Fixed-height block; `align(top + left, …)` anchors the text at
      // the top of the band so the eyebrow's visible top edge is at the
      // page anchor. When `eyebrow-text` is none, the block is empty but
      // still consumes its reserved height — so the title doesn't shift.
      block(
        below: eyebrow-gap * scale,
        spacing: 0pt,
        width: 100%,
        height: eyebrow-reserve,
        align(top + left,
          if eyebrow-text != none {
            text(size: eyebrow-size * scale, weight: 700, tracking: 0.18em, fill: dp-purple,
                 upper(eyebrow-text))
          }),
      )
    }
    if title != none {
      // When there's a subtitle, the gap below the title is `subtitle-gap`;
      // otherwise the title carries `trailing-gap` directly.
      let below-title = if subtitle != none { subtitle-gap * scale } else { trailing-gap }
      block(below: below-title, spacing: 0pt, {
        set par(leading: title-leading)
        text(size: title-size * scale, weight: title-weight, fill: dp-purple-deep, tracking: -0.018em, title)
      })
    }
    if subtitle != none {
      block(below: trailing-gap, spacing: 0pt, {
        text(size: subtitle-size * scale, fill: dp-gray-700, subtitle)
      })
    }
  }

  let body-content = bodies.pos().sum(default: none)

  // Vertical layout resolution. `body-align` (when set) wins; otherwise
  // `center-body: true` maps to `horizon`. When neither is set we fall back
  // to the legacy top-flow layout where the title-gap is applied below the
  // header and the body flows directly below.
  let resolved-body-align = if body-align != auto {
    body-align
  } else if center-body {
    horizon
  } else {
    none
  }
  let inner = if resolved-body-align != none and body-content != none {
    // Header at top, body pinned to `resolved-body-align` inside a 1fr row.
    // `horizon` centres the body; `bottom` parks it at the bottom edge —
    // useful when you want breathing room below the title; `top` is
    // equivalent to legacy top-flow but inside the grid structure so the
    // alignment behaviour stays consistent.
    grid(
      rows: (auto, 1fr),
      row-gutter: 0pt,
      header-stack(0pt),
      align(resolved-body-align, body-content),
    )
  } else {
    header-stack(title-gap * scale)
    body-content
  }

  // ── Why the `block(width: 100%, height: 100%, _page-pad(inner))` wrapper ──
  //
  // Typst's `place(...)` is documented to *"insert an invisible block-level
  // element in the flow"*. So when `_chrome` (which is purely `place(...)`
  // calls for the brand-mark / slide-counter / optional footer-card) sits
  // next to `_page-pad(inner)` (a flow `pad(...)`), Typst wraps the two
  // siblings in an auto-fit content region — and `pad(top: 7.8%, ...)`
  // resolves its ratio inset against *that* auto-fit region, not the page.
  // The eyebrow's y then drifts with body size between slides.
  //
  // Wrapping the pad inside an explicit-size `block(width: 100%, height: 100%)`
  // pins pad's ratio insets to the parent block. Combined with the theme's
  // `config-common(breakable: false)` (which makes Touying wrap the entire
  // body in `block(height: 1fr, …) = page-margin box`), the inner block's
  // `height: 100%` resolves to the page-margin box's height — so
  // `pad(top: 7.8%, ...)` becomes 7.8% × page-height, identical on every
  // slide regardless of body content.
  //
  // See https://typst.app/docs/reference/layout/place/
  // ("Overlaid elements don't take space in the flow of content, but a
  //  `place` call inserts an invisible block-level element in the flow.")
  // and https://typst.app/docs/reference/layout/relative/ — ratios resolve
  // against the *parent container*, which a sibling `place(...)` can
  // re-anchor to an auto-fit subregion.
  //
  // Verification (run from `deeppumas-slides/`):
  //   typst compile --root . --font-path assets/fonts --format png \
  //     --pages 3,4,5,6,7,9 --ppi 300 \
  //     example/deeppumas-acop2024.typ "example/p-{p}.png"
  //   for p in 3 4 5 6 7 9; do
  //     echo -n "p-$p eyebrow y: "
  //     magick example/p-$p.png -crop 800x300+165+0 txt:- \
  //       | grep -E "\(107,1,199" \
  //       | awk '{print $1}' | awk -F, '{print $2}' | sort -u | head -1
  //   done
  //
  //   What it does: renders six content slides (3 & 4 are the two pause-
  //   subslides of slide 3; 5-7 are vanilla content; 9 has a footer-card)
  //   at 300 dpi, then for each PNG probes an 800×300 strip starting at
  //   the content margin (x=165px = 4.7% × 3508px) and finds the FIRST
  //   scan-line containing the brand-purple eyebrow color
  //   (sRGB 107,1,199 = #6B00C7).
  //
  //   Passing result: every probe prints the SAME y-coordinate
  //   (currently y=158, == 7.8% × 1973px / page height at 300 dpi).
  //   If any value differs by more than ~2 px the eyebrow-drift fix has
  //   regressed — the most likely cause is the `block(height: 100%)`
  //   wrapper or the theme's `breakable: false` having been removed.
  let body = {
    _chrome(
      brand-logo: brand-logo,
      brand-logo-height: brand-logo-height,
      brand-corner-x: corner-x,
      footer-card: footer-card,
    )
    block(width: 100%, height: 100%, _page-pad(inner))
  }

  touying-slide(self: self, config: config, composer: composer, body)
})


// Quote slide — dark, arcs, opening quote glyph, big body, attribution row.
#let quote-slide(
  config: (:),
  body,
  attribution-name: none,
  attribution-role: none,
  brand-logo: "deeppumas-primary-white",
  brand-logo-height: _BRAND-LOGO-H, // tunes the corner brand-mark glyph height
  corner-x: auto,                   // brand-mark distance from page right edge
  // Show the decorative brand-purple arcs in the slide background. Pass
  // `false` to fall back to a flat purple fill.
  show-arcs: true,
) = touying-slide-wrapper(self => {
  self = utils.merge-dicts(
    self,
    config-page(fill: dp-purple-deep, margin: 0pt, header: none, footer: none),
  )

  let slide-body = {
    if show-arcs { _arcs(variant: "quote") }
    _slide-num(color: rgb(255, 255, 255, 140))
    _brand-mark(brand-logo, logo-height: brand-logo-height, corner-x: corner-x)

    place(left + horizon, dx: 7%,
      block(width: 86%, {
        text(size: 64pt, weight: 700, fill: dp-purple-mid.transparentize(30%),
             "\u{201C}")
        v(-16pt)
        set par(leading: 0.4em)
        text(size: 24pt, weight: 700, fill: white, tracking: -0.012em, body)
        v(18pt)
        grid(
          columns: (32pt, 1fr),
          column-gutter: 12pt,
          align: (center + horizon, left + horizon),
          circle(radius: 16pt, fill: dp-purple-mid),
          {
            if attribution-name != none {
              text(size: 12pt, weight: 600, fill: white, attribution-name)
            }
            if attribution-role != none {
              linebreak()
              text(size: 10pt, fill: dp-purple-mid, attribution-role)
            }
          },
        )
      }),
    )
  }

  touying-slide(self: self, config: config, slide-body)
})


// Thanks slide — brand background, cobrand logos, big "Thank you." + links.
#let thanks-slide(
  config: (:),
  cobrand: ("pumasai-primary-white", "deeppumas-primary-white"),
  // Global default height for the cobrand logo row. Individual logos can
  // override via the `cobrand:` tuple's dict form — see `cobrand-row` for
  // the absolute (`length`) vs relative (`ratio`) semantics.
  cobrand-height: 32pt,
  title: [Thank you.],
  subtitle: none,
  links: (),
  // Show the decorative brand-purple arcs in the slide background. Pass
  // `false` to fall back to a flat purple fill.
  show-arcs: true,
) = touying-slide-wrapper(self => {
  self = utils.merge-dicts(
    self,
    // Closing slide — same exclusion logic as title-slide and section-slide:
    // not real content, shouldn't bump the counter or the denominator.
    config-common(freeze-slide-counter: true),
    config-page(fill: dp-purple, margin: 0pt, header: none, footer: none),
  )

  let slide-body = {
    if show-arcs { _arcs(variant: "thanks") }

    if cobrand.len() > 0 {
      place(top + left, dx: _PAD-X, dy: _PAD-TOP * 0.5,
        cobrand-row(..cobrand, height: cobrand-height, divider: false))
    }

    place(left + horizon, dx: _PAD-X,
      block(width: _CONTENT-W, {
        set par(leading: 0.4em)
        text(size: 60pt, weight: 700, fill: white, tracking: -0.025em, title)
        if subtitle != none {
          v(12pt)
          set par(leading: 0.4em)
          text(size: 14pt, fill: rgb("#E7D1F5"), subtitle)
        }
        if links.len() > 0 {
          v(26pt)
          stack(
            dir: ltr,
            spacing: 28pt,
            ..links.map(((label, value)) => {
              text(size: 13pt, {
                text(fill: dp-purple-mid, label)
                h(6pt)
                text(fill: white, value)
              })
            }),
          )
        }
      }),
    )
  }

  touying-slide(self: self, config: config, slide-body)
})


// ─── Theme entry point ───────────────────────────────────────────────────

#let deeppumas-theme(
  aspect-ratio: "16-9",
  width: none,
  height: none,
  font: ("Source Sans 3", "Helvetica Neue", "Helvetica", "Arial"),
  // `true`  → every content-slide reserves eyebrow space, so the title's
  //           top-y is fixed deck-wide. Slides without an eyebrow still
  //           consume the same vertical area.
  // `false` → eyebrows are turned off completely; the title floats up into
  //           the reclaimed space. Passing `eyebrow-text:` to any content-slide
  //           is then an error (panics at compile time).
  eyebrows: true,
  body,
) = {
  // Page size: explicit pixel dimensions if both given, otherwise Touying's
  // standard paper for the requested aspect ratio (e.g. 841.89pt × 473.56pt
  // for "16-9"). Layout is expressed in percentages of the page, so any of
  // these sizes reflow correctly.
  let page-args = if width != none and height != none {
    (width: width, height: height)
  } else {
    utils.page-args-from-aspect-ratio(aspect-ratio)
  }

  show: touying-slides.with(
    config-page(..page-args, margin: 0pt, fill: white),
    // `breakable: false` is the Touying-idiomatic way to make % layout
    // ratios resolve cleanly against the page. With the default
    // `breakable: true`, Touying renders the slide body directly into the
    // page flow. Then, if the body mixes `place(...)` (out-of-flow corner
    // chrome) with `pad(top: 7.8%, ...)` (the content padding), Typst's
    // layouter wraps both in an enclosing region whose height is the
    // *content fit* — and `7.8%` resolves against that varying region, not
    // against the page. Effect: eyebrow / title positions drift with body
    // size.
    //
    // With `breakable: false`, Touying instead wraps the body in
    // `components.page-container(...)`, which is
    // `block(width: 100%, height: 1fr, breakable: false, body)`. The
    // `1fr` makes the block fill the page-margin box, so child `%`
    // resolve against a fixed page-sized container — every slide's
    // eyebrow + title land at pixel-identical y.
    //
    // (Why `place()` triggers the auto-fit at all: Typst's `place()` is
    // documented to "insert an invisible block-level element in the
    // flow" — i.e. it does affect surrounding layout even though it
    // contributes no visible space. See the Typst layout reference.)
    //
    // Caveat: `breakable: false` clips slide content that overflows the
    // page rather than spilling onto a new physical page. Touying's
    // `detect-overflow: true` (default) emits a warning when this
    // happens — that's the right tradeoff for slides.
    //
    // Verification: the same pixel-probe recipe at `content-slide`'s
    // body-wrap site (search "Verification" in this file) — every
    // content-slide eyebrow must be on the same scan-line. If you ever
    // try `breakable: true` here, that recipe will show the drift come
    // back.
    config-common(
      slide-fn: content-slide,
      handout: false,
      breakable: false,
      // Touying's `detect-overflow: true` (default, when breakable: false)
      // emits a warning whenever a slide's measured content height is 0pt.
      // Our title-/section-/quote-/thanks-slides are intentionally composed
      // entirely of out-of-flow `place(...)` calls — so they measure as
      // zero-height and produce a noisy "empty content" warning at every
      // compile. Disable the check; we don't rely on the warning anywhere.
      detect-overflow: false,
    ),
    config-methods(
      init: (self: none, body) => {
        set text(font: font, size: 12pt, fill: dp-purple-deep)
        // Tight paragraph spacing — slides use explicit v(...) between
        // elements rather than relying on auto-paragraph gaps.
        set par(leading: 0.55em, spacing: 0.4em)
        show heading: set text(weight: 700, fill: dp-purple-deep)
        body
      },
    ),
    config-colors(
      primary: dp-purple,
      neutral-darkest: dp-purple-deep,
      neutral-lightest: white,
    ),
    // Per-deck flags read by individual slide functions via `self.store`.
    config-store(eyebrows-enabled: eyebrows),
  )

  body
}

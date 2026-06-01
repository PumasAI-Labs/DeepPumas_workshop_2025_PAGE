// ──────────────────────────────────────────────────────────────────────────
//  deeppumas-slides — logo handling
//  Resolves bundled logos by short name, supports user-supplied paths,
//  and exposes inline logo macros for in-text use (à la \LaTeX).
// ──────────────────────────────────────────────────────────────────────────

// Internal: decide if `name-or-path` is a path (contains "/" or ends in ".svg/.png/.pdf")
// or a short name for a bundled logo (e.g. "deeppumas-primary-purple").
#let _is-path(s) = {
  if type(s) != str { return true }                  // anything non-string is "external"
  s.contains("/") or s.ends-with(".svg") or s.ends-with(".png") or s.ends-with(".pdf") or s.ends-with(".jpg") or s.ends-with(".jpeg")
}

#let _bundled-path(name) = "assets/logos/" + name + ".svg"

// Resolve a logo to a Typst `image()` element.
//
// `name-or-path`: either a bundled short name (e.g. `"deeppumas-primary"`) or
// a path to an external image file (relative to the caller's source file).
// `height` / `width` are forwarded to `image()` — pass only one to preserve aspect ratio.
#let logo(name-or-path, height: auto, width: auto, alt: none) = {
  let path = if _is-path(name-or-path) { name-or-path } else { _bundled-path(name-or-path) }
  if alt == none {
    image(path, height: height, width: width)
  } else {
    image(path, height: height, width: width, alt: alt)
  }
}

// ─── Inline brand macros (the \LaTeX-style trick) ────────────────────────
//
// Two flavours are available:
//
//  A. Amplitude-rendered (default).  Best line-flow inside body text.
//     The brand name is typeset in the Amplitude wordmark font, preceded
//     by the small logomark icon. Logomark sits on the text baseline so
//     the surrounding line-height does not change.
//
//        "We're announcing #deeppumas-inline — a learned drift on NLME."
//
//     Available as content values (no parens): `pumas-inline`,
//     `pumas-inline-white`, `pumas-inline-black`, `deeppumas-inline`,
//     `deeppumas-inline-white`, `deeppumas-inline-black`.
//
//  B. SVG wordmark.  If you'd rather use the bundled "logo-with-text"
//     SVG instead of typesetting the name, call `inline-logo("...")`
//     with any bundled name or external path.
//
//        #inline-logo("pumasai-primary-purple", height: 1em)
//
//   To switch flavour A to use the SVG wordmark too, see GUIDE.md.

// Brand-name typesetting. The brand book uses Amplitude at weight 500
// (Medium-ish) for the wordmark, with the brand-identifier half (e.g.
// "AI" / "Deep") visually heavier than the rest. We only have two static
// Amplitude weights bundled in `assets/fonts/`:
//   • Amplitude Book.otf       (Book, ~400)
//   • Amplitude-Bold Regular.ttf (Bold, 700)
// so a real Medium isn't available. We **synthesise** the heavier half
// by rendering it in the Book face with a thin stroke that matches the
// fill colour — this thickens the glyph outlines without going all the
// way to Bold. Stroke width scales with the surrounding font size.
//
// `parts` is an ordered array of (string, weight) tuples — the order
// matches the rendered order:
//   PumasAI    → (Pumas, regular) + (AI, "heavy")
//   DeepPumas  → (Deep, "heavy")  + (Pumas, regular)
#let _brand-text(parts, color, stroke-width: 0.03em) = {
  for (chunk, weight) in parts {
    if weight == "regular" {
      text(font: "Amplitude Book", fill: color, tracking: 0.02em, chunk)
    } else {
      // Synthesised "medium": Book face + thin matching stroke.
      text(
        font: "Amplitude Book",
        fill: color,
        stroke: stroke-width + color,
        tracking: 0.02em,
        chunk,
      )
    }
  }
}

// ───────────────────────────────────────────────────────────────────
//  Two rendering modes.
//
//  Inline mode (default `*-inline` content values):
//    Logo box vertically spans from the descender depth of the
//    surrounding text (the bottom of a "g") up to its cap-top (the
//    top of a "P"). Icon size ≈ 0.95em + the box is baseline-shifted
//    down by 0.2em so its bottom sits at the descender line. Reads
//    cleanly inside running text.
//
//  Centered mode (`*-centered(...)` function calls):
//    Logo box is taller (~1.4em) and centred vertically on the text's
//    cap-line midpoint, so the icon extends both above and below the
//    text — visually echoing the original SVG wordmark layout. Used
//    by `_affil-eyebrow` for institution rows.

#let _brand-render(
  logomark,
  parts,
  color: black,
  logo-height: 0.95em,
  logo-baseline: 0.2em,      // shift logo box DOWN by this amount → bottom sits at descender
  gap: 0.08em,
) = {
  box(baseline: logo-baseline,
      image("assets/logos/" + logomark + ".svg", height: logo-height))
  h(gap)
  _brand-text(parts, color)
}

// ─── Wordmark composition for the two brands ─────────────────────────
//
//  PumasAI: "Pumas" (regular) + "AI" (medium-heavy)
//  DeepPumas: "Deep" (medium-heavy) + "Pumas" (regular)
//
//  i.e. the heavy emphasis is on the *brand-identifier* half: "AI" for
//  PumasAI, "Deep" for DeepPumas. (Earlier versions had this flipped
//  for DeepPumas.)

#let _pumas-parts     = (("Pumas", "regular"), ("AI",    "medium"))
#let _deeppumas-parts = (("Deep",  "medium"),  ("Pumas", "regular"))

// ─── Inline content values ───────────────────────────────────────────

#let pumas-inline = _brand-render(
  "pumasai-logomark-tight-black", _pumas-parts, color: black,
)
#let pumas-inline-purple = _brand-render(
  "pumasai-logomark-tight-purple", _pumas-parts, color: rgb("#6B00C7"),
)
#let pumas-inline-white = _brand-render(
  "pumasai-logomark-tight-white", _pumas-parts, color: white,
)
#let pumas-inline-black = pumas-inline   // alias

#let deeppumas-inline = _brand-render(
  "deeppumas-logomark-tight-purple", _deeppumas-parts, color: rgb("#6B00C7"),
)
#let deeppumas-inline-white = _brand-render(
  "deeppumas-logomark-tight-white", _deeppumas-parts, color: white,
)
#let deeppumas-inline-black = _brand-render(
  "deeppumas-logomark-tight-black", _deeppumas-parts, color: black,
)
#let deeppumas-inline-purple = deeppumas-inline   // alias

// ─── Centered variants (function calls) ──────────────────────────────
// Larger icon, vertically centred on the text — used for institution
// eyebrows on the title slide so the logo+name lockup mimics the
// original SVG wordmark layout.

#let pumas-centered(color: black, scale: 1.0) = _brand-render(
  if color == white { "pumasai-logomark-tight-white" } else if color == rgb("#6B00C7") { "pumasai-logomark-tight-purple" } else { "pumasai-logomark-tight-black" },
  _pumas-parts,
  color: color,
  logo-height: 1.4em * scale,
  // Centre the icon on the text's vertical midpoint:
  //   text midpoint ≈ -0.35em above baseline (half cap-height of ~0.7em)
  //   logo midpoint = box bottom - logo-height/2 = box bottom - 0.7em
  //   want box bottom such that box bottom - 0.7em = -0.35em
  //   ⇒ box bottom = +0.35em (below baseline) ⇒ baseline-shift 0.35em
  logo-baseline: 0.35em,
)

#let deeppumas-centered(color: rgb("#6B00C7"), scale: 1.0) = _brand-render(
  if color == white { "deeppumas-logomark-tight-white" } else if color == black { "deeppumas-logomark-tight-black" } else { "deeppumas-logomark-tight-purple" },
  _deeppumas-parts,
  color: color,
  logo-height: 1.4em * scale,
  logo-baseline: 0.35em,
)

// SVG-based inline logo — use this when you'd rather render the bundled
// "logo-with-text" SVG (or any user-supplied image) than typeset the
// brand name inline. Baseline-aware so it sits cleanly in body text.
#let inline-logo(name-or-path, height: 0.9em, baseline: 0.15em) = box(
  baseline: baseline,
  logo(name-or-path, height: height),
)

// ─── Cobrand row ─────────────────────────────────────────────────────────
// Render one, two, or N logos side-by-side with optional thin vertical
// dividers between them. Each entry can be either:
//   • a string — a bundled name or a path. The logo is rendered at the
//     row's global `height`.
//   • a dictionary `(path: "...", height: ...)` — per-logo override. The
//     `height:` value may be:
//       – a `length` (e.g. `38pt`)  → absolute height, used verbatim;
//       – a `ratio`  (e.g. `90%`)   → height = global `height` × ratio;
//       – omitted                   → same as global (= `100%`).
//     Any other type panics with a message naming the supported types.

#let _normalize-logo-entry(entry, default-height) = {
  let _resolve-height(h) = {
    if type(h) == length { h }
    else if type(h) == ratio { default-height * h }
    else {
      panic(
        "cobrand-row: per-logo `height` must be a length (e.g. 38pt) or a ratio (e.g. 90%), got "
          + repr(h),
      )
    }
  }
  if type(entry) == dictionary {
    let path = entry.at("path", default: entry.at("name", default: none))
    assert(path != none, message: "cobrand-row: each dict entry must have `path` or `name`")
    let h = entry.at("height", default: none)
    (path: path, height: if h == none { default-height } else { _resolve-height(h) })
  } else {
    (path: entry, height: default-height)
  }
}

#let cobrand-row(
  ..logos,
  height: 60pt,
  gap: 28pt,
  divider: true,
  divider-color: rgb("#E3E3E3"),
  divider-height: auto,
) = {
  let entries = logos.pos().map(e => _normalize-logo-entry(e, height))
  if entries.len() == 0 { return }

  let div-h = if divider-height == auto { height * 0.8 } else { divider-height }
  let div = box(width: 1pt, height: div-h, fill: divider-color)

  let items = ()
  for (i, e) in entries.enumerate() {
    if i > 0 and divider {
      items.push(div)
    }
    items.push(logo(e.path, height: e.height))
  }

  stack(
    dir: ltr,
    spacing: gap,
    ..items,
  )
}

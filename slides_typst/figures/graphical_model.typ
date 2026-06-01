// Bayesian-graph schematic for an NLME model: θ at the top, branching to
// per-subject (η_i, x_i, y_i) plates. Reused across MeNets and Augment.

#import "@preview/cetz:0.4.2": canvas, draw

#let _theta-fill = rgb("#5588C8")
#let _eta-fill   = rgb("#D85C5C")
#let _xy-fill    = rgb("#4FA471")
#let _edge       = rgb("#3F66A6")

#let graphical-model-fig = canvas(length: 1cm, {
  import draw: *
  circle((0, 2.5), radius: 0.55, fill: _theta-fill, stroke: 0.6pt + _edge)
  content((0, 2.5), text(size: 14pt, fill: white, weight: 700)[$theta$])

  let subjs = ((-4.5, 0), (-1.5, 0), (1.5, 0))
  for (i, base) in subjs.enumerate() {
    let n = i + 1
    let (bx, by) = base
    let eta = (bx, by)
    let xs  = (bx + 1.2, by - 1.5)
    let ys  = (bx + 1.2, by)

    circle(eta, radius: 0.45, fill: _eta-fill, stroke: 0.6pt + _edge)
    content(eta, text(size: 11pt, fill: white, weight: 700)[$eta_#n$])
    circle(xs, radius: 0.45, fill: _xy-fill, stroke: 0.6pt + _edge)
    content(xs, text(size: 11pt, fill: white, weight: 700)[$x_#n$])
    circle(ys, radius: 0.45, fill: _xy-fill, stroke: 0.6pt + _edge)
    content(ys, text(size: 11pt, fill: white, weight: 700)[$y_#n$])

    let arrow = (mark: (end: (symbol: ">", fill: _edge, scale: 0.4)),
                 stroke: 0.7pt + _edge)
    line((0, 1.95), eta, ..arrow)
    line((0, 1.95), ys, ..arrow)
    line(eta, ys, ..arrow)
    line(xs, ys, ..arrow)
  }
})

// Legend rendered as a small block to drop next to the figure.
#let graphical-model-legend = {
  let _sq(color) = box(width: 10pt, height: 10pt, fill: color,
                       stroke: 0.5pt + _edge)
  stack(spacing: 6pt,
    grid(columns: (auto, auto), column-gutter: 6pt,
      _sq(_theta-fill), [Fixed effects]),
    grid(columns: (auto, auto), column-gutter: 6pt,
      _sq(_eta-fill),   [Random effects]),
    grid(columns: (auto, auto), column-gutter: 6pt,
      _sq(_xy-fill),    [Known quantities]),
  )
}

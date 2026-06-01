// Simple feed-forward neural network: 3 inputs (orange) → 4 hidden (gray)
// → 2 outputs (blue). Fully connected.

#import "@preview/cetz:0.4.2": canvas, draw

#let _input  = rgb("#F2A777")
#let _hidden = rgb("#A8A8B0")
#let _output = rgb("#7DA0CC")
#let _edge   = rgb("#5B5B66")
#let _label  = rgb("#3A3A40")

#let nn-fig = canvas(length: 1cm, {
  import draw: *
  let r = 0.32

  let xs = (0.0, 2.4, 4.8)
  let n_in = 3
  let n_hi = 4
  let n_out = 2

  let col-ys(n) = {
    let span = 2.2
    let step = if n > 1 { span / (n - 1) } else { 0 }
    range(n).map(i => span / 2 - i * step)
  }

  let inp = col-ys(n_in).map(y => (xs.at(0), y))
  let hid = col-ys(n_hi).map(y => (xs.at(1), y))
  let out = col-ys(n_out).map(y => (xs.at(2), y))

  // Edges first (under nodes)
  for a in inp { for b in hid { line(a, b, stroke: 0.3pt + _edge) } }
  for a in hid { for b in out { line(a, b, stroke: 0.3pt + _edge) } }

  // Nodes
  for c in inp { circle(c, radius: r, fill: _input,  stroke: 0.5pt + _edge) }
  for c in hid { circle(c, radius: r, fill: _hidden, stroke: 0.5pt + _edge) }
  for c in out { circle(c, radius: r, fill: _output, stroke: 0.5pt + _edge) }

  // Column labels
  content((xs.at(0), 1.6), text(size: 9pt, fill: _label)[Input])
  content((xs.at(1), 1.8), text(size: 9pt, fill: _label)[Hidden])
  content((xs.at(2), 1.4), text(size: 9pt, fill: _label)[Output])
})

// Figures for the reflowed 07_embeddings deck.
// An embedding model is NOT the encoder half of a VAE — no decoder, no
// reconstruction. These schematics keep that framing.

#import "@preview/cetz:0.4.2": canvas, draw

#let _purple = rgb("#6B00C7")
#let _deep   = rgb("#250044")
#let _mid    = rgb("#AE7FD6")
#let _tint   = rgb("#EBECF7")
#let _border = rgb("#AE7FD6")
#let _gray   = rgb("#5B5B66")

#let _arrow = (mark: (end: (symbol: ">", fill: _purple, scale: 0.55)), stroke: 1.6pt + _purple)

// data → embedding model → vector
#let embedder-fig(input-label: "Text · Image · ...") = canvas(length: 1cm, {
  import draw: *
  rect((0, 0), (3.0, 1.6), radius: 0.18, fill: _tint, stroke: 1.3pt + _border, name: "in")
  content((1.5, 0.8), text(size: 11pt, fill: _gray)[#input-label])

  rect((4.2, 0.0), (8.4, 1.6), radius: 0.18, fill: _purple, stroke: none, name: "emb")
  content((6.3, 0.8), text(size: 12pt, weight: 700, fill: white)[Embedding model])

  let x0 = 9.8
  for i in range(6) {
    rect((x0, 0.0 + i * 0.27), (x0 + 1.0, 0.27 + i * 0.27), fill: white, stroke: 1pt + _border)
  }
  content((x0 + 0.5, 2.05), text(size: 10pt, fill: _gray)[Embedding (vector)])

  line("in.east",  "emb.west", .._arrow)
  line("emb.east", (x0, 0.8), .._arrow)
})

// Embedding space: similar inputs land near each other (3 clusters).
#let similarity-fig() = canvas(length: 1cm, {
  import draw: *
  rect((0, 0), (12, 6.6), radius: 0.25, stroke: 1pt + _border, fill: rgb("#FBFAFE"))
  let cloud = ((0, 0), (0.5, 0.35), (-0.4, 0.25), (0.25, -0.4), (-0.3, -0.25), (0.55, -0.05), (0.05, 0.5), (-0.55, -0.1))
  let dots(cx, cy, col) = {
    for o in cloud { circle((cx + o.at(0), cy + o.at(1)), radius: 0.13, fill: col, stroke: none) }
  }
  dots(2.7, 4.7, _deep)
  dots(9.1, 4.9, _mid)
  dots(5.9, 1.9, _purple)
  content((2.7, 5.9), text(size: 12pt, weight: 600, fill: _gray)[notes about the *liver*])
  content((9.1, 6.1), text(size: 12pt, weight: 600, fill: _gray)[notes about the *heart*])
  content((5.9, 0.8), text(size: 12pt, weight: 600, fill: _gray)[notes about the *kidney*])
})

// Embedding used as a covariate: data → embedding → into the model's prior.
#let embedder-to-model-fig() = canvas(length: 1cm, {
  import draw: *
  rect((0, 0), (2.8, 1.6), radius: 0.18, fill: _tint, stroke: 1.3pt + _border, name: "in")
  content((1.4, 0.8), text(size: 10.5pt, fill: _gray)[Text · Image])

  rect((3.9, 0.0), (7.7, 1.6), radius: 0.18, fill: _purple, stroke: none, name: "emb")
  content((5.8, 0.8), text(size: 11pt, weight: 700, fill: white)[Embedding model])

  let x0 = 8.6
  for i in range(6) {
    rect((x0, 0.0 + i * 0.27), (x0 + 0.8, 0.27 + i * 0.27), fill: white, stroke: 1pt + _border)
  }
  content((x0 + 0.4, 2.05), text(size: 9.5pt, fill: _gray)[vector])

  rect((10.9, 0.0), (15.2, 1.6), radius: 0.18, fill: _deep, stroke: none, name: "mod")
  content((13.05, 0.8), text(size: 10.5pt, weight: 700, fill: white)[random-effect prior])

  line("in.east",  "emb.west", .._arrow)
  line("emb.east", (x0, 0.8), .._arrow)
  line((x0 + 0.8, 0.8), "mod.west", .._arrow)
})

// Encoder/Decoder schematic. Input → Encoder → Latent → Decoder → Output.
// Used across 07_embeddings.

#import "@preview/cetz:0.4.2": canvas, draw

#let _purple = rgb("#6B00C7")
#let _tint = rgb("#EBECF7")
#let _border = rgb("#AE7FD6")
#let _gray = rgb("#5B5B66")

#let encoder-decoder-fig(input-label: "Input", output-label: "Output") = canvas(length: 1cm, {
  import draw: *
  let arrow = (mark: (end: (symbol: ">", fill: _purple, scale: 0.55)),
               stroke: 1.6pt + _purple)

  // Input box
  rect((0, 0), (2.6, 1.6), radius: 0.18, fill: _tint, stroke: 1.3pt + _border, name: "in")
  content((1.3, 0.8), text(size: 11pt, fill: _gray)[#input-label])

  // Encoder trapezoid (approximated as rect)
  rect((3.6, 0.1), (6.4, 1.5), radius: 0.18, fill: _purple, stroke: none, name: "enc")
  content((5.0, 0.8), text(size: 12pt, weight: 700, fill: white)[Encoder])

  // Latent
  rect((7.4, 0.4), (8.8, 1.2), radius: 0.12, fill: rgb("#5C00A3"), stroke: none, name: "lat")
  content((8.1, 0.8), text(size: 10pt, fill: white)[Latent])

  // Decoder trapezoid
  rect((9.8, 0.1), (12.6, 1.5), radius: 0.18, fill: _purple, stroke: none, name: "dec")
  content((11.2, 0.8), text(size: 12pt, weight: 700, fill: white)[Decoder])

  // Output
  rect((13.6, 0), (16.2, 1.6), radius: 0.18, fill: _tint, stroke: 1.3pt + _border, name: "out")
  content((14.9, 0.8), text(size: 11pt, fill: _gray)[#output-label])

  line("in.east",  "enc.west", ..arrow)
  line("enc.east", "lat.west", ..arrow)
  line("lat.east", "dec.west", ..arrow)
  line("dec.east", "out.west", ..arrow)
})

// Encoder-only variant for the embedding-model slide.
#let encoder-only-fig(input-label: "Input") = canvas(length: 1cm, {
  import draw: *
  let arrow = (mark: (end: (symbol: ">", fill: _purple, scale: 0.55)),
               stroke: 1.6pt + _purple)

  rect((0, 0), (2.6, 1.6), radius: 0.18, fill: _tint, stroke: 1.3pt + _border, name: "in")
  content((1.3, 0.8), text(size: 11pt, fill: _gray)[#input-label])

  rect((3.6, 0.1), (6.4, 1.5), radius: 0.18, fill: _purple, stroke: none, name: "enc")
  content((5.0, 0.8), text(size: 12pt, weight: 700, fill: white)[Encoder])

  // Vector latent (stack of cells)
  let x0 = 7.4
  for i in range(6) {
    rect((x0, 0.0 + i * 0.27), (x0 + 1.0, 0.27 + i * 0.27),
         fill: white, stroke: 1pt + _border)
  }
  content((x0 + 0.5, 2.0), text(size: 10pt, fill: _gray)[Latent])

  line("in.east",  "enc.west", ..arrow)
  line("enc.east", (x0, 0.8), ..arrow)
})

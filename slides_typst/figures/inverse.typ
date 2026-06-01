// "The encoder is the inverse of the decoder."
//   Decoder (shared): η → data, the structural/generative model.
//   NLME  : invert that decoder, per subject  (dashed, direct — no separate model).
//   VAE   : a separate encoder model does the inverse in one forward pass.
// Used by 08_genai_model_space (inference intuition slide).

#import "@preview/cetz:0.4.2": canvas, draw

#let _purple = rgb("#6B00C7")
#let _deep   = rgb("#250044")
#let _mid    = rgb("#AE7FD6")
#let _tint   = rgb("#EBECF7")
#let _border = rgb("#AE7FD6")
#let _gray   = rgb("#5B5B66")

#let inverse-fig() = canvas(length: 1cm, {
  import draw: *
  let solid = (mark: (end: (symbol: ">", fill: _purple, scale: 0.5)), stroke: 1.6pt + _purple)
  let dash  = (mark: (end: (symbol: ">", fill: _deep,   scale: 0.5)),
               stroke: (paint: _deep, thickness: 1.6pt, dash: "dashed"))

  // shared anchors: latent η (left) and data (right)
  circle((1.0, 3.0), radius: 0.48, fill: _tint, stroke: 1.2pt + _border, name: "eta")
  content("eta", text(size: 14pt, weight: 700, fill: _deep)[$eta$])
  rect((11.9, 2.45), (13.8, 3.55), radius: 0.12, fill: _tint, stroke: 1.2pt + _border, name: "data")
  content("data", text(size: 11pt, fill: _gray)[data])

  // decoder on top, encoder on the bottom
  rect((4.5, 4.3), (9.3, 5.6), radius: 0.18, fill: _purple, stroke: none, name: "dec")
  content("dec", text(size: 11.5pt, weight: 700, fill: white)[Decoder · structural model])
  rect((4.5, 0.4), (9.3, 1.7), radius: 0.18, fill: _mid, stroke: none, name: "enc")
  content("enc", text(size: 11.5pt, weight: 700, fill: white)[Encoder · a separate model])

  // forward (generate): η → decoder → data
  line("eta", "dec.west", ..solid)
  line("dec.east", "data", ..solid)
  content((7.0, 6.05), text(size: 10pt, fill: _gray)[generate (forward):  $eta arrow.r$ data])

  // NLME: invert the decoder directly, per subject (dashed, straight)
  line("data", "eta", ..dash)
  content((7.0, 3.0), box(fill: white, inset: 3pt,
    text(size: 10pt, weight: 600, fill: _deep)[NLME: invert the decoder · per subject]))

  // VAE: data → separate encoder → η
  line("data", "enc.east", ..solid)
  line("enc.west", "eta", ..solid)
  content((7.0, 0.05), text(size: 10pt, fill: _gray)[VAE: one forward pass through a separate model])
})

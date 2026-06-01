// Bigger Venn used in 01_introduction's closing overview slide.
// Three circles (Machine Learning · Dynamics · Statistics) with numbered
// labels in each region: 1 = single discipline, 3-4 = pairwise overlap,
// 5 = triple intersection (DeepNLME).

#import "@preview/cetz:0.4.2": canvas, draw

#let _fill = rgb(80, 130, 200, 75)
#let _stroke = rgb("#3F66A6")
#let _label = rgb("#1F3A5F")
#let _badge = rgb("#3F66A6")

#let _num(n) = box(
  width: 22pt, height: 22pt,
  fill: rgb("#82A8CC"),
  stroke: 0.6pt + _stroke,
  radius: 50%,
  inset: 4pt,
  align(center + horizon, text(size: 11pt, weight: 700, fill: white, [#n]))
)

#let big-venn-fig = canvas(length: 1cm, {
  import draw: *

  let r = 3.0
  let d = 2.7
  let h = d * calc.sqrt(3) / 2

  let ml = (0,        h * 2/3)
  let dy = (-d / 2,  -h / 3)
  let st = ( d / 2,  -h / 3)

  for c in (ml, dy, st) {
    circle(c, radius: r, fill: _fill, stroke: 1pt + _stroke)
  }

  // Single-discipline labels (deep into each cap) + a "1" badge
  content((ml.at(0), ml.at(1) + 1.45), text(size: 13pt, fill: _label)[Machine Learning])
  content((ml.at(0), ml.at(1) + 0.95), _num(2))

  content((dy.at(0) - 0.5, dy.at(1) - 1.5), text(size: 12pt, fill: _label)[Dynamics])
  content((dy.at(0) - 0.5, dy.at(1) - 1.95), _num(1))

  content((st.at(0) + 0.5, st.at(1) - 1.5), text(size: 12pt, fill: _label)[Statistics])
  content((st.at(0) + 0.5, st.at(1) - 1.95), _num(1))

  // Pairwise overlaps
  // ML ∩ Dyn → UDEs
  content(((ml.at(0) + dy.at(0)) / 2 - 0.6, (ml.at(1) + dy.at(1)) / 2 + 0.25),
          text(size: 12pt, weight: 600, fill: _label)[UDEs])
  content(((ml.at(0) + dy.at(0)) / 2 - 0.6, (ml.at(1) + dy.at(1)) / 2 - 0.25), _num(3))

  // ML ∩ Stat → GenAI
  content(((ml.at(0) + st.at(0)) / 2 + 0.6, (ml.at(1) + st.at(1)) / 2 + 0.25),
          text(size: 12pt, fill: _label)[GenAI])
  content(((ml.at(0) + st.at(0)) / 2 + 0.6, (ml.at(1) + st.at(1)) / 2 - 0.25), _num(4))

  // Dyn ∩ Stat → NLME
  content(((dy.at(0) + st.at(0)) / 2, (dy.at(1) + st.at(1)) / 2 - 0.8),
          text(size: 12pt, fill: _label)[NLME])
  content(((dy.at(0) + st.at(0)) / 2, (dy.at(1) + st.at(1)) / 2 - 1.25), _num(1))

  // Centre — DeepNLME
  let centre-y = (ml.at(1) + dy.at(1) + st.at(1)) / 3
  content((0, centre-y + 0.3), text(size: 14pt, weight: 600, fill: _label)[DeepNLME])
  content((0, centre-y - 0.25), _num(5))
})

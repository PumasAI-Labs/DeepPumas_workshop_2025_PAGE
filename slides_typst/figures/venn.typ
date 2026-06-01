// Three-circle Venn: Machine Learning · Dynamics · Statistics.
// Overlap zones: UDEs (ML∩Dyn) · MeNets (ML∩Stat) · NLME (Dyn∩Stat).
// Centre = DeepPumas. Checkmarks above the three discipline labels and the
// NLME overlap, matching the source pptx.

#import "@preview/cetz:0.4.2": canvas, draw

#let _circle-fill = rgb(80, 130, 200, 70)
#let _circle-stroke = rgb("#3F66A6")
#let _check = rgb("#5E8C3A")
#let _label = rgb("#1F3A5F")

#let venn-fig = canvas(length: 1cm, {
  import draw: *

  let r = 2.6                // circle radius
  let d = 2.4                // centre-to-centre distance (< r => good overlap)
  let h = d * calc.sqrt(3) / 2

  // Equilateral triangle of circle centres (ML up top, Dyn bottom-left, Stat bottom-right)
  let ml = (0,        h * 2/3)
  let dy = (-d / 2,  -h / 3)
  let st = ( d / 2,  -h / 3)

  // Circles
  for c in (ml, dy, st) {
    circle(c, radius: r, fill: _circle-fill, stroke: 1pt + _circle-stroke)
  }

  // ── Discipline labels, deep inside each circle, with the check above ──
  // Top circle (ML): label well above the centre, in the "free" cap above the
  // pairwise lenses.
  let ml-label = (ml.at(0), ml.at(1) + 1.55)
  content((ml-label.at(0), ml-label.at(1) + 0.55),
          text(size: 15pt, weight: 700, fill: _check)[#sym.checkmark])
  content(ml-label, text(size: 13pt, fill: _label)[Machine Learning])

  // Bottom-left (Dynamics): label in the bottom-left "free" cap.
  let dy-label = (dy.at(0) - 1.05, dy.at(1) - 1.25)
  content((dy-label.at(0) + 0.0, dy-label.at(1) + 0.55),
          text(size: 15pt, weight: 700, fill: _check)[#sym.checkmark])
  content(dy-label, text(size: 13pt, fill: _label)[Dynamics])

  // Bottom-right (Statistics): label in the bottom-right "free" cap.
  let st-label = (st.at(0) + 1.05, st.at(1) - 1.25)
  content((st-label.at(0) + 0.0, st-label.at(1) + 0.55),
          text(size: 15pt, weight: 700, fill: _check)[#sym.checkmark])
  content(st-label, text(size: 13pt, fill: _label)[Statistics])

  // ── Pairwise overlap labels (in the lenses, off-centre away from the middle) ──
  // ML ∩ Dyn → UDEs (top-left lens). UDEs is the lobe we focus on → bold.
  let udes = ((ml.at(0) + dy.at(0)) / 2 - 0.7, (ml.at(1) + dy.at(1)) / 2 + 0.2)
  content(udes, text(size: 14pt, weight: 700, fill: _label)[UDEs])

  // ML ∩ Stat → MeNets (top-right lens).
  let menets = ((ml.at(0) + st.at(0)) / 2 + 0.7, (ml.at(1) + st.at(1)) / 2 + 0.2)
  content(menets, text(size: 13pt, fill: _label)[MeNets])

  // Dyn ∩ Stat → NLME (bottom lens). Plus a check.
  let nlme = ((dy.at(0) + st.at(0)) / 2, (dy.at(1) + st.at(1)) / 2 - 0.85)
  content((nlme.at(0), nlme.at(1) + 0.55),
          text(size: 14pt, weight: 700, fill: _check)[#sym.checkmark])
  content(nlme, text(size: 13pt, fill: _label)[NLME])

  // ── Centre — DeepPumas ──
  let centre = (0, (ml.at(1) + dy.at(1) + st.at(1)) / 3)
  content(centre, text(size: 15pt, weight: 600, fill: _label)[DeepPumas])
})

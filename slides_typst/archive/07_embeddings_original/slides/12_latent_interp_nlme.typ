#import "../../deeppumas-slides/lib.typ": *

#content-slide(
  eyebrow-text: "Latent variables",
  title: [What do "latent variables" represent?],
)[
  #v(8pt)
  #sub-eyebrow[NLME]
  #v(6pt)
  Interpretation comes from #strong[structural constraints] --- e.g.
  $eta_1$ is a multiplicative deviation on clearance because the model
  literally writes $C L_i = C L thick exp(eta_(i,1))$.

  #v(10pt)
  #outline-card(stroke-weight: "outer", body-size: 14pt)[
    These interpretations can fool us.
  ]

  #v(12pt)
  #sub-eyebrow[Without structural constraints]
  #v(6pt)
  $eta$ has no built-in meaning --- it is whatever direction in latent
  space the fit found useful.
]

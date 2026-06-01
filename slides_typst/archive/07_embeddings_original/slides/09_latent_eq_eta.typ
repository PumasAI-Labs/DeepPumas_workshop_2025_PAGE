#import "../../deeppumas-slides/lib.typ": *

#content-slide(
  eyebrow-text: "Equivalence",
  title: [Latent variables \= random effects],
)[
  #v(20pt)
  #align(center, brand-callout([How do we fit?])[
    #strong[Maximise the marginal likelihood] --- marginalise over $bold(eta)$.
    #v(4pt)
    $ cal(L)(theta) = integral p(bold(d v) | theta, bold(eta), bold(x)) thick p(bold(eta) | theta) thick d bold(eta) $
  ])
]

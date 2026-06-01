#import "../../deeppumas-slides/lib.typ": *

#content-slide(
  eyebrow-text: "Definitions",
  title: [Generative models],
)[
  #v(4pt)
  #sub-eyebrow[Definitions]
  #v(4pt)
  - $bold(z)$ : latent variables of dimension $d$.
  - $bold(y)$ : observed response/data.
  - $bold(y)_g$ : generated/simulated/synthetic response/data.

  #v(10pt)
  #sub-eyebrow[Model]
  #v(4pt)
  #align(center)[
    $ bold(y)_g &= f(bold(z)) + bold(epsilon) \
      bold(z) &tilde cal(N)(0, I_(d times d)) \
      epsilon_i &tilde cal(N)(0, sigma^2) $
  ]

  #v(10pt)
  #strong[Objective]: choose $f$ such that the distribution of $bold(y)_g$
  is close to the distribution of the observed data $bold(y)$.
]

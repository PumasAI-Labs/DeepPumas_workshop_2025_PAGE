#import "../../deeppumas-slides/lib.typ": *

#content-slide(
  eyebrow-text: "Definitions",
  title: [Conditional generative models],
)[
  #v(4pt)
  #sub-eyebrow[Definitions]
  #v(4pt)
  - $bold(z)$ : latent variables of dimension $d$.
  - $bold(x)$ : observed covariates.
  - $bold(y)$ : observed response.
  - $bold(y)_g$ : generated/simulated/synthetic response.

  #v(10pt)
  #sub-eyebrow[Model]
  #v(4pt)
  #align(center)[
    $ bold(y)_g &= f(bold(z), bold(x)) + bold(epsilon) \
      bold(z) &tilde cal(N)(0, I_(d times d)) \
      epsilon_i &tilde cal(N)(0, sigma^2) $
  ]

  #v(10pt)
  #strong[Objective]: choose $f$ so the conditional distribution of
  $bold(y)_g | bold(x)$ is close to that of $bold(y) | bold(x)$.
]

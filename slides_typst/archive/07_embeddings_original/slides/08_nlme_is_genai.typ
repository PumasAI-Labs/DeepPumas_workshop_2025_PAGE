#import "../../deeppumas-slides/lib.typ": *

#content-slide(
  eyebrow-text: "Bridge",
  title: [NLME is generative AI!],
)[
  #v(4pt)
  #sub-eyebrow[Definitions]
  #v(4pt)
  - $bold(eta)$ : latent variables of dimension $d$, covariance $Omega$.
  - $bold(x)$ : observed covariates.
  - $bold(d v)$ : observed response.
  - $bold(d v)_g$ : generated/simulated/synthetic response.

  #v(10pt)
  #sub-eyebrow[Model]
  #v(4pt)
  #align(center)[
    $ bold(d v)_g &= f_theta (bold(eta), bold(x)) + bold(epsilon) \
      bold(eta) &tilde cal(N)(0, Omega) \
      epsilon_i &tilde cal(N)(0, sigma^2) $
  ]

  #v(10pt)
  #align(center, pill[
    A population NLME #emph[is] a conditional generative model.
  ])
]

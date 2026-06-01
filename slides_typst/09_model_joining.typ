// ─────────────────────────────────────────────────────────────────────────
//  Day 2 — Joining models: information transfer without the joint fit
//
//  Capstone for the day. Builds on the model-space deck (NLME random effects
//  are a latent space; you can reshape / replace the prior over them) and lands
//  on a concrete, available-today move: fit endpoint models INDEPENDENTLY, then
//  learn a JOINT PRIOR over their random effects post-hoc (Gaussian or a
//  Normalizing Flow) with `replace_randeffs_dist`. Information then flows across
//  endpoints at prediction time — no joint refit, no structural change.
//
//  Source: ModelJoining_PAGE2026/poster (poster.typ, NARRATIVE.md, figures),
//  condition_align.qmd, thoughts.md. Frames the hands-on joining exercise.
//
//  Compile:  ./compile.sh 09_model_joining.typ
// ─────────────────────────────────────────────────────────────────────────

#import "deeppumas-slides/lib.typ": *

#show: deeppumas-theme.with(aspect-ratio: "16-9")

// ── Cover ─────────────────────────────────────────────────────────────────
#title-slide(
  conference: "DeepPumas workshop · PAGE 2026 · Day 2",
  cobrand: ("pumasai-primary-black", "deeppumas-primary-purple"),
  title: [Joining models],
  subtitle: [Information transfer across endpoints — without ever fitting them together],
  affiliations: (
    (label: "PumasAI", authors: [Niklas Korsbo]),
  ),
  affiliations-layout: "stacked",
)

// ── The need ────────────────────────────────────────────────────────────
#content-slide(
  eyebrow-text: "The need",
  title: [Use the biomarkers to sharpen the endpoint],
)[
  #v(6pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 26pt,
    align: (left + horizon, left + horizon),
    [
      #sub-eyebrow[A recurring ask]
      Nearly half of our projects want *longitudinal biomarkers to improve a
      primary-endpoint prediction*. AST, ALT, imaging, dozens of markers.

      #v(6pt)
      The textbook answer: a *joint NLME model* with correlated random effects
      across all endpoints.
    ],
    outline-card(header: [...but the joint fit is a wall], stroke-weight: "outer")[
      #v(3pt)
      #set text(size: 15pt)
      - 20 biomarkers → 40+ random effects, thousands of parameters. *FOCE will
        not fit that.* Even 4 endpoints gets hard.
      - Tangled, slow, hard-to-iterate workflows.
      - *Brittle* once neural components enter the structural model.
    ],
  )
]

// ── The compromise + its cost ───────────────────────────────────────────
#content-slide(
  eyebrow-text: "The compromise",
  title: [Fit each endpoint on its own — and pay one price],
)[
  #v(8pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 26pt,
    align: (left + horizon, left + horizon),
    [
      Fit the endpoints *independently* (or conditionally independently, given
      PK). Suddenly everything is easy: fast, plannable, parallelisable across a
      team, robust.
    ],
    outline-card(header: [The cost], stroke-weight: "outer")[
      #v(3pt)
      #set text(size: 15pt)
      Independent fits assume *independent priors* over the random effects — even
      though the true effects are *coupled*. You have thrown away the
      cross-endpoint dependence.
    ],
  )
  #v(10pt)
  #align(center, brand-callout[The whole trick][
    That discarded dependence is a *misspecified prior* — and the prior is the
    one thing we can fix *after the fact*, without touching the structural models.
  ])
]

// ── Assumption ladder: the integral splitting step by step ───────────────
#let _astep(label) = block(below: 4pt, text(size: 13pt, weight: 700, fill: dp-purple, label))

#content-slide(
  eyebrow-text: "From one joint fit to independent fits",
  title: [Four assumptions separate the joint likelihood — we relax one],
)[
  #v(6pt)
  #set text(size: 16pt)
  Start from the joint marginal likelihood of two endpoints $X, Y$ — the same
  integral as before, now over *both*:
  $ p(Y, X | theta) = integral p(Y, X | eta, theta) thin p(eta | theta) thin dif eta $
  #v(5pt)
  #_astep[A1 · conditional independence given the random effects $eta$]
  $ = integral p(Y | eta, theta) thin p(X | eta, theta) thin p(eta | theta) thin dif eta $
  #v(5pt)
  #_astep[A2, A3 · separate parameters $theta = (theta_Y, theta_X)$ and latent spaces $eta = (eta_Y, eta_X)$]
  $ = integral.double p(Y | eta_Y, theta_Y) thin p(X | eta_X, theta_X) thin p(eta_Y, eta_X | theta) thin dif eta_Y dif eta_X $
  #v(8pt)
  #brand-callout[A4 · prior independence — the only one we relax][
    Assume $p(eta_Y, eta_X | theta) = p(eta_Y | theta_Y) thin p(eta_X | theta_X)$ and the
    integral *factorises completely* into two models you can fit in isolation:
    #v(3pt)
    $ p(Y, X | theta) = p(Y | theta_Y) thin dot thin p(X | theta_X) $
  ]
]

// ── The correlational bridge (schematic) ─────────────────────────────────
#content-slide(
  eyebrow-text: "The correlational bridge",
  title: [The data carry the coupling — even when the prior doesn't],
)[
  #v(2pt)
  #align(center, image("mj_figures/fig-joint-schematic.png", height: 50%))
  #v(6pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 24pt,
    align: (left + top, left + top),
    [
      #sub-eyebrow[Left — what A4 assumes]
      Marginals × marginals: circular, no coupling between $bold(eta)_X$ and
      $bold(eta)_Y$.
    ],
    [
      #sub-eyebrow[Right — what the data carry]
      Tilt and coupling. Each patient's *posteriors* already show it. That gap is
      the misspecification we rectify.
    ],
  )
]

// ── How: refit a joint prior on the posteriors ───────────────────────────
#content-slide(
  eyebrow-text: "The move",
  title: [Re-fit one joint prior on the per-patient posteriors],
)[
  #v(4pt)
  The coupling shows up *consistently* in the per-patient posteriors — so it
  belongs in the *prior*. Fit a flexible density $p_phi$ to the joint those
  posteriors imply:
  #v(4pt)
  $ underbrace(tilde(p)(eta_Y, eta_X), "target") = EE_((Y,X)) [ underbrace(p(eta_Y | Y, theta_Y), "independent fit") thin underbrace(p(eta_X | X, theta_X), "independent fit") ] $
  #v(4pt)
  #align(center, text(size: 14pt, fill: dp-gray-700)[
    Each factor is independent by design — yet $tilde(p)$ is *correlated*, because both observations come from the *same patient*.
  ])
  #v(10pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 24pt,
    align: (left + horizon, left + top),
    [
      Replace the independent priors with $p_phi approx tilde(p)$ — one call, the
      structural models left untouched:
      #v(6pt)
      #align(center, pill[`replace_randeffs_dist(...)`])
    ],
    outline-card(header: [Two density families], stroke-weight: "outer")[
      #v(3pt)
      #set text(size: 15pt)
      #table(
        columns: (auto, 1fr),
        stroke: none,
        inset: (x: 4pt, y: 4pt),
        [*Gaussian*], [linear cross-correlation only],
        [*Normalizing Flow*], [non-linear, non-Gaussian; no family to specify],
      )
      #v(2pt)
      Same supervision, different summary of it.
    ],
  )
]

// ── The recipe ────────────────────────────────────────────────────────────
#content-slide(
  eyebrow-text: "The recipe",
  title: [Four steps, structural models untouched],
)[
  #v(10pt)
  #numbered-list(
    [*Fit each endpoint independently.* Share structure first (e.g. fit PK, carry its typical values into the PD models as fixed).],
    [*Compute per-patient Laplace posteriors* of the random effects on the training subjects.],
    [*Fit a joint prior* $p_phi(bold(eta))$ on those posteriors — a Gaussian, or a Normalizing Flow.],
    [*Plug the learned prior back in* and evaluate on held-out test subjects.],
  )
  #v(10pt)
  #align(center, brand-callout[No joint fit, ever][
    The structural models, observation models, and $theta$ never change. Only the
    prior over the random effects is replaced.
  ])
]

// Results are demonstrated live in the hands-on exercise (07_model_joining.jl):
// fit two endpoints independently, see the correlation in the EBEs, join with
// replace_randeffs_dist, predict the sparse endpoint from the other. The full
// quantitative benchmarks (post-hoc vs the joint fit; Gaussian vs flow on
// non-Gaussian truth) live on the poster — pointed to from the closing slide.

// ── Why post-hoc (workflow) ──────────────────────────────────────────────
#content-slide(
  eyebrow-text: "Why join post-hoc",
  title: [The workflow win is the real win],
)[
  #v(6pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 22pt,
    row-gutter: 12pt,
    align: (left + top, left + top),
    outline-card(header: [Plannable], stroke-weight: "outer", body-size: 14pt)[
      Each endpoint owns its model, data, diagnostics. Joining is a separate,
      restartable step.
    ],
    outline-card(header: [Scalable — and the reason we built it], stroke-weight: "outer", body-size: 14pt)[
      Joint fits with neural components are slow and initialisation-sensitive.
      Modular joining sidesteps that. (DeepNLME.)
    ],
    outline-card(header: [Reusable], stroke-weight: "outer", body-size: 14pt)[
      Re-couple a model from a past study without touching its structure — the
      flow can even fix a mildly misspecified prior.
    ],
    outline-card(header: [Mixed families], stroke-weight: "outer", body-size: 14pt)[
      Couple an NLME with *any* latent-variable model that exposes a tractable
      posterior.
    ],
  )
  #v(8pt)
  #align(center, text(size: 14pt, fill: dp-gray-700)[
    A *correlational* bridge — within-population dependence, not causal. And it is already in DeepPumas, today.
  ])
]

// ── Your turn — the exercise (results live here, not on slides) ───────────
#content-slide(
  eyebrow-text: "Your turn",
  title: [Let an early biomarker predict a late endpoint],
)[
  #v(6pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 26pt,
    align: (left + horizon, left + top),
    [
      One drug, two endpoints — a *fast biomarker* (sampled early, densely) and a
      *slow clinical endpoint* (sampled late, sparsely). In the exercise you:
      #v(4pt)
      #numbered-list(
        [fit an IDR model to *each endpoint independently*,],
        [scatter the per-subject *EBEs* — the correlation is already in the data,],
        [*join* them with `replace_randeffs_dist`,],
        [predict a new patient's *clinical endpoint from the biomarker alone*.],
      )
    ],
    outline-card(header: [The takeaway], stroke-weight: "outer")[
      #v(3pt)
      #set text(size: 15pt)
      Nothing about the two models changed. The coupling was *in the data all
      along* — joining moves it into the prior, where predictions can use it.
    ],
  )
  #v(10pt)
  #align(center, brand-callout[Want the full benchmarks?][
    How close post-hoc joining gets to the *full joint fit*, and where a *flow*
    beats a Gaussian on non-Gaussian structure — see the *poster*.
  ])
]

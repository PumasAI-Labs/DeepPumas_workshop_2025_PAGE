// ─────────────────────────────────────────────────────────────────────────
//  Day 2 — Generative AI and the model design space
//
//  The conceptual backbone for the DeepNLME session: NLME *is* a conditional
//  generative model; VAEs and NLME maximise the same marginal likelihood and
//  differ only in how they handle the latent variable. From there we lay out a
//  2-D map (mechanism ↔ data-driven  ×  how individual variation is inferred)
//  and locate ODE / UDE / NODE / Latent UDE / Latent NODE / VAE / NLME and the
//  DeepNLME band (a dial spanning hybrid → fully neural) as neighbours
//  reachable by single moves.
//
//  Sources: context/condition_align.qmd, ViralDynamics 04-generative-ai.qmd,
//  08-JSM slides 10_bridge / 11_image_vae.
//
//  Compile:  ./compile.sh 08_genai_model_space.typ
// ─────────────────────────────────────────────────────────────────────────

#import "deeppumas-slides/lib.typ": *
#import "@preview/cetz:0.4.2"
#import "figures/inverse.typ": inverse-fig

#show: deeppumas-theme.with(aspect-ratio: "16-9")

// ── Cover ─────────────────────────────────────────────────────────────────
#title-slide(
  conference: "DeepPumas workshop · PAGE 2026 · Day 2",
  cobrand: ("pumasai-primary-black", "deeppumas-primary-purple"),
  title: [Generative AI and the\ model design space],
  subtitle: [Where does DeepNLME live, and what can we borrow from machine learning?],
  affiliations: (
    (label: "PumasAI", authors: [Niklas Korsbo]),
  ),
  affiliations-layout: "stacked",
)

// ── Motivation: the tools feel separate, but they're one space ─────────────
#content-slide(
  eyebrow-text: "Where we are",
  title: [So far we have collected a box of seemingly separate tools],
)[
  #v(6pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 26pt,
    align: (left + horizon, left + horizon),
    [
      #sub-eyebrow[Yesterday and this morning]
      - *ODEs* — mechanism we write down by hand
      - *UDEs / NODEs* — let a neural net learn part (or all) of the dynamics
      - *Random effects* — let each subject differ

      #v(6pt)
      Each felt like its own trick, with its own fitting story.
    ],
    [
      #sub-eyebrow[The claim of this session]
      They are all *points in a single design space*.

      #v(6pt)
      And the most powerful moves through that space are exactly the ideas that
      power *generative AI*.
      #v(8pt)
      #brand-callout[One map][
        DeepNLME is not a new gadget — it is one *coordinate* on a map that also
        contains VAEs and Latent Neural ODEs.
      ]
    ],
  )
]

// ── The goal, visually: match a distribution (even over faces) ─────────────
#content-slide(
  eyebrow-text: "Generative AI",
  title: [Match the data's distribution — in any space],
)[
  #v(4pt)
  #grid(
    columns: (1.15fr, 1fr),
    column-gutter: 28pt,
    align: (center + horizon, center + horizon),
    [
      #image("img/observed_synthetic_contour.png", width: 86%)
      #v(2pt)
      #text(size: 13pt, fill: dp-gray-700)[Generated samples fall in the *same distribution* as the real data.]
    ],
    [
      #grid(
        columns: (1fr, 1fr), column-gutter: 8pt,
        image("img/people_1.png", width: 100%),
        image("img/people_2.png", width: 100%),
      )
      #v(4pt)
      #text(size: 13pt, fill: dp-gray-700)[...even when the space is *faces* — these people don't exist.]
    ],
  )
  #v(8pt)
  #align(center, brand-callout[A distribution over a very different space][
    A generative model learns the *distribution* of the data — scalars, time
    courses, or images of faces — and draws new samples from it.
  ])
]

// ── What is a generative model ─────────────────────────────────────────────
#content-slide(
  eyebrow-text: "Generative AI",
  title: [A generative model invents the unobserved part],
)[
  #v(6pt)
  #grid(
    columns: (1.1fr, 1fr),
    column-gutter: 26pt,
    align: (left + horizon, left + horizon),
    [
      #sub-eyebrow[The catch]
      Data is a mix of
      - *observed* quantities — pixels, tokens, concentrations
      - *unobserved* quantities — the face, the meaning, the patient

      A model has to invent the unobserved part before it can generate the rest.
    ],
    outline-card(header: [The latent-variable recipe], stroke-weight: "outer")[
      #v(4pt)
      $ bold(y) &= f(bold(z)) + bold(epsilon) \
        bold(z) &~ cal(N)(0, I) $
      #v(6pt)
      #set text(size: 14pt)
      Draw an unobserved *latent* $bold(z)$, push it through a generator $f$, add
      noise. Train $f$ so generated $bold(y)$ matches real data in distribution.
    ],
  )
]

// ── NLME is a (conditional) generative model ───────────────────────────────
#content-slide(
  eyebrow-text: "The reframe",
  title: [An NLME model is a conditional generative model],
)[
  #v(6pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 26pt,
    align: (center + horizon, center + horizon),
    outline-card(header: [Generative model], header-align: center, stroke-weight: "outer")[
      #v(4pt)
      $ bold(y) &= f(bold(z)) + bold(epsilon) \
        bold(z) &~ cal(N)(0, I) $
      #v(6pt)
      #set text(size: 14pt)
      latent $bold(z)$ — "what we can't see"
    ],
    outline-card(header: [NLME model], header-align: center, stroke-weight: "outer")[
      #v(4pt)
      $ "dv" &= f_theta(bold(eta), bold(x)) + bold(epsilon) \
        bold(eta) &~ cal(N)(0, Omega) $
      #v(6pt)
      #set text(size: 14pt)
      random effects $bold(eta)$ — "how this subject differs"
    ],
  )
  #v(10pt)
  #align(center, brand-callout[Same shape][
    The *random effects are the latent variables*. The structural model $f_theta$
    is the generator. We just *condition* on covariates and dose $bold(x)$.
  ])
]

// ── The shared objective: the marginal likelihood ──────────────────────────
#content-slide(
  eyebrow-text: "Going one level deeper",
  title: [NLME and VAEs maximise the same objective],
)[
  #v(8pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 28pt,
    align: (center + horizon, center + horizon),
    outline-card(header: [NLME], header-align: center, stroke-weight: "outer")[
      #v(4pt)
      $ p(bold(y) | bold(x)) = integral p(bold(y) | bold(eta), bold(x)) thin p(bold(eta)) thin d bold(eta) $
      #v(4pt)
      marginalise out the random effects $bold(eta)$
    ],
    outline-card(header: [Variational auto-encoder], header-align: center, stroke-weight: "outer")[
      #v(4pt)
      $ p(bold(x)) = integral p(bold(x) | bold(z)) thin p(bold(z)) thin d bold(z) $
      #v(4pt)
      marginalise out the latent variables $bold(z)$
    ],
  )
  #v(1fr)
  #align(center, brand-callout[The same target][
    Both go after the *marginal likelihood* — integrating over an unobserved
    latent. Neither reaches it exactly: NLME approximates the integral; the VAE
    maximises a *lower bound* on it (the ELBO). The latent variables #emph[are]
    the random effects: $bold(z) equiv bold(eta)$.
  ])
]

// ── The integral is intractable ────────────────────────────────────────────
#content-slide(
  eyebrow-text: "Where methods diverge",
  title: [That integral is the whole problem],
)[
  #v(6pt)
  $ p(bold(y) | bold(x)) = integral underbrace(p(bold(y) | bold(eta), bold(x)), "generator / structural model") thin underbrace(p(bold(eta)), "prior on latents") thin d bold(eta) $
  #v(10pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 26pt,
    align: (left + top, left + top),
    [
      #sub-eyebrow[Why it's hard]
      We never observe $bold(eta)$. To score a model we must integrate it out —
      and that integral has no closed form for a nonlinear $f_theta$.
    ],
    [
      #sub-eyebrow[The key insight]
      Every method below is just a *different way to approximate this one
      integral*. That choice is what separates NLME, VAEs and Latent NODEs —
      not the equation.
    ],
  )
]

// ── Two ways to handle the latent — the amortization axis ──────────────────
#content-slide(
  eyebrow-text: "Axis 1 — inference",
  title: [Two ways to pin down the latent variable],
)[
  #v(4pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 24pt,
    align: (left + top, left + top),
    outline-card(header: [A posterior per subject — NLME], stroke-weight: "outer")[
      #v(3pt)
      Each subject keeps *its own* posterior, by Bayes' rule:
      $ p(bold(eta)_i | bold(y)_i, bold(x)_i) prop p(bold(y)_i | bold(eta)_i, bold(x)_i) thin p(bold(eta)_i) $
      #v(4pt)
      #set text(size: 14pt)
      Represented *per subject* — by FOCE, Laplace, EM/SAEM or MCMC. The method
      varies; the *per-patient representation* does not.
    ],
    outline-card(header: [Amortised inference — VAE], stroke-weight: "outer")[
      #v(3pt)
      One shared *encoder* outputs the *parameters* of each subject's posterior:
      $ q(bold(eta)_i thin ";" thin lambda_phi (bold(y)_i, bold(x)_i)) $
      #v(4pt)
      #set text(size: 14pt)
      The parameters $lambda$ are a *deterministic function of the data*; the
      family of $q$ is free — Gaussian, flow, mixture. One model for all
      subjects, trained once (ELBO). *No per-subject fit.*
    ],
  )
  #v(8pt)
  #align(center, pill[The shift: from *representing* each patient's posterior individually → *predicting* it with one shared model])
]

// ── Encoder = inverse of the decoder (intuition + the trade-off) ──────────
#content-slide(
  eyebrow-text: "Axis 1 — inference, intuitively",
  title: [The encoder is the inverse of the decoder],
)[
  #v(4pt)
  #align(center, box(width: 92%, inverse-fig()))
  #v(8pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 26pt,
    align: (left + top, left + top),
    outline-card(header: [NLME — invert the decoder], stroke-weight: "outer")[
      #set text(size: 14.5pt)
      Solve the decoder's *inverse for each subject* — no extra model. *Less bias*
      (it really solves the inverse), but a *per-subject* optimisation.
    ],
    outline-card(header: [VAE — a separate encoder], stroke-weight: "outer")[
      #set text(size: 14.5pt)
      Train *one* model to do the inverse in a forward pass. *Scales* to very
      large problems, at the cost of *more bias* (the amortisation gap).
    ],
  )
]

// ── The other axis: what's inside the generator ────────────────────────────
#content-slide(
  eyebrow-text: "Axis 2 — the generator",
  title: [How much of the generator is mechanism vs. learned?],
)[
  #v(8pt)
  #grid(
    columns: (1fr, 1fr, 1fr),
    column-gutter: 18pt,
    align: (center + top, center + top, center + top),
    outline-card(header: [Mechanistic], header-align: center, stroke-weight: "outer")[
      #v(3pt)
      *ODE*
      #v(3pt)
      #set text(size: 14pt)
      every term written by hand from pharmacology
    ],
    outline-card(header: [Hybrid], header-align: center, stroke-weight: "outer")[
      #v(3pt)
      *UDE*
      #v(3pt)
      #set text(size: 14pt)
      keep the structure you trust, let a neural net learn the rest
    ],
    outline-card(header: [Data-driven], header-align: center, stroke-weight: "outer")[
      #v(3pt)
      *Neural ODE*
      #v(3pt)
      #set text(size: 14pt)
      the whole vector field is a neural network
    ],
  )
  #v(10pt)
  #align(center, brand-callout[You already moved along this axis][
    This morning's UDE exercise *is* a step from mechanism toward data-driven —
    one move along this axis.
  ])
]

// ── THE MAP ────────────────────────────────────────────────────────────────
#content-slide(
  eyebrow-text: "The model design space",
  title: [Two axes, one map],
)[
  #v(2pt)
  #align(center, box(width: 100%, [
    #cetz.canvas(length: 1cm, {
      import cetz.draw: *

      // palette
      let c-mech = rgb("#250044")   // dp-purple-deep
      let c-hyb  = rgb("#6B00C7")   // dp-purple
      let c-data = rgb("#AE7FD6")   // dp-purple-mid
      let c-grid = rgb("#E3E3E3")
      let c-axis = rgb("#6C6874")

      // column x-centres (mechanistic / hybrid / data-driven)
      let xm = 2.4
      let xh = 6.2
      let xd = 10.0
      // row y-centres (none / conditional / amortised / posterior)
      let y1 = 0.9
      let y2 = 2.7
      let y3 = 4.5
      let y4 = 6.3

      // faint guide grid
      for yy in (y1, y2, y3, y4) {
        line((1.2, yy), (11.2, yy), stroke: (paint: c-grid, thickness: .6pt, dash: "dotted"))
      }

      // node helper
      let node(x, y, label, fill, ring: false) = {
        if ring {
          content((x, y), box(
            fill: fill, inset: (x: 9pt, y: 6pt), radius: 6pt,
            stroke: (paint: rgb("#F1B500"), thickness: 2.2pt),
            text(fill: white, weight: 700, size: 12pt, label),
          ))
        } else {
          content((x, y), box(
            fill: fill, inset: (x: 8pt, y: 5pt), radius: 5pt,
            text(fill: white, weight: 600, size: 11.5pt, label),
          ))
        }
      }

      // row 1 — no individual variation
      node(xm, y1, [ODE], c-mech)
      node(xh, y1, [UDE], c-hyb)
      node(xd, y1, [Neural ODE], c-data)

      // row 2 — conditional (covariate-driven)
      node(xm, y2, [cond. ODE], c-mech)
      node(xh, y2, [cond. UDE], c-hyb)

      // row 3 — amortised latents
      node(xh, y3, [Latent UDE], c-hyb)
      node(xd, y3, [Latent NODE / VAE], c-data)

      // row 4 — per-individual posterior (random effects)
      // NLME is the mechanistic endpoint; DeepNLME is a *dial* that spans
      // hybrid → fully neural (gradient band, our focus = gold outline).
      node(xm, y4, [NLME], c-mech)
      content(((xh + xd) / 2, y4), box(
        width: 6cm, inset: (x: 8pt, y: 6pt), radius: 6pt,
        fill: gradient.linear(c-hyb, c-data, angle: 0deg),
        stroke: (paint: rgb("#F1B500"), thickness: 2.4pt),
        align(center, text(fill: white, weight: 700, size: 12.5pt, [DeepNLME])),
      ))

      // axes
      line((1.0, 0.0), (11.4, 0.0), mark: (end: ">"), stroke: (paint: c-axis, thickness: 1.2pt))
      line((1.0, 0.0), (1.0, 7.0), mark: (end: ">"), stroke: (paint: c-axis, thickness: 1.2pt))

      // x-axis labels
      content((xm, -0.55), text(size: 11pt, fill: c-axis, [mechanistic]))
      content((xh, -0.55), text(size: 11pt, fill: c-axis, [hybrid]))
      content((xd, -0.55), text(size: 11pt, fill: c-axis, [data-driven]))
      content((6.2, -1.15), text(size: 12pt, weight: 700, fill: c-axis, [The generator]))

      // y-axis title (rotated)
      content((0.35, 0.0), angle: 90deg, anchor: "west",
        text(size: 12pt, weight: 700, fill: c-axis, [How variation is handled]))
    })
  ]))
  #v(2pt)
  #align(center, text(size: 13pt, fill: dp-gray-700)[
    Up = richer handling of *between-subject variability*: none → conditional → amortised latents → per-individual random effects.
  ])
]

// ── Reading the map: DeepNLME's neighbours ─────────────────────────────────
#content-slide(
  eyebrow-text: "Reading the map",
  title: [DeepNLME's neighbours are one move away],
)[
  #v(6pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 26pt,
    align: (left + horizon, left + top),
    [
      DeepNLME = *per-subject random effects* with a generator you can dial from
      *hybrid* all the way to *fully neural*. From there:
      #v(6pt)
      #numbered-list(
        [*Add full mechanism* (drop the neural net) → plain *NLME* (one move left).],
        [*Amortise the inference* → a *Latent UDE*, then a *Latent NODE / VAE* (move down).],
      )
      #v(4pt)
      "How much mechanism" is a *dial within DeepNLME*, not a jump to a different model.
    ],
    outline-card(header: [Why this is the sweet spot], stroke-weight: "outer")[
      #v(3pt)
      #set text(size: 15pt)
      DeepNLME keeps the *mechanism you trust* and the *per-subject rigour* of
      NLME, while borrowing the *flexible function approximation* of deep
      generative models.

      #v(6pt)
      It sits where pharmacology and machine learning overlap — deliberately, not
      by accident.
    ],
  )
]

// ── Latent NODE made explicit ──────────────────────────────────────────────
#content-slide(
  eyebrow-text: "The amortised cousin",
  title: [A Latent Neural ODE is a VAE over trajectories],
)[
  #v(6pt)
  #grid(
    columns: (1.05fr, 1fr),
    column-gutter: 26pt,
    align: (left + horizon, left + top),
    [
      #sub-eyebrow[The pipeline]
      #numbered-list(
        [*Encoder* reads a subject's time course → a latent $bold(z)_0$.],
        [A *neural ODE* evolves $bold(z)_0$ forward in time.],
        [*Decoder* maps the latent trajectory to observations.],
      )
      #v(6pt)
      Trained exactly like a VAE: an amortised encoder maximising the *ELBO* — a
      *lower bound* on the marginal likelihood.
    ],
    outline-card(header: [Same map, different coordinate], stroke-weight: "outer")[
      #v(3pt)
      #set text(size: 15pt)
      Latent NODE = *data-driven generator* + *amortised latents*.

      #v(6pt)
      Add mechanism and swap amortised inference for a per-subject posterior and
      you have walked back to *DeepNLME*.
    ],
  )
]

// ── Why this matters / the move we make next ───────────────────────────────
#content-slide(
  eyebrow-text: "So what",
  title: [Borrowing from ML = moving around this map],
)[
  #v(8pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 26pt,
    align: (left + horizon, left + horizon),
    [
      Once you see NLME as a latent-variable generative model, a whole toolbox
      opens up:
      #v(4pt)
      - amortise inference when you have many subjects
      - let neural nets fill in unknown mechanism
      - *couple* the random-effect latent space to *other* latent-variable models
    ],
    outline-card(header: [Next: coupling latent spaces], stroke-weight: "outer")[
      #v(3pt)
      #set text(size: 15pt)
      If a text or image model also has a latent space, and our NLME has one, we
      can *link them* — pour rich covariates into the random effects.

      #v(6pt)
      That is exactly the *embeddings* exercise, and the *ModelJoin* idea, coming
      up next.
    ],
  )
]

// ── Takeaways ──────────────────────────────────────────────────────────────
#content-slide(
  eyebrow-text: "Takeaways",
  title: [One space, many neighbours],
)[
  #v(10pt)
  #numbered-list(
    [An *NLME model is a conditional generative model* — random effects are its latent variables.],
    [*NLME and VAEs share the same target — the marginal likelihood*; they differ in how they approximate it (NLME the integral, the VAE a lower bound — the ELBO).],
    [*Two axes* organise the field: mechanism ↔ data-driven, and how individual variation is inferred.],
    [*DeepNLME* is a dial, not a point — per-subject random effects with a generator ranging from NLME (mechanistic) to fully neural, one move from the amortised Latent NODE / VAE.],
    [Moving around the map = *borrowing ML ideas for pharmacometrics*, which is what the rest of the day is about.],
  )
]

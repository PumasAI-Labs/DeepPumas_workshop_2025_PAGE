// ─────────────────────────────────────────────────────────────────────────
//  Day 2 — Tying it together (closing / summary)
//
//  Echoes the intro (a scientific model + neural networks), recaps the two
//  roles NNs play (discover dynamics; power the random effects — seen as
//  individualizable functions OR as a reshapeable BSV distribution), the
//  applications this opens up, and the future vision (VEM + reverse-mode for
//  scaling the fit; embeddings → large joint NLME over time). Ends on the
//  post-workshop survey QR for feedback.
//
//  Compile:  ./compile.sh 10_conclusion.typ
// ─────────────────────────────────────────────────────────────────────────

#import "deeppumas-slides/lib.typ": *

#show: deeppumas-theme.with(aspect-ratio: "16-9")

// ── Cover ───────────────────────────────────────────────────────────────
#title-slide(
  conference: "DeepPumas workshop · PAGE 2026 · wrap-up",
  cobrand: ("pumasai-primary-black", "deeppumas-primary-purple"),
  title:    [Tying it together],
  subtitle: [Scientific models + machine learning — and where it's heading],
  affiliations: ((label: "Pumas-AI", authors: [Niklas Korsbo]),),
)

// ── Echo the intro: one idea, two jobs for the NN ────────────────────────
#content-slide(
  eyebrow-text: "Where we've been",
  title: [Keep the science — let neural nets fill the gaps],
  center-body: true,
)[
  We never threw the mechanistic model away. We let *neural networks* do two
  jobs *inside* it:
  #v(12pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 26pt,
    align: (left + top, left + top),
    outline-card(header: [Discover the dynamics], stroke-weight: "outer")[
      #set text(size: 15pt)
      Where the mechanism is unknown, a neural net learns the missing term(s) —
      *UDEs* and *Neural ODEs*. A dial from hand-written mechanism to fully
      data-driven.
    ],
    outline-card(header: [Power the random effects], stroke-weight: "outer")[
      #set text(size: 15pt)
      The *individual* layer — how each subject differs from the population, and
      how the population itself varies.
    ],
  )
  #v(10pt)
  #align(center, text(size: 15pt, fill: dp-gray-700)[
    Almost everything else today was a consequence of these two.
  ])
]

// ── Random effects, two ways to see them ─────────────────────────────────
#content-slide(
  eyebrow-text: "Random effects, two ways to see them",
  title: [Individualizable functions — or a distribution you can reshape],
  center-body: true,
)[
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 26pt,
    align: (left + top, left + top),
    outline-card(header: [Individualizable functions], stroke-weight: "outer")[
      #set text(size: 15pt)
      A random effect turns the structural model into a *per-subject* function —
      individualization, without fitting a separate model for everyone.
    ],
    outline-card(header: [Transformations of the BSV distribution], stroke-weight: "outer")[
      #set text(size: 15pt)
      The random-effect prior *is* the between-subject-variability distribution —
      and we can *reshape* it: condition it on covariates (*augment*), *join*
      models with a learned joint prior, pour in *embeddings*.
    ],
  )
  #v(10pt)
  #align(center, brand-callout[Why it mattered][
    Seeing the random effects as a *latent distribution* — the generative view —
    is exactly what let us borrow ideas from machine learning.
  ])
]

// ── What it opens up ─────────────────────────────────────────────────────
#content-slide(
  eyebrow-text: "What it opens up",
  title: [A lot follows from those two moves],
  center-body: true,
)[
  #numbered-list(
    [*Discover* prognostic factors and covariate relationships *from data*, instead of pre-specifying them.],
    [Bring *complex data* — text, images — in as covariates through *embeddings*.],
    [*Join* independently-built endpoint models so information *flows across* them.],
    [*Individualized* predictions from sparse data — and richer, mechanism-respecting *virtual populations*.],
  )
]

// ── Where this is going ──────────────────────────────────────────────────
#content-slide(
  eyebrow-text: "Where this is going",
  title: [The next leaps],
  center-body: true,
)[
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 26pt,
    align: (left + top, left + top),
    outline-card(header: [Fit at a different scale], stroke-weight: "outer")[
      #set text(size: 15pt)
      The bottleneck is *fitting*. The biggest levers I see are *Variational EM
      (VEM)* and *reverse-mode* differentiation — they change how the fit
      *scales*, beyond what FOCE reaches, to *large joint and neural* models.
    ],
    outline-card(header: [Complex data → large joint NLME], stroke-weight: "outer")[
      #set text(size: 15pt)
      Turn images and text into embeddings, *track them over time*, and build
      *large joint NLME models* relating those trajectories to both the *drug
      (conditional)* and the *reportable clinical endpoint*.
    ],
  )
]

// ── Thanks + feedback QR ─────────────────────────────────────────────────
#content-slide(
  eyebrow-text: "Thank you",
  title: [Tell us how it went],
  center-body: true,
)[
  #grid(
    columns: (auto, 1fr),
    column-gutter: 40pt,
    align: (center + horizon, left + horizon),
    box(
      inset: 8pt, radius: 8pt, stroke: (paint: dp-purple-mid, thickness: 1pt),
      image("img/feedback_qr.png", width: 4.4cm),
    ),
    [
      #sub-eyebrow[Post-workshop survey]
      Your feedback shapes the next one — what landed, what dragged, what you
      want more of.
      #v(8pt)
      Scan the code, or grab us over coffee. Thank you for spending two days on
      this with us.
    ],
  )
]

// ─────────────────────────────────────────────────────────────────────────
//  Day 2 — Embedding models: turning complex data into covariates  (REFLOWED)
//
//  The GenAI↔NLME derivation that used to fill this deck (old slides 02–12)
//  now lives in deck 08 (the model design space). This deck keeps only the
//  embeddings story — and fixes the old "an embedding model is *just* the
//  encoder of a VAE" identity: an embedding model is any pretrained map
//  data→vector with meaningful geometry. A VAE encoder is one example, but
//  embeddings are broader and usually non-generative (no decoder).
//
//  Original 16-slide version: ./archive/07_embeddings_original/ (+ rendered
//  PDF) and ppt/07_embeddings.pptx.  Compile: ./compile.sh 07_embeddings.typ
// ─────────────────────────────────────────────────────────────────────────

#import "deeppumas-slides/lib.typ": *
#import "figures/embedder.typ": embedder-fig, similarity-fig, embedder-to-model-fig

#show: deeppumas-theme.with(aspect-ratio: "16-9")

// ── Cover ───────────────────────────────────────────────────────────────
#title-slide(
  conference: "DeepPumas workshop · PAGE 2026 · Day 2",
  cobrand: ("pumasai-primary-black", "deeppumas-primary-purple"),
  title:    [Embedding models],
  subtitle: [Turning text, images, and other complex data into covariates],
  affiliations: ((label: "Pumas-AI", authors: [Niklas Korsbo]),),
)

// ── What we actually want from complex data ──────────────────────────────
#content-slide(
  eyebrow-text: "What we actually want",
  title: [A vector that captures *meaning*, not raw bytes],
  center-body: true,
)[
  We want to feed rich data — a scan, a clinical note — into our models. The raw
  bytes are the wrong thing; we want a compact summary of *what the data means*:
  #v(12pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 26pt,
    align: (left + top, left + top),
    outline-card(header: [An image], stroke-weight: "outer")[
      #set text(size: 16pt)
      *not* the pixel intensities — but the objects present, their
      characteristics, what is happening, the style.
    ],
    outline-card(header: [A document], stroke-weight: "outer")[
      #set text(size: 16pt)
      *not* the individual words — but the topic, the information conveyed, the
      sentiment, the language.
    ],
  )
  #v(12pt)
  #align(center, text(size: 16pt, fill: dp-gray-700)[
    An *embedding model* is what turns the data into such a vector.
  ])
]

// ── What an embedding model is — and is not ──────────────────────────────
#content-slide(
  eyebrow-text: "Embedding models",
  title: [A learned map from data to a meaningful vector],
  center-body: true,
)[
  #align(center, embedder-fig())
  #v(14pt)
  An *embedding model* is any (usually *pretrained*) map from data → a
  fixed-length vector whose *geometry is meaningful*.
  #v(12pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 24pt,
    align: (left + top, left + top),
    outline-card(header: [A VAE encoder is one example], stroke-weight: "outer")[
      #set text(size: 15pt)
      The encoder from the generative story maps data → the *parameters* of its
      latent distribution; the posterior *mean* is an embedding.
    ],
    outline-card(header: [...but embeddings are broader], stroke-weight: "outer")[
      #set text(size: 15pt)
      Most aren't generative at all: *no decoder*, trained *directly* for useful
      geometry — e.g. *contrastive* objectives on similar/dissimilar pairs — not
      for reconstruction.
    ],
  )
]

// ── Show the geometry: similar inputs land close ─────────────────────────
#content-slide(
  eyebrow-text: "Meaningful geometry",
  title: [Similar inputs land close together],
  center-body: true,
)[
  #grid(
    columns: (1.4fr, 1fr),
    column-gutter: 30pt,
    align: (center + horizon, left + horizon),
    align(center, box(width: 100%, similarity-fig())),
    [
      Distance in the vector space *means* something: semantically similar inputs
      sit *near each other*, unrelated ones far apart.

      #v(8pt)
      *And that geometry is the training signal:* the model is shown many pairs
      labelled *similar* vs *different*, and learns to pull the similar together
      and push the rest apart — no decoder, no reconstruction.

      #v(8pt)
      #text(size: 12pt, fill: dp-purple)[
        #link("https://huggingface.co/spaces/mteb/leaderboard")[`huggingface.co/spaces/mteb/leaderboard`]
      ]
    ],
  )
]

// ── The unified API + bridge to the exercise ─────────────────────────────
#content-slide(
  eyebrow-text: "Why we care",
  title: [A unified "API": any modality → a vector covariate],
  center-body: true,
)[
  #align(center, embedder-to-model-fig())
  #v(16pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 24pt,
    align: (left + horizon, left + top),
    [
      #sub-eyebrow[The small-N win]
      The model was *pretrained on far more data than our trial* — so we get
      strong features *without* training a big model on our own small dataset.

      #v(6pt)
      And heterogeneous data — images, notes, signals — all become *simple numeric
      vectors*: one interface for all of it.
    ],
    outline-card(header: [Next: use it as a covariate], stroke-weight: "outer")[
      #set text(size: 16pt)
      Feed the embedding into the model — e.g. to *condition the random-effect
      prior* (augment). That is the *embeddings exercise*.
    ],
  )
]

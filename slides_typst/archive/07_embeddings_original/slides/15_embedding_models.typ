#import "../../deeppumas-slides/lib.typ": *
#import "../../figures/encoder_decoder.typ": encoder-only-fig

#content-slide(
  eyebrow-text: "Embedding models",
  title: [Embedding models --- the encoder, alone],
)[
  #v(4pt)
  #align(center, encoder-only-fig())
  #v(12pt)
  #outline-card(stroke-weight: "outer", body-size: 14pt)[
    Extract meaningful, information-dense #strong[features] from the
    data --- without decoding back to the original modality.
  ]
  #v(8pt)
  #align(center, text(size: 12pt, fill: dp-purple)[
    #link("https://huggingface.co/spaces/mteb/leaderboard")[
      `huggingface.co/spaces/mteb/leaderboard`
    ]
  ])
]

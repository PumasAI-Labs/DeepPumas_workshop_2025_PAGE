#import "../../deeppumas-slides/lib.typ": *
#import "../../figures/encoder_decoder.typ": encoder-only-fig

#content-slide(
  eyebrow-text: "Payoff",
  title: [A unified "API" for complex data],
)[
  #v(4pt)
  #align(center, encoder-only-fig(input-label: "Image · Text · ..."))
  #v(14pt)
  #align(center, outline-card(stroke-weight: "outer", body-size: 16pt, width: auto)[
    Different data modalities (images, text, ...) all map to embeddings
    --- simple vectors of numbers.
  ])
]

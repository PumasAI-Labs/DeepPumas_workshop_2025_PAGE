#import "../../deeppumas-slides/lib.typ": *
#import "../../figures/encoder_decoder.typ": encoder-decoder-fig

#content-slide(
  eyebrow-text: "Anatomy",
  title: [Generative AI --- typical anatomy],
)[
  #v(1fr)
  #align(center, encoder-decoder-fig())
  #v(1fr)
  #align(center, text(size: 14pt, fill: dp-gray-700)[
    Input #sym.arrow.r encoder #sym.arrow.r latent #sym.arrow.r decoder #sym.arrow.r output.
  ])
]

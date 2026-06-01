#import "../../deeppumas-slides/lib.typ": *
#import "../../figures/encoder_decoder.typ": encoder-decoder-fig

#content-slide(
  eyebrow-text: "Mapping",
  title: [NLME as GenAI],
)[
  #v(4pt)
  #align(center, encoder-decoder-fig(input-label: "Time series", output-label: "Time series"))
  #v(12pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 22pt,
    align: (left + top, left + top),

    [
      #sub-eyebrow[Input / Output]
      Time series of outcomes / concentrations.

      #v(6pt)
      #sub-eyebrow[Decoder]
      The structural NLME --- parameter transforms, dynamics, etc.
    ],

    [
      #sub-eyebrow[Encoder]
      #v(4pt)
      #text(size: 22pt, weight: 600, fill: dp-purple)[???]
      #v(8pt)
      #text(size: 13pt, fill: dp-gray-700)[
        We don't have one. We use marginal-likelihood fitting to back out
        $bold(eta)$ instead.
      ]
    ],
  )
]

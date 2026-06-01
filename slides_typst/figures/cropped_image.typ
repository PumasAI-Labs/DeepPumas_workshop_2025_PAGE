// Wrap a typst image so its top/bottom strips are clipped off.
// `top-pct`/`bottom-pct` are fractions of the image height. The image
// is sized to fill the available container height (use inside an
// `align(center + horizon, ...)` or a sized container).
//
// Approach: place a larger image inside a clipping box. The image is
// scaled so that its CROPPED region matches the target height.
//
// Usage:
//   #cropped("../../assets/foo.png", aspect: 2400 / 1350, bottom-pct: 6%)

#let cropped(path, aspect: 16 / 9, top-pct: 0%, bottom-pct: 0%,
             height: 100%) = {
  let keep = 100% - top-pct - bottom-pct
  let img-scale = 100% / (keep / 100%)
  // The displayed (cropped) image will have aspect = original-aspect * keep
  // We want height = the given height, so:
  //   displayed height = height
  //   full image height = height / (keep%)
  //   full image width = full height * aspect
  // Clip box: width = displayed width, height = displayed height.
  // Place image with dy offset = -top-pct of full image height.
  // We use percentage offset on the inner image via box layout.
  box(height: height, clip: true)[
    #place(top + left,
           dy: -top-pct * img-scale * (1 / img-scale),
           image(path, height: img-scale * height))
  ]
}

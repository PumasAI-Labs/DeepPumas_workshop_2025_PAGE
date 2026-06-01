#!/bin/sh
set -e
exec typst compile --font-path deeppumas-slides/assets/fonts/ "${1:-08_model_space.typ}" "${@:2}"

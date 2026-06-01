# Day-2 typst slides — working notes

## `08_genai_model_space.typ` — GenAI ↔ NLME and the model design space
Status: **drafted, compiles, conceptually vetted.** Built on the `08-JSM`
`deeppumas-slides` theme (copied locally). Compile: `./compile.sh 08_genai_model_space.typ`.

Spine: NLME is a conditional generative model → NLME & VAEs maximise the same
marginal likelihood → the integral is intractable → two axes (mechanism↔data-driven
× how individual variation is handled) → the 2-D map → DeepNLME as a dial →
Latent NODE → coupling latent spaces (bridge to embeddings) → takeaways.

### "Two ways to pin down the latent variable" slide — LOCKED (do not re-open)
Hard-won, vetted by advisor. The invariant, at altitude:
- **NLME:** a posterior **represented per subject**, by Bayes' rule
  `p(ηᵢ|yᵢ,xᵢ) ∝ p(yᵢ|ηᵢ,xᵢ)·p(ηᵢ)`. Method-agnostic (FOCE/Laplace/EM-SAEM/MCMC).
- **Amortised:** a shared encoder outputs the **parameters** of each posterior,
  `q(ηᵢ ; λ_φ(yᵢ,xᵢ))`. Family-agnostic (Gaussian/flow/mixture).
- Pitfalls already fixed (don't reintroduce): EBE/mode ≠ posterior; not
  Laplace-only; not Gaussian-only; the *parameters* (not the posterior) are the
  deterministic function of data; "no per-subject **fit**" (running the encoder
  is still inference). **Rule: name the structure, never a specific method/family.**

## `09_model_joining.typ` — Joining models (NF joint priors)
Status: **drafted, compiles (12 pp), figures from the poster.** Compile:
`./compile.sh 09_model_joining.typ`. Frames the joining *exercise* (below).

Spine: the need (biomarkers→endpoint) → joint fit is a wall → fit independently,
pay a misspecified (independent) prior → assumption ladder, relax only A4 → the
correlational bridge (`mj_figures/fig-joint-schematic`) → re-fit a joint prior on
the per-patient posteriors via `replace_randeffs_dist` (Gaussian or NF) → recipe →
Exp1 table (post-hoc Gaussian = 98% of joint, no joint fit) → Exp2 yin-yang (NF
earns its keep) → Exp3 sparse-PE info transfer (R² 0.63→0.72, oracle 0.74) →
workflow → "your turn" exercise lead-in. Source: `ModelJoining_PAGE2026/poster`.

## Model-joining EXERCISE — handed to JuliaHub Claude
Brief: `context/modeljoin_exercise_brief.md`. It runs/tests in **JuliaHub** (not
locally). Core = minimal 2-endpoint, correlated REs, fit independently, EBE
scatter shows the correlation, join with `replace_randeffs_dist`, correlation
pops into the prior + cross-prediction. Base on `ModelJoining_PAGE2026`
`analyses/03_2d_toy/yinyang.jl` (swap yin-yang for a correlated Gaussian DGM).

## `07_embeddings.typ` — Embedding models (REFLOWED, done 2026-06-02)
Status: **reflowed to 4 slides, compiles, in main `slides_typst/`.** Original
16-slide deck backed up in `archive/07_embeddings_original/` (+ rendered PDF) and
`ppt/07_embeddings.pptx`. The GenAI↔NLME derivation is gone (deck 08 owns it).
Slides: meaning-not-bytes → **embedding model is a pretrained data→vector map,
NOT a VAE encoder** (no decoder/reconstruction; trained for similarity) via new
`figures/embedder.typ` → unified API → bridge to exercise.

Deck 08 gained a visual **"Match the data's distribution — in any space"** slide
(observed/generated contour + AI faces, from ViralDynamics; in `slides_typst/img/`)
— the useful GenAI visuals Niklas wanted to keep. Deck 08 now 13 content slides.

## Still pending
1. **Full consolidation** of the *other* converted decks (01–06, conclusion,
   07_tgd_os) from worktree `.claude/worktrees/typst-pilot/` into `slides_typst/`.
   Watch the path nesting: worktree per-deck slides import `../../deeppumas-slides`
   & `../../figures`; flatten cleanly when bringing over.
2. **Encoder/latent/decoder schematic** (Niklas's idea) for the `08` inference
   slide: decoder = structural model `η→y`; NLME encoder = the inverse solved per
   patient; amortised = a separate model making a forward attempt at that inverse.
   Reuse `figures/encoder_decoder.typ`.

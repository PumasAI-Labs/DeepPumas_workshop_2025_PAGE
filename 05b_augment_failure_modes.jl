using Random
using Distributions
using DeepPumas
using CairoMakie
using AlgebraOfGraphics
using DataFramesMeta

set_mlp_backend(:staticflux)
set_theme!(deep_light(); backgroundcolor=:white)

#=
# Where `augment` breaks — and how joint estimation fixes it

`augment` is a pragmatic shortcut: fit an NLME, take EBE point estimates,
regress them against covariates, redefine η as η + g(c).  It worked well
on the cases we just used it on, but it's a simplification — there are
specific patterns where the shortcut bites.  Two worth knowing about:

  1. Sharp covariate cleaving of the population.  The conditional
     residual on η is one-sided (half-Gaussian), but augment models it
     with a full Gaussian and leaks density to the wrong side.

  2. Mixed information across subjects.  Point-estimate EBEs are
     differentially shrunk toward the prior mean, and augment's
     EBE-regression can't tell signal from shrinkage.  The fitted g(c)
     underestimates the covariate effect — sometimes severely.

Both are *structural*, not tuning issues.  Joint estimation with the
split-NN architecture from 04b avoids them by construction (Section 3).
The reason we still keep augment in the toolbox: cost and workflow
separability — discussed at the bottom.
=#


############################################################################################
## Failure mode 1 — Half-Gaussian residual ≠ Gaussian
############################################################################################

# Setup: bimodal truth, NN pushforward base fit, augment on top.
# Same pattern as 03b Stage 1.  Rebuilt here so this file stands alone.

Random.seed!(42)
nsubj_fm1 = 400

truth_fm1 = @model begin
  @param begin
    σ ∈ RealDomain(; lower=0., init=0.1)
    ω ∈ RealDomain(; lower=0., init=0.2)
  end
  @random η_true ~ Normal(0, 1)
  @covariates μ_center
  @pre μ = μ_center + ω * η_true
  @derived y ~ @. Normal(μ, σ)
end

subjects_fm1 = [Subject(; id=i, covariates=(; μ_center=rand((-1.0, 1.0)))) for i in 1:nsubj_fm1]
p_truth_fm1 = (; σ=0.1, ω=0.2)
sims_fm1 = simobs(truth_fm1, subjects_fm1, p_truth_fm1; obstimes=0.0:1.0:3.0)
pop_fm1 = Subject.(sims_fm1)

# NN-pushforward base model (no covariates)
model_fm1 = @model begin
  @param begin
    NN ∈ MLPDomain(1, 8, 8, (1, identity); reg=L2(1e-2))
    σ ∈ RealDomain(; lower=0., init=0.2)
  end
  @random η ~ Normal(0, 1)
  @pre μ = NN(η)[1]
  @derived y ~ @. Normal(μ, σ)
end

fpm_fm1 = fit(model_fm1, pop_fm1, init_params(model_fm1), MAP(FOCE());
              optim_options=(; iterations=300))

# Augment with μ_center
target_fm1 = preprocess(fpm_fm1)
nn_aug_fm1 = MLPDomain(numinputs(target_fm1), 6, 6, (numoutputs(target_fm1), identity); reg=L2(1.0))
fnn_aug_fm1 = fit(nn_aug_fm1, target_fm1; training_fraction=0.9, optim_options=(; loss=l2))
augmented_fpm_fm1 = augment(fpm_fm1, fnn_aug_fm1)

# Mean shift learned for each covariate value
g_fm1     = coef(augmented_fpm_fm1).nn
shift_m1  = g_fm1(Subject(; id=1, covariates=(; μ_center=-1.0))).η
shift_p1  = g_fm1(Subject(; id=2, covariates=(; μ_center=+1.0))).η
@show shift_m1 shift_p1

# y-space: conditional marginal of NN(η + g(c)) vs truth
let
    nn_orig = coef(augmented_fpm_fm1).NN
    nsamp = 20_000
    ηs = randn(nsamp)

    μ_aug_m1   = first.(nn_orig.(ηs .+ shift_m1))
    μ_aug_p1   = first.(nn_orig.(ηs .+ shift_p1))
    μ_truth_m1 = -1 .+ p_truth_fm1.ω .* randn(nsamp)
    μ_truth_p1 = +1 .+ p_truth_fm1.ω .* randn(nsamp)

    # Fraction of conditional mass on the wrong (dead) mode
    @show mean(μ_aug_m1 .> 0) mean(μ_aug_p1 .< 0)

    df = DataFrame(
        μ      = vcat(μ_aug_m1, μ_aug_p1, μ_truth_m1, μ_truth_p1),
        c      = vcat(fill("c = -1", nsamp), fill("c = +1", nsamp),
                      fill("c = -1", nsamp), fill("c = +1", nsamp)),
        source = vcat(fill("augment: NN(η + g(c))", 2*nsamp),
                      fill("truth conditional",     2*nsamp)),
    )
    data(df) * mapping(:μ; color=:source, row=:c) *
      AlgebraOfGraphics.density() |> draw
end

# η-space: see the half-Gaussian truth-residual vs augment's full Gaussian
let
    df_ebe  = unique(DataFrame(predict(fpm_fm1)), :id)
    ebes_m1 = collect(skipmissing(df_ebe[df_ebe.μ_center .< 0, :η]))
    ebes_p1 = collect(skipmissing(df_ebe[df_ebe.μ_center .> 0, :η]))

    nsamp     = 5000
    η⁺_aug_m1 = randn(nsamp) .+ shift_m1
    η⁺_aug_p1 = randn(nsamp) .+ shift_p1

    df_η = DataFrame(
        η      = vcat(η⁺_aug_m1, η⁺_aug_p1, ebes_m1, ebes_p1),
        c      = vcat(fill("c = -1", nsamp),         fill("c = +1", nsamp),
                      fill("c = -1", length(ebes_m1)), fill("c = +1", length(ebes_p1))),
        source = vcat(fill("augment η⁺ ~ N(g(c), 1)", 2*nsamp),
                      fill("EBE η̂ (truth residual)", length(ebes_m1) + length(ebes_p1))),
    )
    data(df_η) * mapping(:η; color=:source, row=:c) *
      AlgebraOfGraphics.density() |> draw
end

#=
The base fit sorted subjects to one side of NN's near-step at η = 0, so
the conditional EBE distribution is a half-Gaussian.  Augment models
that one-sided residual with a full N(g(c), 1).  A Gaussian cannot fit a
half-Gaussian — the symmetric tail overshoots back across η = 0, and
that overshoot is the mass pushed through NN to the dead mode.

This bites whenever a covariate cleanly cleaves the population: a
post-hoc Gaussian residual approximation breaks structurally.
=#


############################################################################################
## Failure mode 2 — EBE shrinkage and the regression-to-no-signal trap
############################################################################################

# Augment regresses point-estimate EBEs against covariates.  EBEs are
# Bayesian posterior modes, shrunk toward the prior mean by an amount
# that depends on how informative each subject's data is.  Subjects with
# many observations shrink very little; subjects with few observations
# shrink a lot.  The regression treats both as equally informative.
#
# Setup: same bimodal truth, but half the population has 10 obs and the
# other half has 1.  σ is pushed up so the sparse-subject EBEs visibly
# shrink toward zero.

Random.seed!(43)
nrich, nsparse = 200, 200

truth_fm2 = @model begin
  @param begin
    σ ∈ RealDomain(; lower=0., init=1.0)
    ω ∈ RealDomain(; lower=0., init=0.15)
  end
  @random η_true ~ Normal(0, 1)
  @covariates μ_center
  @pre μ = μ_center + ω * η_true
  @derived y ~ @. Normal(μ, σ)
end
p_truth_fm2 = (; σ=1.0, ω=0.15)

subj_rich   = [Subject(; id="R$i", covariates=(; μ_center=rand((-1.0, 1.0)))) for i in 1:nrich]
subj_sparse = [Subject(; id="S$i", covariates=(; μ_center=rand((-1.0, 1.0)))) for i in 1:nsparse]
sims_rich   = simobs(truth_fm2, subj_rich,   p_truth_fm2; obstimes=0.0:0.5:4.5)  # 10 obs
sims_sparse = simobs(truth_fm2, subj_sparse, p_truth_fm2; obstimes=[1.5])         # 1 obs

pop_rich = Subject.(sims_rich)
pop_mix  = vcat(pop_rich, Subject.(sims_sparse))

# Classical fit: η absorbs the mode shift, ω_fit ≈ 1.
model_fm2 = @model begin
  @param begin
    tvμ ∈ RealDomain(; init=0.0)
    ω   ∈ RealDomain(; lower=0., init=1.0)
    σ   ∈ RealDomain(; lower=0., init=0.5)
  end
  @random η ~ Normal(0, ω)
  @pre μ = tvμ + η
  @derived y ~ @. Normal(μ, σ)
end

fpm_mix  = fit(model_fm2, pop_mix,  init_params(model_fm2), MAP(FOCE());
               optim_options=(; iterations=200))
fpm_rich = fit(model_fm2, pop_rich, init_params(model_fm2), MAP(FOCE());
               optim_options=(; iterations=200))

# EBE shrinkage by group: rich subjects retain most of ±1; sparse subjects
# get pulled toward 0.
df_ebe2 = unique(DataFrame(predict(fpm_mix)), :id)
df_ebe2.group = ifelse.(startswith.(df_ebe2.id, "R"), "rich (10 obs)", "sparse (1 obs)")

data(df_ebe2) *
  mapping(:η; color=:group, row=:μ_center => nonnumeric) *
  AlgebraOfGraphics.density() |> draw

# Augment on mix vs rich-only.  Same data-generating truth in both cases;
# only the included subjects differ.
target_mix  = preprocess(fpm_mix)
target_rich = preprocess(fpm_rich)
nn_aug_fm2  = MLPDomain(numinputs(target_mix), 6, 6, (numoutputs(target_mix), identity); reg=L2(1.0))

fnn_mix  = fit(nn_aug_fm2, target_mix;  training_fraction=0.9, optim_options=(; loss=l2))
fnn_rich = fit(nn_aug_fm2, target_rich; training_fraction=0.9, optim_options=(; loss=l2))
aug_mix  = augment(fpm_mix,  fnn_mix)
aug_rich = augment(fpm_rich, fnn_rich)

g_mix  = coef(aug_mix).nn
g_rich = coef(aug_rich).nn

# Visualize the headline: same EBE cloud (from the mixed-population fit),
# colored by each subject's data richness; on top, the g(c) line that
# augment learned on rich-only vs on the full mix.  The story is in the
# difference between the lines — augment-mix is visibly flatter, pulled
# toward zero by the shrunken sparse-subject EBEs.
begin
  c_grid = collect(-1.5:0.05:1.5)
  g_mix_grid  = [g_mix(Subject(;  id="m$i", covariates=(; μ_center=c))).η for (i,c) in enumerate(c_grid)]
  g_rich_grid = [g_rich(Subject(; id="r$i", covariates=(; μ_center=c))).η for (i,c) in enumerate(c_grid)]

  df_ebe_fm2 = unique(DataFrame(predict(fpm_mix)), :id)
  df_ebe_fm2.group = ifelse.(startswith.(df_ebe_fm2.id, "R"),
                              "Rich subjects (10 obs each)",
                              "Sparse subjects (1 obs each)")
  df_ebe_fm2 = dropmissing(df_ebe_fm2, :η)
  df_ebe_fm2.c_jit = df_ebe_fm2.μ_center .+ 0.05 .* randn(nrow(df_ebe_fm2))

  df_rich_fm2   = filter(:group => ==("Rich subjects (10 obs each)"),   df_ebe_fm2)
  df_sparse_fm2 = filter(:group => ==("Sparse subjects (1 obs each)"), df_ebe_fm2)

  fig_fm2 = Figure(; size=(820, 500))
  ax_fm2  = Axis(fig_fm2[1, 1]; xlabel="μ_center (covariate)", ylabel="η",
                 title="Augment learns a flatter g(c) when sparse subjects join the training set")
  scatter!(ax_fm2, df_rich_fm2.c_jit,   df_rich_fm2.η;   color=:steelblue, alpha=0.4, markersize=7, label="EBE (rich subjects, 10 obs)")
  scatter!(ax_fm2, df_sparse_fm2.c_jit, df_sparse_fm2.η; color=:firebrick, alpha=0.4, markersize=7, label="EBE (sparse subjects, 1 obs)")
  lines!(ax_fm2, c_grid, c_grid;      color=:black,     linestyle=:dash, linewidth=3, label="truth (η = c)")
  lines!(ax_fm2, c_grid, g_rich_grid; color=:steelblue, linewidth=4,                  label="augment trained on rich subjects only")
  lines!(ax_fm2, c_grid, g_mix_grid;  color=:firebrick, linewidth=4,                  label="augment trained on rich + sparse")
  ylims!(ax_fm2, -2.5, 2.5)
  axislegend(ax_fm2; position=:lt, labelsize=10)
  fig_fm2
end

#=
Truth says ±1.  Augment on rich-only lands close (~ ±0.85; some EBE
shrinkage even here).  Augment on the mixed population pulls noticeably
toward zero (~ ±0.7) — the sparse subjects' EBEs are shrunk, the
regression takes them at face value, the learned covariate effect is
biased.

The surprising part — and this is the pedagogical headline — is that
**adding more data made the fit worse**.  Augment-on-mix saw twice as
many subjects as augment-on-rich-only, but learned a flatter relationship.
This is the opposite of the standard NLME instinct ("use every subject
you have, the model handles uncertainty").

The reason is the EBE bottleneck.  A classical NLME fit propagates each
subject's likelihood directly; sparse subjects contribute proportionally
to the information they carry.  Augment, by contrast, regresses on
point-estimate EBEs — it has no view into per-subject certainty, so
shrunken EBEs vote with equal weight against well-determined ones.

Practical implication: when using augment, pre-filter on EBE quality
(min observation count, EBE standard error, …) before the regression
step.  That's not something you'd ever do in a normal NLME workflow,
where every subject contributes.  But it's the right move once you've
chosen the augment shortcut — or use joint estimation, where the
filtering is handled by the likelihood for free.
=#


############################################################################################
## Resolution — joint estimation with split-NN
############################################################################################

# Both failure modes share the same root: augment is *post-hoc*.  The base
# NLME is fit first (with Gaussian η, no covariates), point-estimate EBEs
# get extracted, then a separate ML model regresses them against c.  Each
# step loses information the next can't recover.
#
# The split-NN architecture from 04b puts both pieces inside one Pumas
# model: NN_η(η) absorbs residual structure, NN_cov(c) carries the
# covariate effect, and joint fitting optimizes both against the raw
# likelihood — no EBE round-trip.  Same data → both failure modes go away.

# --- FM1 redo with split-NN ----------------------------------------------------
model_fm1_split = @model begin
  @param begin
    NN_η   ∈ MLPDomain(1, 6, 6, (1, identity, false); reg=L2(1e-1))
    NN_cov ∈ MLPDomain(1, 4,    (1, identity, false); reg=L2(1e-1))
    σ ∈ RealDomain(; lower=0., init=0.2)
  end
  @random η ~ Normal(0, 1)
  @covariates μ_center
  @pre μ = NN_η(η)[1] + NN_cov(μ_center)[1]
  @derived y ~ @. Normal(μ, σ)
end

fpm_fm1_split = fit(model_fm1_split, pop_fm1, init_params(model_fm1_split), MAP(FOCE());
                    optim_options=(; iterations=300))

# Conditional marginal at c = ±1, same comparison as before.  The split-NN
# absorbed the mode shift through NN_cov — NN_η stays near-linear, so the
# residual is Gaussian within each c and the dead mode vanishes.
let
    nn_η_fit   = coef(fpm_fm1_split).NN_η
    nn_cov_fit = coef(fpm_fm1_split).NN_cov
    nsamp = 20_000
    ηs = randn(nsamp)

    μ_split_m1 = first.(nn_η_fit.(ηs)) .+ nn_cov_fit(-1.0)[1]
    μ_split_p1 = first.(nn_η_fit.(ηs)) .+ nn_cov_fit(+1.0)[1]
    μ_truth_m1 = -1 .+ p_truth_fm1.ω .* randn(nsamp)
    μ_truth_p1 = +1 .+ p_truth_fm1.ω .* randn(nsamp)

    @show mean(μ_split_m1 .> 0) mean(μ_split_p1 .< 0)   # both should be ~0

    df = DataFrame(
        μ      = vcat(μ_split_m1, μ_split_p1, μ_truth_m1, μ_truth_p1),
        c      = vcat(fill("c = -1", nsamp), fill("c = +1", nsamp),
                      fill("c = -1", nsamp), fill("c = +1", nsamp)),
        source = vcat(fill("split-NN: NN_η(η) + NN_cov(c)", 2*nsamp),
                      fill("truth conditional",             2*nsamp)),
    )
    data(df) * mapping(:μ; color=:source, row=:c) *
      AlgebraOfGraphics.density() |> draw
end

# --- FM2 redo with split-NN ---------------------------------------------------
model_fm2_split = @model begin
  @param begin
    NN_η   ∈ MLPDomain(1, 6, 6, (1, identity, false); reg=L2(1e-1))
    NN_cov ∈ MLPDomain(1, 4,    (1, identity, false); reg=L2(1e-1))
    σ ∈ RealDomain(; lower=0., init=1.0)
  end
  @random η ~ Normal(0, 1)
  @covariates μ_center
  @pre μ = NN_η(η)[1] + NN_cov(μ_center)[1]
  @derived y ~ @. Normal(μ, σ)
end

# Fit split-NN on both populations.  The key contrast: augment degraded
# when sparse subjects were added; split-NN should be at least as good on
# the mix as on rich-only, recovering the NLME instinct that more data →
# better fit.
fpm_fm2_split_rich = fit(model_fm2_split, pop_rich, init_params(model_fm2_split), MAP(FOCE());
                          optim_options=(; iterations=300))
fpm_fm2_split_mix  = fit(model_fm2_split, pop_mix,  init_params(model_fm2_split), MAP(FOCE());
                          optim_options=(; iterations=300))

# NN_cov is the analogue of g(c) — evaluate it on the same grid.
g_split_rich_grid = [coef(fpm_fm2_split_rich).NN_cov(c)[1] for c in c_grid]
g_split_mix_grid  = [coef(fpm_fm2_split_mix ).NN_cov(c)[1] for c in c_grid]

# Side-by-side comparison: same EBE backdrop, same "rich vs mix" lines,
# two methods.  In the augment panel the two lines diverge — more data
# hurt.  In the split-NN panel they overlap — the joint likelihood
# weights each subject by its information content automatically, so
# adding sparse subjects does no damage.
begin
  fig_fm2_resolved = Figure(; size=(1100, 500))
  for (i, (title_str, line_rich, line_mix)) in enumerate([
      ("Augment (post-hoc regression on EBEs)", g_rich_grid,       g_mix_grid),
      ("Split-NN (joint estimation)",          g_split_rich_grid, g_split_mix_grid),
  ])
      ax = Axis(fig_fm2_resolved[1, i];
                xlabel="μ_center (covariate)", ylabel=i==1 ? "η" : "",
                title=title_str)
      scatter!(ax, df_rich_fm2.c_jit,   df_rich_fm2.η;   color=:steelblue, alpha=0.25, markersize=6, label="EBE — rich (10 obs)")
      scatter!(ax, df_sparse_fm2.c_jit, df_sparse_fm2.η; color=:firebrick, alpha=0.25, markersize=6, label="EBE — sparse (1 obs)")
      lines!(ax, c_grid, c_grid;    color=:black,     linestyle=:dash, linewidth=3, label="truth (η = c)")
      lines!(ax, c_grid, line_rich; color=:steelblue, linewidth=4,                  label="fit on rich subjects only")
      lines!(ax, c_grid, line_mix;  color=:firebrick, linewidth=4,                  label="fit on rich + sparse")
      ylims!(ax, -2.5, 2.5)
      i == 2 && axislegend(ax; position=:lt, labelsize=10)
  end
  fig_fm2_resolved
end

#=
The augment panel shows the FM2 failure: the rich-only line tracks
truth, the mix line is pulled flatter.  The split-NN panel shows the
fix: both lines overlap on truth, regardless of whether sparse subjects
were included.  Joint estimation restores the NLME instinct — more data
is better, the likelihood weights each subject by information content
for free.  For FM1, the same architecture absorbs the mode shift
through NN_cov(c) so NN_η stays near-linear; the conditional marginal
becomes unimodal and the dead mode goes to zero.

So both failure modes resolve under joint estimation.  The tradeoffs
that keep augment in the toolbox anyway:

  • Cost.  For large models — DeepNLME with ODE solves per subject and
    a serious covariate NN — joint fitting can be much slower than
    fit-then-augment.  Augment lets you reuse a base NLME and bolt
    covariates on cheaply.

  • Separable workflow.  Augment is a clean pipeline: base model,
    derived ML target, ML fit, augmented model.  Each stage is
    inspectable and reproducible in isolation; team members can own
    different pieces.  Joint fits collapse that into one optimization
    problem with all of its convergence behaviour entangled.

Use augment when these properties matter and your data doesn't hit
either failure mode.  Reach for joint fitting (DeepNLME, split-NN)
when the covariate cleanly cleaves the population or when subject-level
information is uneven enough that EBE shrinkage is doing real damage.
=#

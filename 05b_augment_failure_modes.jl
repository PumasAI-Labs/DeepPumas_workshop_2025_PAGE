using Random
using Distributions
using DeepPumas
using CairoMakie
using AlgebraOfGraphics
using DataFramesMeta

set_mlp_backend(:staticflux)
set_theme!(deep_light(); backgroundcolor=:white)

#=
# Two failure modes of `augment`, and a joint-estimation fix

Both real, neither catastrophic: augment still beats the no-covariate
baseline in both cases.  These are the predictable consequences of the
simplifications that make augment quick and easy — worth seeing once.
=#


############################################################################################
## FM1 — Sharp covariate splits make augment leak into the "dead mode"
##
## Truth gives one-sided EBE residuals (half-Gaussian).  Augment uses a full Gaussian.
## The symmetric tail leaks density to the side of the NN-step where truth has none.
############################################################################################

# Rebuilt 03b Stage 1 so this file stands alone.
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

target_fm1 = preprocess(fpm_fm1)
nn_aug_fm1 = MLPDomain(numinputs(target_fm1), 6, 6, (numoutputs(target_fm1), identity); reg=L2(1.0))
fnn_aug_fm1 = fit(nn_aug_fm1, target_fm1; training_fraction=0.9, optim_options=(; loss=l2))
augmented_fpm_fm1 = augment(fpm_fm1, fnn_aug_fm1)

g_fm1     = coef(augmented_fpm_fm1).nn
shift_m1  = g_fm1(Subject(; id=1, covariates=(; μ_center=-1.0))).η
shift_p1  = g_fm1(Subject(; id=2, covariates=(; μ_center=+1.0))).η
@show shift_m1 shift_p1

# y-space: conditional marginal vs truth
let
    nn_orig = coef(augmented_fpm_fm1).NN
    nsamp = 20_000
    ηs = randn(nsamp)

    μ_aug_m1   = first.(nn_orig.(ηs .+ shift_m1))
    μ_aug_p1   = first.(nn_orig.(ηs .+ shift_p1))
    μ_truth_m1 = -1 .+ p_truth_fm1.ω .* randn(nsamp)
    μ_truth_p1 = +1 .+ p_truth_fm1.ω .* randn(nsamp)

    @show mean(μ_aug_m1 .> 0) mean(μ_aug_p1 .< 0)  # dead-mode mass

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

# η-space: half-Gaussian truth residual vs augment's full Gaussian
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
EBE residual is half-Gaussian (one side of NN's step at η = 0); augment
fits it with a full Gaussian.  The symmetric tail is the dead-mode mass.
=#


############################################################################################
## FM2 — Adding sparse subjects makes augment WORSE (opposite the NLME instinct)
##
## Augment regresses on point-estimate EBEs.  Sparse subjects' EBEs are shrunk toward zero
## and vote with equal weight against well-determined ones, biasing g(c) toward "no effect."
############################################################################################

# Same bimodal truth; half the population gets 10 obs, half gets 1.
# σ pushed up so sparse-subject EBEs visibly shrink toward zero.
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

# EBE shrinkage by group
df_ebe2 = unique(DataFrame(predict(fpm_mix)), :id)
df_ebe2.group = ifelse.(startswith.(df_ebe2.id, "R"), "rich (10 obs)", "sparse (1 obs)")

data(df_ebe2) *
  mapping(:η; color=:group, row=:μ_center => nonnumeric) *
  AlgebraOfGraphics.density() |> draw

# Augment on mix vs rich-only
target_mix  = preprocess(fpm_mix)
target_rich = preprocess(fpm_rich)
nn_aug_fm2  = MLPDomain(numinputs(target_mix), 6, 6, (numoutputs(target_mix), identity); reg=L2(1.0))

fnn_mix  = fit(nn_aug_fm2, target_mix;  training_fraction=0.9, optim_options=(; loss=l2))
fnn_rich = fit(nn_aug_fm2, target_rich; training_fraction=0.9, optim_options=(; loss=l2))
aug_mix  = augment(fpm_mix,  fnn_mix)
aug_rich = augment(fpm_rich, fnn_rich)

g_mix  = coef(aug_mix).nn
g_rich = coef(aug_rich).nn

# EBEs (colored by data richness) + g(c) learned on rich-only vs full mix.
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
Adding the sparse subjects made augment worse — opposite the NLME
instinct.  Point-estimate EBEs carry no certainty, so shrunken ones
vote equally with well-determined ones and flatten g(c).
=#


############################################################################################
## Fix — Joint estimation with split-NN restores "more data = better"
##
## NN_η(η) reshapes the residual (no dead mode); NN_cov(c) absorbs the covariate effect;
## joint likelihood weights each subject by information content (no shrinkage trap).
############################################################################################

# Split-NN: NN_η(η) absorbs residual structure, NN_cov(c) carries the
# covariate effect, fit jointly against the raw likelihood.

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

# Conditional marginal at c = ±1 — dead mode should be gone.
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
    NN_η   ∈ MLPDomain(1, 6, 6, (1, identity, false); reg=L2(1e-2))
    NN_cov ∈ MLPDomain(1, 4,    (1, identity, false); reg=L2(1))
    σ ∈ RealDomain(; lower=0., init=1.0)
  end
  @random η ~ Normal(0, 1)
  @covariates μ_center
  @pre μ = NN_η(η)[1] + NN_cov(μ_center)[1]
  @derived y ~ @. Normal(μ, σ)
end

# Fit split-NN on both populations for the rich vs mix contrast.
fpm_fm2_split_rich = fit(model_fm2_split, pop_rich, init_params(model_fm2_split), MAP(FOCE());
                          optim_options=(; iterations=300))
fpm_fm2_split_mix  = fit(model_fm2_split, pop_mix,  init_params(model_fm2_split), MAP(FOCE());
                          optim_options=(; iterations=300))

# NN_cov is the analogue of g(c).
g_split_rich_grid = [coef(fpm_fm2_split_rich).NN_cov(c)[1] for c in c_grid]
g_split_mix_grid  = [coef(fpm_fm2_split_mix ).NN_cov(c)[1] for c in c_grid]

# Augment-panel lines diverge with more data; split-NN lines overlap.
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
Joint estimation restores "more data = better".  Augment stays in the
toolbox for cost (big models fit much faster post-hoc) and workflow
separability (each stage inspectable in isolation).
=#

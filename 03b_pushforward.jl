using Random
using Distributions
using DeepPumas
using CairoMakie
using AlgebraOfGraphics
using DataFramesMeta

set_mlp_backend(:staticflux)
set_theme!(deep_light(); backgroundcolor=:white)

#=
# Latent pushforward — NLME random effects as distribution transformers

A neural network transforms the η distribution itself.  Three stages:

  1. Pure pushforward — η ~ N(0,1) → bimodal observations via NN
  2. With a covariate — μ_center absorbs the bimodality; latent simplifies
  3. With time         — same machinery, different marginal at each t
=#


############################################################################################
## Stage 1 — Pure pushforward: η  →  bimodal y
############################################################################################

# Truth: each subject sits at μ_center = ±1, plus BSV ω, plus per-obs noise.
# The fit model never sees μ_center — the NN must learn to push N(0,1) into
# the bimodal subject-mean distribution.

Random.seed!(42)
nsubj = 400

truth_pushforward = @model begin
  @param begin
    σ ∈ RealDomain(; lower=0., init=0.1)
    ω ∈ RealDomain(; lower=0., init=0.2)
  end
  @random η_true ~ Normal(0, 1)
  @covariates μ_center
  @pre μ = μ_center + ω * η_true
  @derived y ~ @. Normal(μ, σ)
end

subjects_s1 = [
  Subject(; id=i, covariates=(; μ_center = rand((-1.0, 1.0))))
  for i in 1:nsubj
]

p_truth = (; σ=0.1, ω=0.2)
sims_s1 = simobs(truth_pushforward, subjects_s1, p_truth; obstimes=0.0:1.0:3.0)
pop_s1 = Subject.(sims_s1)

# Visualize the bimodal marginal.
y_data_s1 = reduce(vcat, [s.observations.y for s in pop_s1])
hist(y_data_s1; bins=40, axis=(; xlabel="y", ylabel="count",
                                title="Stage 1 data — bimodal marginal"))


# The fit model: η ~ N(0, 1), with an NN mapping η → mean of y.
model_s1 = @model begin
  @param begin
    NN ∈ MLPDomain(1, 8, 8, (1, identity); reg=L2(1e-2))
    σ ∈ RealDomain(; lower=0., init=0.2)
  end
  @random η ~ Normal(0, 1)
  @pre μ = NN(η)[1]
  @derived begin
    y ~ @. Normal(μ, σ)
  end
end

fpm_s1 = fit(
  model_s1,
  pop_s1,
  init_params(model_s1),
  MAP(FOCE());
  optim_options=(; iterations=300),
)


# (a) the learned pushforward NN(η)
nn = coef(fpm_s1).NN
lines(-3:0.01:3, η -> first(nn(η));
      axis=(; xlabel="η", ylabel="NN(η)", title="Learned pushforward NN(η)"))

# (b) fitted NN(η) vs the truth's μ distribution (both noiseless subject means).
nsamp   = 10_000
μ_model = first.(nn.(randn(nsamp)))
μ_truth = rand((-1.0, 1.0), nsamp) .+ p_truth.ω .* randn(nsamp)   # matches truth ω=0.2
df_μ = DataFrame(
  μ      = vcat(μ_model, μ_truth),
  source = vcat(fill("fitted NN(η)",      nsamp),
                fill("truth pushforward", nsamp)),
)
data(df_μ) * mapping(:μ, color=:source) * AlgebraOfGraphics.density() |> draw

# (c) empirical-Bayes η distribution, coloured by the (hidden) true mode
df_pred = DataFrame(predict(fpm_s1))
data(unique(df_pred, :id)) *
  mapping(:η, color=:μ_center => nonnumeric) * AlgebraOfGraphics.density(datalimits=extrema) |> draw

#=
The NN learned a near-step function: tight at μ ≈ ±1, sharp through zero.
That's the pushforward turning N(0,1) into the bimodal marginal.
=#


############################################################################################
## Stage 2 — A covariate explains the bimodality
############################################################################################

# Now expose μ_center to the NN.  With the covariate carrying the mode,
# η no longer has to encode it — the conditional pushforward should flatten
# and the EBEs should stop clustering by mode.

model_s2 = @model begin
  @param begin
    NN ∈ MLPDomain(2, 8, 8, (1, identity); reg=L2(1e0))
    σ ∈ RealDomain(; lower=0., init=0.2)
  end
  @covariates μ_center
  @random η ~ Normal(0, 1)
  @pre μ = NN(η, μ_center)[1]
  @derived y ~ @. Normal(μ, σ)
end

fpm_s2 = fit(
  model_s2,
  pop_s1,
  init_params(model_s2),
  MAP(FOCE());
  optim_options=(; iterations=300),
)

# NN(η, c=0) vs NN(η, c=1) — two conditional pushforwards
nn2 = coef(fpm_s2).NN
fig = lines(-3:0.01:3, η -> first(nn2(η, 0));
            label="c = 0",
            axis=(; xlabel="η", ylabel="NN(η, c)",
                    title="Stage 2 — pushforward conditioned on c"))
lines!(-3:0.01:3, η -> first(nn2(η, 1)); label="c = 1")
axislegend()
fig



# (b) fitted NN(η | μ_center) vs the truth's μ distribution, faceted by mode.
nsamp   = 10_000
μ_sample = rand([-1, 1], nsamp)
μ_model = first.(nn2.(randn(nsamp), μ_sample))
μ_truth = μ_sample .+ p_truth.ω .* randn(nsamp)   # matches truth ω=0.2
df_μ = DataFrame(
  μ      = vcat(μ_model, μ_truth),
  c      = Symbol.(vcat(μ_sample, μ_sample)),
  source = vcat(fill("fitted NN(η | c)",      nsamp),
                fill("truth pushforward", nsamp)),
)

data(df_μ) * mapping(:μ; color=:source, row=:c) * AlgebraOfGraphics.density() |> draw

# (c) EBE η — should now overlap across modes, since μ_center carries the
# structure that η carried in Stage 1.  Compare to Stage 1's panel (c).
df_pred_s2 = DataFrame(predict(fpm_s2))
data(unique(df_pred_s2, :id)) *
  mapping(:η, color=:μ_center => nonnumeric) * AlgebraOfGraphics.density(datalimits=extrema) |> draw

#=
The conditional pushforwards flatten — η-dependence collapses, the mode
shift is carried entirely by μ_center.  EBE η stops splitting by mode.

Foreshadow (Day 2 afternoon): `DeepPumas.augment` is the post-hoc analogue —
add a covariate to a fitted model to absorb structure η was carrying.  It
keeps the original η-prior and only shifts its mean by g(c), so it has
some specific failure modes we'll come back to.
=#


############################################################################################
## Stage 3 — Time-dependent pushforward: bridge to dynamical NLME
############################################################################################

# NN now takes (t, η) and outputs μ(t).  Same machinery as before, but the
# marginal of NN(t, η) over η ~ N(0,1) now changes with t.

truth_s3 = @model begin
  @param begin
    σ ∈ RealDomain(; lower=0., init=0.05)
    ω ∈ RealDomain(; lower=0., init=0.15)
  end
  @random η_true ~ Normal(0, 1)
  @covariates μ_center
  @pre μ_t = (μ_center + ω * η_true) * sin(2π * t)   # mode-dependent amplitude + BSV
  @derived y ~ @. Normal(μ_t, σ)
end

subjects_s3 = [
  Subject(; id=i, covariates=(; μ_center = rand((-1.0, 1.0))))
  for i in 1:nsubj
]
sims_s3 = simobs(truth_s3, subjects_s3, (; σ=0.05, ω=0.15); obstimes=0:0.1:1)
pop_s3 = Subject.(sims_s3)

plotgrid(pop_s3[1:12]; ylabel="y(t)")

model_s3 = @model begin
  @param begin
    NN ∈ MLPDomain(2, 14, 14, (1, identity); reg=L2(1e-2))
    σ ∈ RealDomain(; lower=0., init=0.1)
  end
  @random η ~ Normal(0, 1)
  @pre μ = NN(t, η)[1]
  @derived y ~ @. Normal(μ, σ)
end

fpm_s3 = fit(
  model_s3,
  pop_s3,
  init_params(model_s3),
  MAP(FOCE());
  optim_options=(; iterations=300, time_limit=2*60),
)

# (a) Prior trajectories — sample η ~ N(0,1) and trace NN(t, η).
nn3 = coef(fpm_s3).NN
let
    trange = collect(0:0.01:1)
    df_traj = mapreduce(vcat, 1:50) do i
        η = randn()
        DataFrame(t = trange, μ = first.(nn3.(trange, η)), η = η, id = string(i))
    end
    data(df_traj) * mapping(:t, :μ; color=:η, group=:id) * visual(Lines) |> draw
end

# (b) Marginal of NN(t, η) at three fixed t — same NN, different distribution.
let
    nsamp = 5000
    df_marg = mapreduce(vcat, [0.25, 0.5, 0.75]) do t
        η = randn(nsamp)
        DataFrame(t = "t = $t", μ = first.(nn3.(t, η)))
    end
    plt = data(df_marg) *
          mapping(:μ; col=:t => sorter("t = 0.25", "t = 0.5", "t = 0.75")) *
          AlgebraOfGraphics.density()
    draw(plt; facet=(; linkyaxes=false))
end

#=
Bimodal at amplitude peaks, collapsing toward zero at the sine's zero crossing.
Random effects + a (possibly time-dependent) pushforward = the marginal
distribution we observe.  DeepNLME generalises this: replace NN(t, η) with
an ODE parameterised by NN(η).
=#


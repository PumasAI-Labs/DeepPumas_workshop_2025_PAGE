using Pkg
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
    NN ∈ MLPDomain(1, 3, 3, (1, identity); reg=L2(1e-2))
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
begin
  nsamp   = 10_000
  μ_model = first.(nn.(randn(nsamp)))
  μ_truth = rand((-1.0, 1.0), nsamp) .+ p_truth.ω .* randn(nsamp)   # matches truth ω=0.2
  df_μ = DataFrame(
    μ      = vcat(μ_model, μ_truth),
    source = vcat(fill("NN(η) pushforward",      nsamp),
                  fill("truth pushforward", nsamp)),
  )
  data(df_μ) * mapping(:μ, color=:source) * AlgebraOfGraphics.density() |> draw
end

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
    NN ∈ MLPDomain(2, 3, 3, (1, identity); reg=L2(1e0))
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
begin
  fig = lines(-3:0.01:3, η -> first(nn2(η, -1));
              label="μ_center = -1",
              axis=(; xlabel="η", ylabel="NN(η, c)",
                      title="Stage 2 — pushforward conditioned on c"))
  lines!(-3:0.01:3, η -> first(nn2(η, 1)); label="μ_center = 1")
  axislegend()
  fig
end



# (b) fitted NN(η | μ_center) vs the truth's μ distribution, faceted by mode.
begin
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
end

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


############################################################################################
## Stage 4 (optional) — One random effect for two-dimensional heterogeneity
##
## Stages 1–3 matched the model's latent dimension to the truth.  Here the truth varies along
## *two* independent directions but we hand the model only one η.  The interesting thing is
## not that it breaks — it doesn't — but what it does with the constraint.
############################################################################################

# Truth: a saturating time-course with two independent BSV dimensions.
#   c1 → the height,   c2 → the half-saturation time.
# These move a subject's curve in genuinely different ways; no single number stands in
# for both.
Random.seed!(8)

truth_s4 = @model begin
  @param σ ∈ RealDomain(; lower=0., init=0.05)
  @random begin
    c1 ~ Uniform(0.5, 1.5)
    c2 ~ LogNormal(-2, 1.0)
  end
  @pre X = c1 * t / (t + c2)
  @derived Y ~ @. Normal(X, σ)
end

sims_s4     = simobs(truth_s4, [Subject(; id=i) for i in 1:200], (; σ=0.05); obstimes=0:0.05:1)
trainpop_s4 = Subject.(sims_s4[1:100])
testpop_s4  = Subject.(sims_s4[101:end])

plotgrid(trainpop_s4[1:12]; ylabel="Y")


# Matched capacity: two random effects, one for each direction the data varies along.
model_s4_2d = @model begin
  @param begin
    NN ∈ MLPDomain(3, 10, 10, (1, identity); reg=L2(1.))    # inputs: t + 2 η
    σ ∈ RealDomain(; lower=0.)
  end
  @random η ~ MvNormal(2, 0.1)
  @pre X = NN(t, η)[1]
  @derived Y ~ @. Normal(X, σ)
end

# Constrained capacity: a single random effect for two-dimensional heterogeneity.
model_s4_1d = @model begin
  @param begin
    NN ∈ MLPDomain(2, 8, 8, (1, identity); reg=L2(5.))    # inputs: t + 1 η
    σ ∈ RealDomain(; lower=0.)
  end
  @random η ~ Normal(0, 1)
  @pre X = NN(t, η)[1]
  @derived Y ~ @. Normal(X, σ)
end

# If the one-η model lands in a poor optimum, restart it from `sample_params(model_s4_1d)`.
fpm_s4_2d = fit(model_s4_2d, trainpop_s4, init_params(model_s4_2d), MAP(FOCE());
                optim_options=(; iterations=300))
# Individual fits on held-out subjects.
# Two η → the ipreds track each subject, as in Stages 1–3.
plotgrid(predict(fpm_s4_2d, testpop_s4[1:12]; obstimes=0:0.01:1); ylabel="Y — 2 η (matched)")

fpm_s4_1d = fit(model_s4_1d, trainpop_s4, init_params(model_s4_1d), MAP(FOCE());
                optim_options=(; iterations=300))
# One η → still good, give or take a few spots.  It did not drop a dimension; it found the
# most informative one and rode it.
plotgrid(predict(fpm_s4_1d, testpop_s4[1:12]; obstimes=0:0.01:1); ylabel="Y — 1 η (constrained)")

# Held-out log-likelihood puts a number on the gap.
@show loglikelihood(fpm_s4_2d.model, testpop_s4, coef(fpm_s4_2d), FOCE())
@show loglikelihood(fpm_s4_1d.model, testpop_s4, coef(fpm_s4_1d), FOCE())


# The same failure as a pushforward, in the idiom of Stages 1–2.
# Embed each subject by its mean response at an early and a late time — two axes that load
# differently on (c1, c2) — and sample each model's prior pushforward of η.
begin
    t_a, t_b = 0.1, 1.0
    nsamp = 4000
    nn2 = coef(fpm_s4_2d).NN
    nn1 = coef(fpm_s4_1d).NN

    # Truth: draw the latents, map them noiselessly through the structural model.
    c1 = rand(Uniform(0.5, 1.5), nsamp)
    c2 = rand(LogNormal(-2, 1.0), nsamp)
    Xt(t) = c1 .* t ./ (t .+ c2)

    # 2-η model: a 2-D prior fills the cloud.
    H2 = 0.1 .* randn(2, nsamp)
    μ2_a = [first(nn2(t_a, H2[:, i])) for i in 1:nsamp]
    μ2_b = [first(nn2(t_b, H2[:, i])) for i in 1:nsamp]

    # 1-η model: a 1-D prior can only trace a curve through it.
    h1 = randn(nsamp)
    μ1_a = first.(nn1.(t_a, h1))
    μ1_b = first.(nn1.(t_b, h1))

    df_s4 = DataFrame(
        early  = vcat(Xt(t_a), μ2_a, μ1_a),
        late   = vcat(Xt(t_b), μ2_b, μ1_b),
        source = vcat(fill("truth", nsamp),
                      fill("2 η (matched)", nsamp),
                      fill("1 η (constrained)", nsamp)),
    )
end

# (a) The 1-D pushforward marginals — the usual view from Stages 1–2.
# Late time ≈ the dominant direction of variability: all three agree, one η reproduces it.
data(df_s4) * mapping(:late => "μ at t = $t_b"; color=:source) *
  AlgebraOfGraphics.density() |> draw

# Early time loads on the direction one η could not also keep: the 1-η density is a bit too
# peaked — under-dispersed relative to truth.  That missing spread is the variance the cutoff
# trades away.
data(df_s4) * mapping(:early => "μ at t = $t_a"; color=:source) *
  AlgebraOfGraphics.density() |> draw

# (b) The joint of the two times — the manifold view.  Truth fills a 2-D cloud; the one-η
#     pushforward threads a single 1-D curve through it, riding the principal direction of the
#     variability.  The only thing one η can't add is spread orthogonal to that curve.
data(df_s4) *
  mapping(:early => "μ at t = $t_a", :late => "μ at t = $t_b"; color=:source) *
  visual(Scatter; markersize=4, alpha=0.35, strokewidth=0) |> draw

#=
The one-η model does not fail here — and it never had to choose between c1 and c2.  Fit by
maximum likelihood, it found the single direction carrying the most recoverable information
and rode it: a 1-D nonlinear manifold threaded through 2-D heterogeneity.  The individual fits
stay good, the late-time (dominant) marginal is reproduced while the early-time one is somewhat
under-dispersed, and the joint shows how tightly that one manifold threads the cloud — the
truth's two times correlate ≈ 0.82, the one-η pushforward
≈ 0.99.  All it gives up is the spread orthogonal to the manifold, modest here.

It's a PCA cutoff in disguise: take fewer random effects than the data's heterogeneity and you
trade explained variance for simplicity, but you keep the dominant sources of variability.  The
one twist is that "dominant" is ranked by recoverable *information* — through the structural and
observation model — not by latent variance, so the kept axis is not the PC of (c1, c2).  Toggle
`obstimes` toward mostly-early or mostly-late, re-fit, and the kept direction shifts, though the
latent distribution never moved.  Easier to see than to say.

That makes it `03a`'s "best compression into the channel we chose", now as a count of dimensions.
DeepNLME turns that count into an explicit dial — and even one well-used effect keeps the
dominant variability; `04_DeepNLME.jl` carries it into the dynamics.
=#


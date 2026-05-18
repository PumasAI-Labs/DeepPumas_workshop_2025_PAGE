using Random
using Distributions
using DeepPumas
using CairoMakie
using AlgebraOfGraphics
using DataFramesMeta

set_mlp_backend(:staticflux)
set_theme!(deep_light(); backgroundcolor=:white)

#=
# Latent pushforward — how NLME random effects transform distributions

Day 1 showed how a neural network can sit *inside a dynamical system*
(NeuralODEs, UDEs). This script explores the other half of the DeepNLME
machinery: how a neural network transforms the *random-effect distribution*.

The story is built in three stages:

  1. Pure pushforward.  η ~ N(0, 1)  →  bimodal observations via an NN.
  2. With a covariate.  c absorbs the bimodality; the residual pushforward
     becomes simple.
  3. With time.         The pushforward depends on t — the same machinery
                        gives a different distribution at each time point,
                        bridging us back to the dynamical case.

Throughout, the random effect η is just N(0, 1).  All the structure in the
output distribution is learned by the neural network.
=#


############################################################################################
## Stage 1 — Pure pushforward: η  →  bimodal y
############################################################################################

#=
The data has a bimodal marginal: each subject sits in group A or B (group mean
μ_center = ±1), with some between-subject variability (BSV) around that group
mean, plus Gaussian observation noise on each measurement.  The fit model
never sees which group a subject is in — it only has a single Gaussian random
effect to work with.

The neural network must therefore learn to *transform* N(0, 1) into the
bimodal observation distribution.
=#

Random.seed!(42)
nsubj = 400

# Truth: each subject's group mean is ±1, plus BSV around that mean, plus
# per-observation Gaussian noise.  The fit model will never see μ_center.
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
sims_s1 = simobs(truth_pushforward, subjects_s1, (; σ=0.1, ω=0.2); obstimes=0.0:1.0:3.0)
pop_s1 = Subject.(sims_s1)

# Visualize the bimodal marginal.
y_data_s1 = reduce(vcat, [s.observations.y for s in pop_s1])
hist(y_data_s1; bins=40, axis=(; xlabel="y", ylabel="count",
                                title="Stage 1 data — bimodal marginal"))


# The fit model: η ~ N(0, 1), with an NN mapping η → mean of y.
model_s1 = @model begin
  @param begin
    NN ∈ MLPDomain(1, 8, 8, (1, identity); reg=L2(1e-3))
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

# (b) data marginal vs. the model's marginal predictive distribution.
#     Two distinct sources of variability are combined: the pushforward of
#     η ~ N(0,1) through NN (population variability across subjects),
#     convolved with the residual noise N(0, σ) (within-subject uncertainty
#     for an individual measurement).
sims_marg = simobs(model_s1, [Subject(; id=i) for i in 1:5000],
                   coef(fpm_s1); obstimes=[0.0])
df_marg = DataFrame(
  y = vcat(y_data_s1, reduce(vcat, [s.observations.y for s in sims_marg])),
  source = vcat(fill("data",  length(y_data_s1)),
                fill("model", 5000)),
)
data(df_marg) * mapping(:y, color=:source) * AlgebraOfGraphics.density() |> draw

# (c) empirical-Bayes η distribution, coloured by the (hidden) true mode
df_pred = DataFrame(predict(fpm_s1))
data(unique(df_pred, :id)) *
  mapping(:η, color=:μ_center => nonnumeric) * AlgebraOfGraphics.histogram() |> draw

#=
The NN learns a near-step function of η: small slope around μ = ±1, sharp
transition through zero.  That is exactly the pushforward needed to turn a
Gaussian prior into a bimodal marginal.

This is the core DeepNLME claim made visible: with a sufficiently expressive
pushforward, simple latents can encode complex marginal structure.
=#


############################################################################################
## Stage 2 — A covariate explains the bimodality
############################################################################################

#=
Now suppose we know which mode each subject is in.  We add a binary covariate
c ∈ {0, 1} indicating the mode, and let the NN see it alongside η.

Question: what happens to the learned pushforward?

Prediction: c carries the bulk of the structure, and the NN's dependence on
η becomes much milder — a near-Gaussian residual within each mode.
=#

modes_s2 = rand((-1.0, 1.0), nsubj)
subjects_s2 = [
  Subject(; id=i,
            covariates=(; μ_center = modes_s2[i],
                          c = (modes_s2[i] > 0) ? 1.0 : 0.0))
  for i in 1:nsubj
]
sims_s2 = simobs(truth_pushforward, subjects_s2, (; σ=0.1, ω=0.2); obstimes=0.0:1.0:3.0)
pop_s2 = Subject.(sims_s2)

model_s2 = @model begin
  @param begin
    NN ∈ MLPDomain(2, 8, 8, (1, identity); reg=L2(1e-3))
    σ ∈ RealDomain(; lower=0., init=0.2)
  end
  @covariates c
  @random η ~ Normal(0, 1)
  @pre μ = NN(η, c)[1]
  @derived y ~ @. Normal(μ, σ)
end

fpm_s2 = fit(
  model_s2,
  pop_s2,
  init_params(model_s2),
  MAP(FOCE());
  optim_options=(; iterations=300),
)

# NN(η, c=0) vs NN(η, c=1) — two conditional pushforwards
let
  nn = coef(fpm_s2).NN
  fig = lines(-3:0.01:3, η -> first(nn(η, 0));
              label="c = 0",
              axis=(; xlabel="η", ylabel="NN(η, c)",
                      title="Stage 2 — pushforward conditioned on c"))
  lines!(-3:0.01:3, η -> first(nn(η, 1)); label="c = 1")
  axislegend()
  fig
end

#=
The two conditional pushforwards sit near μ = −1 and μ = +1 with very mild
η-dependence.  The "step" that Stage 1's NN had to learn has been absorbed
by c; the residual latent structure is near-Gaussian within each mode.

Same machinery, but now the model is structurally simpler because the
covariate is doing the work that the latent transformation was forced to do
before.

(Foreshadow for Day 2 afternoon: this is also the conceptual baseline for
`DeepPumas.augment` — adding a covariate to absorb structure carried by η.
Note though that `augment` does not refit the pushforward, so it cannot
recover this simplification on an already-fitted model.)
=#


############################################################################################
## Stage 3 — Time-dependent pushforward: bridge to dynamical NLME
############################################################################################

#=
Finally we add time.  The NN now takes (t, η) and produces the mean of y(t).
Different subjects follow different trajectories, encoded entirely through η.

The point: the very same machinery — a neural network pushforward of η —
produces a *different distribution at every time point*.  That is exactly
what an NLME model is doing when the underlying dynamics depend on
individualized parameters.

Conceptually this is one step away from DeepNLME with explicit dynamics:
instead of putting the NN directly on (t, η), DeepNLME puts the NN's output
into the parameters of an ODE and lets the dynamics carry the temporal
structure.
=#

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
    NN ∈ MLPDomain(2, 10, 10, (1, identity); reg=L2(1e-3))
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

# Per-subject predicted trajectories, coloured by EB η
df_pred_s3 = DataFrame(predict(fpm_s3; obstimes=0:0.01:1))
data(df_pred_s3) * mapping(:time, :y_ipred, color=:η, group=:id) * visual(Lines) |> draw

#=
Each subject's trajectory is a different curve through the (t, NN(t, η))
surface.  The marginal distribution of y at any fixed t is the pushforward of
N(0, 1) through NN(t, ⋅) — and that distribution's shape changes with t.

This is the entire engine of NLME viewed as a transformation of latents:
random effects + a (possibly time-dependent) pushforward = the marginal
distribution we observe.  DeepNLME swaps "static t-dependent NN" for
"ODE parameterized by NN(η)", but the conceptual story is the same.
=#

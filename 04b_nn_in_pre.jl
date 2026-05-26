using Random
using DeepPumas
using CairoMakie
using PairPlots
using PumasPlots

set_mlp_backend(:staticflux)
set_theme!(deep_light())

#=
# DeepNLME variant: NN in @pre, dynamics stay mechanistic

A different way to use a neural network inside an NLME model.  Instead of
having the NN replace a dynamical term (as in `04_DeepNLME.jl`), here the
NN replaces the *η-to-parameter map*.  The mechanism — a 2-compartment
oral PK with closed-form solution — stays intact.  The NN's job is to
translate random effects (and optionally covariates) into parameter
perturbations on (Ka, CL, Vc, Q, Vp).

Two variants:
  • model_pre        — NN takes only η.
  • model_pre_cov    — NN takes η AND covariates.

η is fixed at N(0, I): scaling/correlation would otherwise be
non-identifiable with the NN's first-layer weights.

The closed-form PK keeps fits fast enough for a workshop.
=#


############################################################################################
## Truth and data — 2-compartment PK with depot
############################################################################################

datamodel = @model begin
  @param begin
    tvKa ∈ RealDomain(; lower=0, init=0.5)
    tvCL ∈ RealDomain(; lower=0, init=1.0)
    tvVc ∈ RealDomain(; lower=0, init=10.0)
    tvQ  ∈ RealDomain(; lower=0, init=1.5)
    tvVp ∈ RealDomain(; lower=0, init=20.0)
    Ω    ∈ PDiagDomain(; init=fill(0.04, 5))
    σ_pk ∈ RealDomain(; lower=0, init=0.05)
  end
  @random η ~ MvNormal(Ω)
  @covariates genotype c_wt c_age
  @pre begin
    Ka = tvKa * exp(η[1])
    # Binary pharmacogenetic effect: extensive metabolizer (genotype=1) ⇒ CL ×2, Q ×0.5.
    # Bimodal CL/Q distributions — classical NLME's Gaussian Ω cannot reproduce them.
    CL = tvCL * exp(η[2] + log(2) * genotype + 0.3 * c_age)
    Vc = tvVc * exp(η[3] + 0.5 * c_wt)
    Q  = tvQ  * exp(η[4] - log(2) * genotype)
    Vp = tvVp * exp(η[5] + 0.4 * c_wt)
  end
  @dynamics Depots1Central1Periph1
  @derived cp ~ @. Normal(Central / Vc, σ_pk * Central / Vc + 1e-3)
end

p_data = (;
  tvKa=0.5, tvCL=1.0, tvVc=10.0, tvQ=1.5, tvVp=20.0,
  Ω=Diagonal(fill(0.04, 5)),
  σ_pk=0.05,
)

dr = DosageRegimen(50.0)
pop = synthetic_data(
  datamodel, dr, p_data;
  covariates=(;
    genotype = Bernoulli(0.5),   # binary pharmacogenetic split
    c_wt     = Normal(),         # standardised weight
    c_age    = Normal(),         # standardised age
  ),
  nsubj=200, rng=MersenneTwister(123),
  obstimes=[0.25, 0.5, 1, 2, 4, 8, 12, 24],
)

trainpop = pop[1:100]
testpop  = pop[101:200]


############################################################################################
## Baseline — classical NLME (no NN, fitted Ω)
############################################################################################
# What a pharmacometrician would write by default: each parameter has its own
# log-normal random effect with a fitted variance.  This is the case to beat.
# It can't capture nonlinear η-to-parameter relationships, and it can't see
# covariates.

model_classical = @model begin
  @param begin
    tvKa ∈ RealDomain(; lower=0, init=0.5)
    tvCL ∈ RealDomain(; lower=0, init=1.0)
    tvVc ∈ RealDomain(; lower=0, init=10.0)
    tvQ  ∈ RealDomain(; lower=0, init=1.5)
    tvVp ∈ RealDomain(; lower=0, init=20.0)
    Ω    ∈ PDiagDomain(; init=fill(0.05, 5))
    σ_pk ∈ RealDomain(; lower=0, init=0.05)
  end
  @random η ~ MvNormal(Ω)
  @pre begin
    Ka = tvKa * exp(η[1])
    CL = tvCL * exp(η[2])
    Vc = tvVc * exp(η[3])
    Q  = tvQ  * exp(η[4])
    Vp = tvVp * exp(η[5])
  end
  @dynamics Depots1Central1Periph1
  @derived cp ~ @. Normal(Central / Vc, σ_pk * Central / Vc + 1e-3)
end

fpm_classical = fit(
  model_classical, trainpop, init_params(model_classical), MAP(FOCE());
  optim_options=(; iterations=200),
)


############################################################################################
## Model A — NN takes only η
############################################################################################
# NN learns a flexible η → 5-parameter-perturbation map; mechanism intact.
# Covariate effects from the truth cannot be captured.

model_pre = @model begin
  @param begin
    NN ∈ MLPDomain(5, 8, 8, (5, identity, false); reg=L2(1e1))
    tvKa ∈ RealDomain(; lower=0, init=0.5)
    tvCL ∈ RealDomain(; lower=0, init=1.0)
    tvVc ∈ RealDomain(; lower=0, init=10.0)
    tvQ  ∈ RealDomain(; lower=0, init=1.5)
    tvVp ∈ RealDomain(; lower=0, init=20.0)
    σ_pk ∈ RealDomain(; lower=0, init=0.05)
  end
  @random η ~ MvNormal(5, 1.0)   # fixed at N(0, I); NN's first layer absorbs scaling
  @pre begin
    perturb = NN(η[1], η[2], η[3], η[4], η[5])
    Ka = tvKa * exp(perturb[1])
    CL = tvCL * exp(perturb[2])
    Vc = tvVc * exp(perturb[3])
    Q  = tvQ  * exp(perturb[4])
    Vp = tvVp * exp(perturb[5])
  end
  @dynamics Depots1Central1Periph1
  @derived cp ~ @. Normal(Central / Vc, σ_pk * Central / Vc + 1e-3)
end

fpm_pre = fit(
  model_pre, trainpop, init_params(model_pre), MAP(FOCE());
  optim_options=(; iterations=200),
)

pred_pre = predict(model_pre, testpop[1:24], coef(fpm_pre); obstimes=0:0.1:24);
plotgrid(pred_pre; observation=:cp)


############################################################################################
## Model B — split NN: separate channels for η and covariates
############################################################################################
# Two NNs map to the same 5 parameter perturbations.  NN_η learns nonlinear
# η-to-perturbation structure; NN_cov learns covariate-driven shifts.  They
# add in log-space (mirrors classical `θ * exp(η + β·cov)` but with NNs
# replacing each piece).  Separate L2 buckets let us regularize the two
# channels independently — heavier on covariates, lighter on η.

model_pre_cov = @model begin
  @param begin
    # Output-layer biases dropped (last-layer `false`) on both NNs.  At η = 0
    # and zero-valued covariates each NN outputs zero, so tvKa..tvVp keep
    # their meaning as the typical-subject baseline.  Without this the
    # biases trade off with the tv* values and the fitted typical-value
    # parameters drift from the truth.
    NN_η   ∈ MLPDomain(5, 8, 8, (5, identity, false); reg=L2(0.5))
    NN_cov ∈ MLPDomain(3, 4,    (5, identity, false); reg=L2(10.0))
    tvKa ∈ RealDomain(; lower=0, init=0.5)
    tvCL ∈ RealDomain(; lower=0, init=1.0)
    tvVc ∈ RealDomain(; lower=0, init=10.0)
    tvQ  ∈ RealDomain(; lower=0, init=1.5)
    tvVp ∈ RealDomain(; lower=0, init=20.0)
    σ_pk ∈ RealDomain(; lower=0, init=0.05)
  end
  @random η ~ MvNormal(5, 1.0)
  @covariates genotype c_wt c_age
  @pre begin
    p_η   = NN_η(η[1], η[2], η[3], η[4], η[5])
    p_cov = NN_cov(genotype, c_wt, c_age)
    Ka = tvKa * exp(p_η[1] + p_cov[1])
    CL = tvCL * exp(p_η[2] + p_cov[2])
    Vc = tvVc * exp(p_η[3] + p_cov[3])
    Q  = tvQ  * exp(p_η[4] + p_cov[4])
    Vp = tvVp * exp(p_η[5] + p_cov[5])
  end
  @dynamics Depots1Central1Periph1
  @derived cp ~ @. Normal(Central / Vc, σ_pk * Central / Vc + 1e-3)
end

fpm_pre_cov = fit(
  model_pre_cov, trainpop, init_params(model_pre_cov), MAP(FOCE());
  optim_options=(; iterations=200),
)

pred_pre_cov = predict(model_pre_cov, testpop, coef(fpm_pre_cov); obstimes=0:0.1:24);
plotgrid(pred_pre_cov[1:12]; observation=:cp)


############################################################################################
## Compare
############################################################################################
#
begin
  println("Loglikelihoods on train population:")
  println("  model_classical (Ω only, baseline):  ",
          round(loglikelihood(model_classical, trainpop, coef(fpm_classical), FOCE()); digits=2))
  println("  model_pre       (NN ← η only):       ",
          round(loglikelihood(model_pre, trainpop, coef(fpm_pre), FOCE()); digits=2))
  println("  model_pre_cov   (NN ← η + covariates):",
          round(loglikelihood(model_pre_cov, trainpop, coef(fpm_pre_cov), FOCE()); digits=2))
end

begin
  println("Loglikelihoods on test population:")
  println("  model_classical (Ω only, baseline):  ",
          round(loglikelihood(model_classical, testpop, coef(fpm_classical), FOCE()); digits=2))
  println("  model_pre       (NN ← η only):       ",
          round(loglikelihood(model_pre, testpop, coef(fpm_pre), FOCE()); digits=2))
  println("  model_pre_cov   (NN ← η + covariates):",
          round(loglikelihood(model_pre_cov, testpop, coef(fpm_pre_cov), FOCE()); digits=2))
end

############################################################################################
## Visual Predictive Checks — prior predictive at the observation level
############################################################################################
# Sample many synthetic subjects from each fitted model (η from its prior,
# covariates from the observed distribution where applicable), simulate cp
# trajectories, overlay percentile bands on the observed test data.
#
# The classical baseline is what the textbook NLME workflow would produce.
# model_pre adds NN flexibility on the η-to-parameter map.
# model_pre_cov additionally lets the NN see covariates.
# Compare the percentile-band fidelity across the three.

vpc_classical = vpc(fpm_classical; observations=[:cp])
vpc_plot(vpc_classical)

vpc_pre = vpc(fpm_pre; observations=[:cp])
vpc_plot(vpc_pre)

vpc_pre_cov = vpc(fpm_pre_cov; observations=[:cp])
vpc_plot(vpc_pre_cov)


############################################################################################
## Visualise the 5-D pushforward as a pairplot
############################################################################################
# For each of truth / model_pre / model_pre_cov: sample fresh subjects (covariates from
# the same distributions as the data, η from the appropriate prior), evaluate the @pre
# parameter values per subject, and stack into one DataFrame per source.

let
  nsamp = 5000
  rng   = MersenneTwister(456)

  genotype = rand(rng, Bernoulli(0.5), nsamp) .|> Float64
  c_wt     = randn(rng, nsamp)
  c_age    = randn(rng, nsamp)

  # Truth pushforward (uses p_data.Ω).
  η_t = [rand(rng, MvNormal(zeros(5), p_data.Ω)) for _ in 1:nsamp]
  df_truth = DataFrame(map(1:nsamp) do i
    η = η_t[i]
    (; Ka = p_data.tvKa * exp(η[1]),
       CL = p_data.tvCL * exp(η[2] + log(2) * genotype[i] + 0.3 * c_age[i]),
       Vc = p_data.tvVc * exp(η[3] + 0.5 * c_wt[i]),
       Q  = p_data.tvQ  * exp(η[4] - log(2) * genotype[i]),
       Vp = p_data.tvVp * exp(η[5] + 0.4 * c_wt[i]))
  end)

  # model_pre pushforward (NN of η only).  η ~ N(0, I).
  cp = coef(fpm_pre)
  df_pre = DataFrame(map(1:nsamp) do _
    η = randn(rng, 5)
    p = cp.NN(η[1], η[2], η[3], η[4], η[5])
    (; Ka = cp.tvKa * exp(p[1]),
       CL = cp.tvCL * exp(p[2]),
       Vc = cp.tvVc * exp(p[3]),
       Q  = cp.tvQ  * exp(p[4]),
       Vp = cp.tvVp * exp(p[5]))
  end)

  # model_pre_cov pushforward (split NN: NN_η for individual variation, NN_cov for covariate shift).
  cc = coef(fpm_pre_cov)
  df_pre_cov = DataFrame(map(1:nsamp) do i
    η = randn(rng, 5)
    pη = cc.NN_η(η[1], η[2], η[3], η[4], η[5])
    pc = cc.NN_cov(genotype[i], c_wt[i], c_age[i])
    (; Ka = cc.tvKa * exp(pη[1] + pc[1]),
       CL = cc.tvCL * exp(pη[2] + pc[2]),
       Vc = cc.tvVc * exp(pη[3] + pc[3]),
       Q  = cc.tvQ  * exp(pη[4] + pc[4]),
       Vp = cc.tvVp * exp(pη[5] + pc[5]))
  end)

  pairplot(
    df_truth   => (PairPlots.Scatter(color=:black,     markersize=2),),
    df_pre     => (PairPlots.Scatter(color=:tomato,    markersize=2),),
    df_pre_cov => (PairPlots.Scatter(color=:steelblue, markersize=2),),
  )
end


# Note: if FOCE is slow on your machine, swap in VEM (reverse-mode AD, often
# much faster for NN-heavy parameter sets):
#
#   using Pumas.Experimental: VEM
#   fpm_pre = fit(model_pre, trainpop, init_params(model_pre), MAP(VEM()); ...)

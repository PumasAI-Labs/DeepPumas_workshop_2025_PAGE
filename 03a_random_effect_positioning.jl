using Random

using Distributions
using LinearAlgebra
using DeepPumas
using CairoMakie
set_theme!(deep_light(); backgroundcolor=:white)

#=
# Random-effect positioning

Does fitting `CL = tvCL · exp(η_CL)` actually recover Var(log CL)?
We'll generate from a 1-cmpt IV bolus PK truth with BSV on both CL and Vc,
then fit three single-η placements:

  • model_a:  η on CL only       (textbook)
  • model_b:  η on Vc only       (also classical)
  • model_c:  η on a *learned*   linear mixture of CL and Vc
=#

Random.seed!(7)
nsubj = 100

# Truth: 1-cmpt IV bolus, Dose = 100, BSV on both CL and Vc.
Ω_truth = [0.4 0.2; 0.2 0.3]
# Ω_truth = [0.4 0.0; 0.0 0.3]   # toggle to diagonal

truth = @model begin
    @param begin
        σ ∈ RealDomain(; lower=0., init=0.5)
        tvCL ∈ RealDomain(; lower=0., init=0.8)
        tvVc ∈ RealDomain(; lower=0., init=2.)
    end
    @random begin
        η ~ MvNormal(Ω_truth)
    end
    @pre begin
        CL = tvCL * exp(η[1])
        Vc = tvVc * exp(η[2])
    end
    @dynamics Central1
    @derived y ~ @. Normal(Central/Vc, σ)
end

dose = DosageRegimen(100; cmt=:Central, time=0)
subjects = [Subject(; id=i, events=dose) for i in 1:nsubj]
p_truth = init_params(truth)
sims = simobs(truth, subjects, p_truth; obstimes=[0.25, 0.5, 1, 2, 4, 8])
pop  = Subject.(sims)

plotgrid(pop[1:12])


# --- Model A: η on CL 
model_a = @model begin
    @param begin
        tvCL ∈ RealDomain(; lower=0, init=1.0)
        tvVc ∈ RealDomain(; lower=0, init=5.0)
        ω    ∈ RealDomain(; lower=0, init=0.3)
        σ    ∈ RealDomain(; lower=0, init=0.5)
    end
    @random η ~ Normal(0, ω)
    @pre begin
        CL = tvCL * exp(η)
        Vc = tvVc
    end
    @dynamics Central1
    @derived y ~ @. Normal(Central/Vc, σ)
end

fpm_a = fit(model_a, pop, init_params(model_a), MAP(FOCE()))


# --- Model B: η on Vc
model_b = @model begin
    @param begin
        tvCL ∈ RealDomain(; lower=0, init=1.0)
        tvVc ∈ RealDomain(; lower=0, init=5.0)
        ω    ∈ RealDomain(; lower=0, init=0.3)
        σ    ∈ RealDomain(; lower=0, init=0.5)
    end
    @random η ~ Normal(0, ω)
    @pre begin
        CL = tvCL
        Vc = tvVc * exp(η)
    end
    @dynamics Central1
    @derived y ~ @. Normal(Central/Vc, σ)
end

fpm_b = fit(model_b, pop, init_params(model_b), MAP(FOCE()))


# --- Model C: η on a learned linear mixture of CL and Vc
# c=1 → model_a, c=0 → model_b, between → blend.  The data picks.
model_c = @model begin
    @param begin
        tvCL ∈ RealDomain(; lower=0, init=1.0)
        tvVc ∈ RealDomain(; lower=0, init=5.0)
        ω    ∈ RealDomain(; lower=0, init=0.3)
        c    ∈ RealDomain(; lower=0, upper=1, init=0.5)
        σ    ∈ RealDomain(; lower=0, init=0.5)
    end
    @random η ~ Normal(0, ω)
    @pre begin
        CL = tvCL * exp(c * η)
        Vc = tvVc * exp((1 - c) * η)
    end
    @dynamics Central1
    @derived y ~ @. Normal(Central/Vc, σ)
end

fpm_c = fit(model_c, pop, init_params(model_c), MAP(FOCE()))


# --- Compare ---------------------------------------------------------------
begin
    println("Loglikelihoods:")
    println("  model_a (η on CL):          ", round(loglikelihood(fpm_a); digits=2))
    println("  model_b (η on Vc):          ", round(loglikelihood(fpm_b); digits=2))
    println("  model_c (learned mixture):  ", round(loglikelihood(fpm_c); digits=2))
    println("  model_c learned c = ", round(coef(fpm_c).c; digits=3),
            "   (c=1 → all η on CL, c=0 → all η on Vc)")
end

plotgrid(predict(fpm_a; obstimes=0:0.05:12)[1:12]; ylabel="conc (model_a, η on CL)")
plotgrid(predict(fpm_b; obstimes=0:0.05:12)[1:12]; ylabel="conc (model_b, η on Vc)")
plotgrid(predict(fpm_c; obstimes=0:0.05:12)[1:12]; ylabel="conc (model_c, learned mix)")


# --- The pushforward as a (weighted) principal component of the BSV --------
# A single η can only move a subject along ONE direction in parameter space.
# The truth spreads subjects across a 2-D cloud, so each model threads a 1-D
# line through it (the linear, parameter-space cousin of 03b's Stage 4).
begin
    ca, cb, cc = coef(fpm_a), coef(fpm_b), coef(fpm_c)

    # truth's 2σ BSV ellipse and its latent first principal component
    θ = range(0, 2π; length=200)
    circ = reduce(hcat, [[cos(t), sin(t)] for t in θ])
    E = eigen(Symmetric(Ω_truth))
    ell = 2 .* (E.vectors * Diagonal(sqrt.(E.values)) * circ)
    pc1 = E.vectors[:, argmax(E.values)] .* (2 * sqrt(maximum(E.values)))

    # each model's η as the ray it rides: ±2σ of the (Δlog CL, Δlog Vc) it induces
    ray(d) = (2 .* [-d[1], d[1]], 2 .* [-d[2], d[2]])
    ra = ray([ca.ω, 0.0])                       # η → CL only
    rb = ray([0.0, cb.ω])                       # η → Vc only
    rc = ray([cc.ω * cc.c, cc.ω * (1 - cc.c)])  # η → learned mix

    fig = Figure(; size=(560, 520))
    ax = Axis(fig[1, 1]; xlabel="Δ log CL", ylabel="Δ log Vc", aspect=DataAspect(),
              title="A single η rides one direction through the BSV")
    lines!(ax, ell[1, :], ell[2, :]; color=(:gray, 0.9), linewidth=2, label="truth BSV (2σ)")
    lines!(ax, [-pc1[1], pc1[1]], [-pc1[2], pc1[2]]; color=(:gray, 0.6),
           linestyle=:dot, linewidth=2, label="latent PC1")
    lines!(ax, ra...; color=:tomato, linewidth=3, label="model_a (η→CL)")
    lines!(ax, rb...; color=:dodgerblue, linewidth=3, label="model_b (η→Vc)")
    lines!(ax, rc...; color=:seagreen, linewidth=4,
           label="model_c (learned mix, c=$(round(cc.c; digits=2)))")
    axislegend(ax; position=:rb, labelsize=10)
    fig
end
# model_c rotates onto the ellipse's major axis — a rough principal component
# of the BSV — but lands slightly off the *latent* PC1, tilted toward Vc.
# Ray length is each model's fitted 2σ: model_a's overshoots the ellipse because
# it inflates ω trying to push all heterogeneity through the wrong channel.


# --- Why the tilt is toward Vc: observability, not latent variance ----------
# How strongly does a unit of log-BSV in each parameter move what we measure?
begin
    conc(CL, Vc, t) = (100 / Vc) * exp(-(CL / Vc) * t)
    CL0, Vc0 = p_truth.tvCL, p_truth.tvVc
    sens_CL(t) = abs(conc(CL0, Vc0, t) * (-(CL0 / Vc0) * t))         # ∂conc/∂log CL
    sens_Vc(t) = abs(conc(CL0, Vc0, t) * (-1 + (CL0 / Vc0) * t))     # ∂conc/∂log Vc

    ts = 0:0.05:8
    obst = [0.25, 0.5, 1, 2, 4, 8]   # the actual sampling times
    fig = Figure(; size=(560, 460))
    ax = Axis(fig[1, 1]; xlabel="t", ylabel="|∂ conc / ∂ log(param)|",
              title="How much the data sees a unit of BSV")
    lines!(ax, ts, sens_CL.(ts); color=:tomato, linewidth=3, label="via CL")
    lines!(ax, ts, sens_Vc.(ts); color=:dodgerblue, linewidth=3, label="via Vc")
    scatter!(ax, obst, sens_CL.(obst); color=:tomato, markersize=9)
    scatter!(ax, obst, sens_Vc.(obst); color=:dodgerblue, markersize=9)
    axislegend(ax; position=:rt)
    fig
end
# Vc enters in two places (intercept + rate) so it dominates at the early
# samples; CL only shows up through the decay rate and starts at zero. With
# this sampling the data weights the pushforward toward Vc. Variation 3 below
# samples late, where CL's curve dominates — re-fit and watch c climb.


## Variations to try — does c shift?
##   1. Toggle Ω diagonal (line ~30)
##   2. Skew variances:  Ω = [0.9 0; 0 0.09]  (CL 10× more BSV)
##   3. Late observations only:  obstimes = [4, 8, 12, 16]

#=
model_a (η on CL) loses badly even though Ω_CL is the largest BSV in the
truth.  model_b (η on Vc) wins because Vc enters the observed
concentration in two places (intercept and rate), so each unit of Vc BSV
carries more observable information.  model_c lands near c ≈ 0.5 whether
Ω is correlated or diagonal — it picks the direction along which BSV
translates into observable structure, not the parameter with the most BSV.

The fitted η is the model's best compression of subject heterogeneity
into the channel we chose for it — geometrically, the single direction it
rides through the BSV cloud (the ellipse figure above).  Compare
`coef(fpm_a).ω^2` to Ω_CL — they won't match.  η_CL is not Var(log CL);
it's whatever subject variation the data could push through the CL channel
given our placement.

In `03b_pushforward.jl` we replace the scalar c with a neural network,
letting the data choose the entire *function* from η to individual
parameters.
=#

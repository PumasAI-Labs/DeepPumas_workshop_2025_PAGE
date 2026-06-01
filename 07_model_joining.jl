using Pkg
Pkg.activate(@__DIR__())

using DeepPumas
using DeepPumas.Experimental.PumasNeuralDistributions  # replace_randeffs_dist, CouplingFlow, LaplaceApproximation
using DataFrames
using Distributions
using StableRNGs
using Statistics
using CairoMakie
set_theme!(deep_light())

#=
# Post-hoc model joining — letting one endpoint inform another

Two endpoints, A and B, are measured on the same patients. We fit a model for
each endpoint *independently* — neither model knows the other exists. Yet the
per-subject random effects are correlated *in the data*: a patient who runs high
on A tends to run high on B.

An independently-fit model can never exploit that: its prior over η_B knows
nothing about A, so observing A tells it nothing about B. Here we *join* the two
fitted models post-hoc — we learn a single joint prior over (η_A, η_B) with
`replace_randeffs_dist` — and the cross-endpoint correlation that was in the data
all along "pops into" the prior. Now observing A sharpens the prediction of B,
with no re-fitting of either structural model.
=#


############################################################################################
## 1. Simulate a 2-endpoint system with CORRELATED random effects
############################################################################################
# Each subject has a latent (θ_A, θ_B) drawn from a bivariate normal with
# correlation ρ. We then take a handful of noisy observations of each endpoint.
# The identity link μ = η keeps the focus on the joining idea, not the PK/PD.

ρ      = 0.7    # <-- the cross-endpoint correlation we build into the data
N_SUBJ = 200
K_OBS  = 8      # observations / subject / endpoint (enough to pin η down: low shrinkage)
σ_OBS  = 0.1    # within-subject observation noise

rng = StableRNG(20260601)
θ = rand(rng, MvNormal(zeros(2), [1.0 ρ; ρ 1.0]), N_SUBJ)'   # N_SUBJ × 2 true latents

obs = NamedTuple[]
for i in 1:N_SUBJ, j in 1:K_OBS
    push!(obs, (
        id   = i,
        time = float(j),
        DV_A = θ[i, 1] + σ_OBS * randn(rng),
        DV_B = θ[i, 2] + σ_OBS * randn(rng),
    ))
end
df = DataFrame(obs)

# `event_data = false`: these subjects only carry observations, no dosing events.
pop_A     = read_pumas(df; observations = [:DV_A], event_data = false)
pop_B     = read_pumas(df; observations = [:DV_B], event_data = false)
pop = read_pumas(df; observations = [:DV_A, :DV_B], event_data = false)


############################################################################################
## 2. Fit each endpoint INDEPENDENTLY
############################################################################################
# Simplest possible NLME per endpoint: η ~ N(0, 1), μ = η, y ~ Normal(μ, σ).
# Each fit sees only its own endpoint — there is no way for it to learn that A and
# B move together.

model_A = @model begin
    @param σ ∈ RealDomain(; lower = 1e-4, init = 0.1)
    @random η ~ Normal(0, 1)
    @pre μ = η
    @derived DV_A ~ @. Normal(μ, σ)
end

model_B = @model begin
    @param σ ∈ RealDomain(; lower = 1e-4, init = 0.1)
    @random η ~ Normal(0, 1)
    @pre μ = η
    @derived DV_B ~ @. Normal(μ, σ)
end

fpm_A = fit(model_A, pop, init_params(model_A), MAP(FOCE()); optim_options = (; show_trace = false))
fpm_B = fit(model_B, pop, init_params(model_B), MAP(FOCE()); optim_options = (; show_trace = false))


############################################################################################
## 3. The correlation is already there — look at the per-subject EBEs
############################################################################################
# Stack the two independent endpoints into one joint model with INDEPENDENT priors
# (η_A and η_B each ~ N(0,1), no cross term). We do not re-fit it — we just plug in
# the σ's from the independent fits and read off the per-subject empirical Bayes
# estimates (EBEs) of (η_A, η_B).

model_joint = @model begin
    @param begin
        σ_A ∈ RealDomain(; lower = 1e-4)
        σ_B ∈ RealDomain(; lower = 1e-4)
    end
    @random begin
        η_A ~ Normal(0, 1)
        η_B ~ Normal(0, 1)
    end
    @pre begin
        μ_A = η_A
        μ_B = η_B
    end
    @derived begin
        DV_A ~ @. Normal(μ_A, σ_A)
        DV_B ~ @. Normal(μ_B, σ_B)
    end
end

joint_params = (; σ_A = coef(fpm_A).σ, σ_B = coef(fpm_B).σ)

ebe  = map(s -> empirical_bayes(model_joint, s, joint_params, FOCE()), pop)
ebeA = [e.η_A for e in ebe]
ebeB = [e.η_B for e in ebe]

@info "EBE cross-endpoint correlation" cor(ebeA, ebeB) DGM_ρ = ρ

# Even though each endpoint was fit on its own, the point estimates line up along
# a diagonal — the correlation is plainly in the data, the model just can't use it.
fig_ebe = let
    fig = Figure(; size = (520, 520))
    ax = Axis(fig[1, 1]; aspect = 1,
        xlabel = "η̂_A  (EBE, endpoint A)", ylabel = "η̂_B  (EBE, endpoint B)",
        title = "Per-subject EBEs — correlation ρ ≈ $(round(cor(ebeA, ebeB); digits = 2)) is in the data")
    scatter!(ax, ebeA, ebeB; color = (:steelblue, 0.6), markersize = 9)
    fig
end


############################################################################################
## 4. JOIN the models — learn a joint prior with `replace_randeffs_dist`
############################################################################################
# `replace_randeffs_dist` takes the two endpoints' random effects (:η_A, :η_B),
# computes their joint posterior per subject (Laplace), and fits a single joint
# distribution `η` over them — here a small normalizing flow (`CouplingFlow`). The
# learned prior carries the correlation that the independent priors threw away.

model_join, param_join = replace_randeffs_dist(
    model_joint, pop, joint_params,
    (:η_A, :η_B) => :η;                                  # stack the two REs into one joint η
    params_to_remove = (),                               # independent priors had no ω params to drop
    posterior_method = LaplaceApproximation(),
    architecture     = CouplingFlow(; ncouplings = 2, nlayers = 2),
    optim_options    = (; maxiters = 2000, early_stopping = true),
    rng              = StableRNG(20260603),
)
# Note: the first call pays a one-time compilation cost (~a minute); it is fast on re-run.

# Sample the learned joint prior and confirm the correlation "popped in".
rng_s  = StableRNG(7)
prior  = [sample_randeffs(rng_s, model_join, param_join) for _ in 1:5000]
priorA = [s.η.η_A for s in prior]
priorB = [s.η.η_B for s in prior]

@info "Learned joint-prior correlation" flow = cor(priorA, priorB) EBE = cor(ebeA, ebeB)

fig_join = let
    fig = Figure(; size = (520, 520))
    ax = Axis(fig[1, 1]; aspect = 1, xlabel = "η_A", ylabel = "η_B",
        title = "Joined prior captured the correlation")
    scatter!(ax, priorA, priorB; color = (:gray65, 0.18), markersize = 4, strokewidth = 0, label = "joined-prior samples")
    scatter!(ax, ebeA, ebeB; color = (:steelblue, 0.85), markersize = 9, strokewidth = 0, label = "EBEs (from data)")
    axislegend(ax; position = :lt)
    fig
end


############################################################################################
## 5. Payoff — observing ONLY endpoint A now predicts endpoint B
############################################################################################
# Give every subject A-data only (mask B), then estimate η_B from A alone:
#   * independent prior → η_B stays at its prior mean (0): A carries no information.
#   * joined prior      → η_B is pulled toward the correlation-implied value: A informs B.

df_Aonly = copy(df)
df_Aonly.DV_B .= missing
pop_Aonly = read_pumas(df_Aonly; observations = [:DV_A, :DV_B], event_data = false)

ηB_indep = [empirical_bayes(model_joint, s, joint_params, FOCE()).η_B   for s in pop_Aonly]
ηB_join  = [empirical_bayes(model_join,  s, param_join,   FOCE()).η.η_B for s in pop_Aonly]

@info "Predicting held-out η_B from A-only data" cor_independent = cor(ηB_indep, θ[:, 2]) cor_joined = cor(ηB_join, θ[:, 2])
@info "Spread of the A-only η_B prediction" std_independent = std(ηB_indep) std_joined = std(ηB_join)

fig_payoff = let
    fig = Figure(; size = (560, 540))
    ax = Axis(fig[1, 1]; aspect = 1,
        xlabel = "true θ_B  (held-out endpoint B)",
        ylabel = "predicted η_B from A-only data",
        title = "Observing only A ⇒ the joined model predicts B")
    ablines!(ax, 0, 1; color = :black, linestyle = :dash, label = "identity (oracle)")
    scatter!(ax, θ[:, 2], ηB_indep; color = (:firebrick, 0.6), markersize = 9, label = "independent prior (no info)")
    scatter!(ax, θ[:, 2], ηB_join;  color = (:steelblue, 0.7), markersize = 9, label = "joined prior (A informs B)")
    axislegend(ax; position = :lt)
    fig
end

#=
What just happened: two models fitted in isolation share nothing, so endpoint A
cannot inform endpoint B. `replace_randeffs_dist` learned a *joint* prior over
their random effects from the per-subject posteriors — the cross-endpoint
correlation that lived in the data popped into the prior — and now a patient with
only A measured gets a sharpened prediction of B, for free.
=#


############################################################################################
## Your turn
############################################################################################
# 1. Change ρ at the top (try 0.0 and 0.9) and re-run. How does the payoff scatter
#    in step 5 change? (ρ = 0 ⇒ joining buys nothing; the two clouds collapse onto
#    each other at η_B ≈ 0.)
#
# 2. The join above used a small normalizing flow. For a purely linear correlation
#    like this one, a Gaussian joint prior is the natural "right tool". Swap the
#    architecture for more flow capacity — `CouplingFlow(; ncouplings = 3, nlayers = 3)`
#    — and check that the recovered correlation barely moves: the extra flexibility
#    is wasted on linear truth (and can even over-fit on small data).

using Pkg
Pkg.activate(@__DIR__())

using DeepPumas
using DeepPumas.Experimental.PumasNeuralDistributions  # replace_randeffs_dist, CouplingFlow, LaplaceApproximation
using DataFrames
using DataFramesMeta
using Distributions
using StableRNGs
using Statistics
using CairoMakie
set_theme!(deep_light())

#=
# Post-hoc model joining — let an early biomarker predict a late clinical endpoint

The same drug drives two indirect-response (IDR) endpoints in each patient:

  * a **biomarker** that responds fast and is sampled **densely and early**, and
  * a **clinical endpoint** that responds slowly and is sampled **sparsely and late**.

A patient's drug *responsiveness* is correlated across the two endpoints — strong
responders on the biomarker tend to be strong responders on the clinical endpoint.
But if we fit an IDR model to each endpoint **independently**, neither knows the
other exists, so the dense early biomarker is useless for predicting the clinical
endpoint: the prior on the clinical random effect ignores it.

Here we **join** the two independently-fitted models post-hoc — learning a single
joint prior over their random effects with `replace_randeffs_dist` — so the
cross-endpoint correlation "pops into" the prior. Then, for a new patient who has
**only the biomarker measured**, the joined model predicts their (sparsely sampled)
clinical endpoint. No structural model is re-fitted.

(The biomarker is an *early* readout by its nature — every biomarker sample falls in the
first 12 h, because that is when this fast marker carries signal. So "biomarker data"
already means "early data"; nowhere below do we select an early subset by hand.)
=#


############################################################################################
## 1. Simulate: shared PK + two IDR endpoints with correlated responsiveness
############################################################################################
# One oral dose, 1-compartment PK. Two indirect-response PD endpoints share the drug
# concentration but differ in turnover (biomarker fast, clinical slow). Each subject's
# drug responsiveness enters Smax; the two endpoints' responsiveness REs are correlated.

datamodel = @model begin
    @param begin
        tvKa ∈ RealDomain(); tvCL ∈ RealDomain(); tvVc ∈ RealDomain()
        tvSmax_bio ∈ RealDomain(); tvSC50_bio ∈ RealDomain(); tvKout_bio ∈ RealDomain(); tvR0_bio ∈ RealDomain()
        tvSmax_cli ∈ RealDomain(); tvSC50_cli ∈ RealDomain(); tvKout_cli ∈ RealDomain(); tvR0_cli ∈ RealDomain()
        Ω ∈ PSDDomain(2)
        σ_bio ∈ RealDomain(); σ_cli ∈ RealDomain()
    end
    @random η ~ MvNormal(Ω)          # η[1], η[2]: responsiveness on the two endpoints (correlated)
    @pre begin
        Ka = tvKa; CL = tvCL; Vc = tvVc
        Smax_bio = tvSmax_bio * exp(η[1])
        Smax_cli = tvSmax_cli * exp(η[2])
        SC50_bio = tvSC50_bio; Kout_bio = tvKout_bio; Kin_bio = tvR0_bio * tvKout_bio
        SC50_cli = tvSC50_cli; Kout_cli = tvKout_cli; Kin_cli = tvR0_cli * tvKout_cli
    end
    @init begin
        R_bio = tvR0_bio
        R_cli = tvR0_cli
    end
    @vars begin
        cp = max(Central / Vc, 0.0)
        EFF_bio = Smax_bio * cp^1.5 / (SC50_bio^1.5 + cp^1.5)
        EFF_cli = Smax_cli * cp^1.0 / (SC50_cli^1.0 + cp^1.0)
    end
    @dynamics begin
        Depot'   = -Ka * Depot
        Central' =  Ka * Depot - (CL / Vc) * Central
        R_bio'   =  Kin_bio * (1 + EFF_bio) - Kout_bio * R_bio   # fast turnover  → early
        R_cli'   =  Kin_cli * (1 + EFF_cli) - Kout_cli * R_cli   # slow turnover  → late
    end
    @derived begin
        biomarker ~ @. Normal(R_bio, σ_bio)
        clinical  ~ @. Normal(R_cli, σ_cli)
    end
end

ρ = 0.7                                  # <-- cross-endpoint responsiveness correlation
ω_bio, ω_cli = 0.5, 0.6
Ωtrue = [ω_bio^2  ρ*ω_bio*ω_cli; ρ*ω_bio*ω_cli  ω_cli^2]
p_data = (;
    tvKa = 0.7, tvCL = 0.5, tvVc = 1.0,
    tvSmax_bio = 2.5, tvSC50_bio = 0.05, tvKout_bio = 2.0,  tvR0_bio = 10.0,
    tvSmax_cli = 5.0, tvSC50_cli = 0.08, tvKout_cli = 0.25, tvR0_cli = 5.0,
    Ω = Ωtrue, σ_bio = 0.8, σ_cli = 0.3,
)

ntrain, ntest = 120, 60
union_times = [0.0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 24, 36, 48]
sims = simobs(
    datamodel,
    [Subject(; id = i, events = DosageRegimen(1.0)) for i in 1:(ntrain + ntest)],
    p_data; obstimes = union_times, rng = StableRNG(10),
)
ηtrue = reduce(hcat, [s.randeffs.η for s in sims])'    # (ntrain+ntest) × 2 — true responsiveness

# Dense EARLY biomarker (t ≤ 12); SPARSE LATE clinical (t ∈ {6, 12, 24, 36, 48}).
bio_times = Set([0.0, 1, 2, 3, 4, 5, 6, 8, 10, 12])
cli_times = Set([6.0, 12, 24, 36, 48])
df = @rtransform DataFrame(sims) begin
    :biomarker = :time in bio_times ? :biomarker : missing
    :clinical  = :time in cli_times ? :clinical  : missing
end

idnum(id) = parse(Int, id)
df_train = @rsubset(df, idnum(:id) <= ntrain)
df_test  = @rsubset(df, idnum(:id) >  ntrain)

# A look at the data: a fast biomarker sampled densely & early, and a slow clinical endpoint
# sampled sparsely & late (first 6 subjects; each panel on its own scale).
pop6 = read_pumas(@rsubset(df_train, idnum(:id) <= 6); observations = [:biomarker, :clinical])
fig_data_bio = plotgrid(pop6; observation = :biomarker)   # dense, early
fig_data_cli = plotgrid(pop6; observation = :clinical)    # sparse, late


############################################################################################
## 2. Fit an IDR model to each endpoint INDEPENDENTLY
############################################################################################
# PK and the structural PD constants are taken as known (characterised separately). Each
# model estimates only the population responsiveness `tvSmax`, its between-subject spread
# `ω`, and the residual error `σ`. Neither model sees the other endpoint.

model_bio = @model begin
    @param begin
        tvSmax ∈ RealDomain(; lower = 0, init = 2.0)
        ω ∈ RealDomain(; lower = 0, init = 0.4)
        σ ∈ RealDomain(; lower = 0, init = 0.8)
    end
    @random η ~ Normal(0, ω)
    @pre begin
        Ka = 0.7; CL = 0.5; Vc = 1.0           # PK fixed (characterised separately)
        Smax = tvSmax * exp(η)
        SC50 = 0.05; Kout = 2.0; Kin = 20.0
    end
    @init R = 10.0
    @vars begin
        cp = max(Central / Vc, 0.0)
        EFF = Smax * cp^1.5 / (SC50^1.5 + cp^1.5)
    end
    @dynamics begin
        Depot'   = -Ka * Depot
        Central' =  Ka * Depot - (CL / Vc) * Central
        R'       =  Kin * (1 + EFF) - Kout * R
    end
    @derived biomarker ~ @. Normal(R, σ)
end

model_cli = @model begin
    @param begin
        tvSmax ∈ RealDomain(; lower = 0, init = 4.0)
        ω ∈ RealDomain(; lower = 0, init = 0.5)
        σ ∈ RealDomain(; lower = 0, init = 0.3)
    end
    @random η ~ Normal(0, ω)
    @pre begin
        Ka = 0.7; CL = 0.5; Vc = 1.0
        Smax = tvSmax * exp(η)
        SC50 = 0.08; Kout = 0.25; Kin = 1.25
    end
    @init R = 5.0
    @vars begin
        cp = max(Central / Vc, 0.0)
        EFF = Smax * cp^1.0 / (SC50^1.0 + cp^1.0)
    end
    @dynamics begin
        Depot'   = -Ka * Depot
        Central' =  Ka * Depot - (CL / Vc) * Central
        R'       =  Kin * (1 + EFF) - Kout * R
    end
    @derived clinical ~ @. Normal(R, σ)
end

pop_bio = read_pumas(df_train; observations = [:biomarker])
pop_cli = read_pumas(df_train; observations = [:clinical])

fpm_bio = fit(model_bio, pop_bio, init_params(model_bio), MAP(FOCE()); optim_options = (; show_trace = false))
fpm_cli = fit(model_cli, pop_cli, init_params(model_cli), MAP(FOCE()); optim_options = (; show_trace = false))


############################################################################################
## 3. The correlation is already there — look at the per-subject EBEs
############################################################################################
# Stack the two endpoints into one joint model with INDEPENDENT priors (the σ's, ω's and
# tvSmax's come from the independent fits — we never re-fit it). Reading off the per-subject
# empirical Bayes estimates of the two responsiveness REs shows the correlation plainly.

model_joint = @model begin
    @param begin
        tvSmax_bio ∈ RealDomain(; lower = 0); ω_bio ∈ RealDomain(; lower = 0); σ_bio ∈ RealDomain(; lower = 0)
        tvSmax_cli ∈ RealDomain(; lower = 0); ω_cli ∈ RealDomain(; lower = 0); σ_cli ∈ RealDomain(; lower = 0)
    end
    @random begin
        η_bio ~ Normal(0, ω_bio)
        η_cli ~ Normal(0, ω_cli)
    end
    @pre begin
        Ka = 0.7; CL = 0.5; Vc = 1.0
        Smax_bio = tvSmax_bio * exp(η_bio); SC50_bio = 0.05; Kout_bio = 2.0;  Kin_bio = 20.0
        Smax_cli = tvSmax_cli * exp(η_cli); SC50_cli = 0.08; Kout_cli = 0.25; Kin_cli = 1.25
    end
    @init begin
        R_bio = 10.0
        R_cli = 5.0
    end
    @vars begin
        cp = max(Central / Vc, 0.0)
        EFF_bio = Smax_bio * cp^1.5 / (SC50_bio^1.5 + cp^1.5)
        EFF_cli = Smax_cli * cp^1.0 / (SC50_cli^1.0 + cp^1.0)
    end
    @dynamics begin
        Depot'   = -Ka * Depot
        Central' =  Ka * Depot - (CL / Vc) * Central
        R_bio'   =  Kin_bio * (1 + EFF_bio) - Kout_bio * R_bio
        R_cli'   =  Kin_cli * (1 + EFF_cli) - Kout_cli * R_cli
    end
    @derived begin
        biomarker ~ @. Normal(R_bio, σ_bio)
        clinical  ~ @. Normal(R_cli, σ_cli)
    end
end

joint_params = (;
    tvSmax_bio = coef(fpm_bio).tvSmax, ω_bio = coef(fpm_bio).ω, σ_bio = coef(fpm_bio).σ,
    tvSmax_cli = coef(fpm_cli).tvSmax, ω_cli = coef(fpm_cli).ω, σ_cli = coef(fpm_cli).σ,
)

pop_jtrain = read_pumas(df_train; observations = [:biomarker, :clinical])
ebe = map(s -> empirical_bayes(model_joint, s, joint_params, FOCE()), pop_jtrain)
ebe_bio = [e.η_bio for e in ebe]
ebe_cli = [e.η_cli for e in ebe]

@info "EBE cross-endpoint correlation" cor(ebe_bio, ebe_cli) DGM_ρ = ρ

fig_ebe = let
    fig = Figure(; size = (520, 520))
    ax = Axis(fig[1, 1]; aspect = 1,
        xlabel = "η̂  biomarker responsiveness", ylabel = "η̂  clinical responsiveness",
        title = "Per-subject EBEs — correlation ρ ≈ $(round(cor(ebe_bio, ebe_cli); digits = 2)) is in the data")
    scatter!(ax, ebe_bio, ebe_cli; color = (:steelblue, 0.6), markersize = 9)
    fig
end


############################################################################################
## 4. JOIN the models — learn a joint prior with `replace_randeffs_dist`
############################################################################################
# Take the two endpoints' random effects (:η_bio, :η_cli), compute their joint posterior per
# subject (Laplace), and fit a single joint distribution `η` over them with a small
# normalizing flow. The learned prior carries the correlation the independent priors lacked.

model_join, param_join = replace_randeffs_dist(
    model_joint, pop_jtrain, joint_params,
    (:η_bio, :η_cli) => :η;
    params_to_remove = (:ω_bio, :ω_cli),     # the independent-prior spreads are now superseded
    posterior_method = LaplaceApproximation(),
    architecture     = CouplingFlow(; ncouplings = 2, nlayers = 2),
    optim_options    = (; maxiters = 2000, early_stopping = true),
    rng              = StableRNG(20260603),
)
# Note: the first call pays a one-time compilation cost (~1.5 min); it is fast on re-run.

rng_s = StableRNG(7)
prior     = [sample_randeffs(rng_s, model_join, param_join) for _ in 1:5000]
prior_bio = [s.η.η_bio for s in prior]
prior_cli = [s.η.η_cli for s in prior]

@info "Learned joint-prior correlation" flow = cor(prior_bio, prior_cli) EBE = cor(ebe_bio, ebe_cli)

fig_join = let
    fig = Figure(; size = (520, 520))
    ax = Axis(fig[1, 1]; aspect = 1,
        xlabel = "η  biomarker responsiveness", ylabel = "η  clinical responsiveness",
        title = "Joined prior captured the correlation")
    scatter!(ax, prior_bio, prior_cli; color = (:gray65, 0.18), markersize = 4, strokewidth = 0, label = "joined-prior samples")
    scatter!(ax, ebe_bio, ebe_cli; color = (:steelblue, 0.85), markersize = 9, strokewidth = 0, label = "EBEs (from data)")
    axislegend(ax; position = :lt)
    fig
end


############################################################################################
## 5. Payoff — predict the clinical endpoint from the biomarker alone
############################################################################################
# New patients have the biomarker measured but no clinical observations (we mask them). The
# biomarker is sampled only in the early window (≤ 12 h; see `bio_times`), so keeping "the
# biomarker" already means keeping the early data — we are not hand-picking a subset here.
# Estimate each patient's clinical responsiveness from the biomarker only:
#   * independent prior → η_cli stays at its prior mean (0): the biomarker is ignored.
#   * joined prior      → η_cli is pulled toward the biomarker-implied value.

df_test_bioonly = @rtransform(df_test, :clinical = missing)
pop_test_bioonly = read_pumas(df_test_bioonly; observations = [:biomarker, :clinical])
ηtrue_test_cli = ηtrue[(ntrain + 1):end, 2]

ηcli_indep = [empirical_bayes(model_joint, s, joint_params, FOCE()).η_cli   for s in pop_test_bioonly]
ηcli_join  = [empirical_bayes(model_join,  s, param_join,   FOCE()).η.η_cli for s in pop_test_bioonly]

@info "Predict held-out clinical responsiveness from biomarker-only data" independent = cor(ηcli_indep, ηtrue_test_cli) joined = cor(ηcli_join, ηtrue_test_cli)

# Quantitative: predicted vs true clinical responsiveness.
fig_payoff = let
    fig = Figure(; size = (560, 540))
    ax = Axis(fig[1, 1]; aspect = 1,
        xlabel = "true clinical responsiveness (held out)",
        ylabel = "predicted from biomarker-only",
        title = "Biomarker predicts the clinical endpoint")
    ablines!(ax, 0, 1; color = :black, linestyle = :dash, label = "identity (oracle)")
    scatter!(ax, ηtrue_test_cli, ηcli_indep; color = (:firebrick, 0.6), markersize = 9, label = "independent (no info)")
    scatter!(ax, ηtrue_test_cli, ηcli_join;  color = (:steelblue, 0.75), markersize = 9, label = "joined (biomarker → clinical)")
    axislegend(ax; position = :lt, labelsize = 10)
    fig
end

# Intuition: predicted clinical trajectory from biomarker-only data, for 6 patients spanning
# weak → strong responders. Independent gives everyone the population curve; joined
# individualises it. Black dots are the (held-out) true clinical observations.
fig_payoff_traj = let
    pick = sortperm(ηtrue_test_cli)[round.(Int, range(1, ntest; length = 6))]
    sub  = pop_test_bioonly[pick]
    dfj = DataFrame(predict(model_join,  sub, param_join;   obstimes = 0:0.5:48))
    dfi = DataFrame(predict(model_joint, sub, joint_params; obstimes = 0:0.5:48))
    fig = Figure(; size = (940, 560)); axs = Axis[]
    for (k, sid) in enumerate(getfield.(sub, :id))
        r, c = fldmod1(k, 3)
        ax = Axis(fig[r, c]; title = "patient $sid", xlabel = r == 2 ? "time" : "", ylabel = c == 1 ? "clinical" : "")
        push!(axs, ax)
        gj  = @rsubset(dfj, :id == sid); gi = @rsubset(dfi, :id == sid)
        obs = @rsubset(df_test, :id == sid, !ismissing(:clinical))
        lines!(ax, gi.time, gi.clinical_ipred; color = :firebrick, linewidth = 2.5, label = "independent (population)")
        lines!(ax, gj.time, gj.clinical_ipred; color = :steelblue, linewidth = 2.5, label = "joined (from biomarker)")
        scatter!(ax, obs.time, obs.clinical; color = :black, markersize = 9, label = "true clinical (held out)")
        k == 1 && axislegend(ax; position = :rt, labelsize = 8)
    end
    linkaxes!(axs...)
    Label(fig[0, :], "Predicting the sparse clinical endpoint from the biomarker alone"; fontsize = 15, font = :bold)
    fig
end

#=
What just happened: two IDR models fitted in isolation share nothing, so the densely
sampled early biomarker is useless for the sparsely sampled late clinical endpoint —
its prior predicts the population mean for everyone. `replace_randeffs_dist` learned a
joint prior over the two responsiveness random effects; the cross-endpoint correlation
that lived in the data popped into the prior. Now a patient with only the biomarker
measured gets an individualised prediction of their clinical endpoint, with neither
structural model re-fitted.
=#


############################################################################################
## Your turn
############################################################################################
# 1. Set ρ = 0.0 at the top and re-run. The EBE scatter loses its tilt and the payoff
#    collapses: with no cross-endpoint correlation, the biomarker carries nothing about
#    the clinical endpoint and joining buys nothing.
#
# 2. Make the clinical endpoint sparser still — drop `cli_times` to just `Set([24.0, 48])` —
#    and compare the independent vs joined clinical predictions. The sparser the endpoint
#    you care about, the more the early biomarker (via the join) is worth.
#
# 3. Swap the flow for more capacity — `CouplingFlow(; ncouplings = 3, nlayers = 3)` — and
#    confirm the recovered correlation barely moves: the extra flexibility is wasted on a
#    correlation this close to linear (and can over-fit on small training sets).

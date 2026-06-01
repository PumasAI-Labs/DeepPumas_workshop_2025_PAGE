# Brief: build the "model joining" hands-on exercise (for the JuliaHub Claude)

**You are** a Claude Code instance running in **JuliaHub**, where this workshop's
DeepPumas environment actually runs. **Your job:** develop and *test end-to-end*
a small hands-on exercise on **post-hoc model joining** for the DeepPumas PAGE
2026 workshop (Day 2). It was scoped on a laptop without the Julia env — so the
design below is firm on *pedagogy* but you own *making it run*. Validate every
cell in the real env before declaring done.

---

## 1. The one-sentence goal

Two endpoint models are fitted **independently**; we then learn a **joint prior**
over their random effects with `replace_randeffs_dist`, and the learner **shows
that the cross-endpoint correlation — which was in the data all along — "pops
into" the prior**, so observing one endpoint now sharpens the other.

This is the payoff slide of deck `slides_typst/09_model_joining.typ`; the exercise
is its hands-on counterpart. It must visibly **show off `replace_randeffs_dist`**.

## 2. The pedagogical arc (keep it minimal and fast)

1. **Simulate** a 2-endpoint system whose per-subject random effects are
   **correlated across endpoints** in the data-generating model (DGM).
2. **Fit each endpoint independently** → each gets its *own, independent* prior
   (no cross-endpoint covariance).
3. **Scatter the per-subject EBEs** `η̂_A` vs `η̂_B` → the correlation is plainly
   *there in the data*, even though neither model knows about the other.
4. **Join** with `replace_randeffs_dist(...)`. It fits a flexible joint prior
   `p_φ` to the **target the posteriors imply** — the population-average of the
   per-patient posterior products,
   `p̃(η_A, η_B) = E_(A,B)[ p(η_A | A, θ_A) · p(η_B | B, θ_B) ]`. The factors are
   independent *by design*, yet `p̃` is **correlated** because both observations
   come from the *same patient*. The narration should name this target; the
   learned prior then carries that correlation (show its covariance / sampled
   cloud vs. the EBE scatter).
5. **Payoff:** with the joined model, **observing endpoint A shifts the
   prediction/posterior for endpoint B** — impossible under the independent fit.
   Show one or two subjects, and/or an aggregate (e.g. EBE of η_B from A-only
   data vs. oracle).

Each numbered step should be a clearly-commented section. End with a 1-2 line
"what just happened" recap mirroring the slide's takeaway.

## 3. Recommended data setup (confirm runtime, then commit)

**Core (build this first): minimal 2-endpoint, *linear* correlation.** Easiest
"correlation pops in" story, fast to fit live. The cleanest base is the toy in
`analyses/03_2d_toy/yinyang.jl` (two endpoints, **one scalar RE each, ~8 noisy
obs per subject per endpoint**) — but **replace the yin-yang DGM draw with a
correlated bivariate normal** (e.g. `cor ≈ 0.7`). That gives the clean linear
case. Use **Gaussian** as the default joint prior; it's the right tool for linear
truth and fits fast.

**Optional level-2 extension (only if time/runtime allow): the yin-yang.** Keep
the original non-Gaussian DGM and add a **Normalizing Flow** join to show the
Gaussian gains ≈0 while the flow recovers the shape (+~500 nats train). This is
the "why a flow" point — nice bonus, not required for v1.

Target: **fits + flow in well under a couple of minutes** on the workshop image.
~100-200 train subjects should suffice; verify shrinkage is low enough that the
EBE correlation is visible (enough obs/subject to identify the REs).

## 4. Concrete API (verified from the ModelJoining repo)

Reference repo: **`PumasAI-Labs/ModelJoining_PAGE2026`** (clone it in JuliaHub).
Read `CLAUDE.md` there for `@model` conventions, then mine these scripts:
- `analyses/03_2d_toy/yinyang.jl` — closest structure to the target exercise.
- `analyses/01_perfect_rich_sparse/test_nf_joining.jl` — the canonical join +
  EBE + cross-prediction (`cnll`) pipeline; copy its idioms.
- `analyses/02_pk_multi_pd/test_nf_joining_pkpd.jl` — full PKPD version (heavier).

Key calls (from `test_nf_joining.jl`):

```julia
# fit each endpoint independently
fpm1 = fit(model1, pop1_train, init_params(model1), MAP(FOCE()))
fpm2 = fit(model2, pop2_train, init_params(model2), MAP(FOCE()))

# build a joint model with INITIALLY INDEPENDENT priors (PDiagDomain), then
# learn a joint prior over the stacked random effects:
model_nf, param_nf = replace_randeffs_dist(
    model_joint, pop_joint_train, coef(fpm_joint),
    (:η1, :η2) => :η;                       # endpoints' REs → one joint vector
    params_to_remove = (:ω_cl1, :ω_v1, :ω_cl2, :ω_v2),  # the old independent ω's
    posterior_method = LaplaceApproximation(),
    architecture = CouplingFlow(; ncouplings = 2, nlayers = 3),  # NF; omit/swap for Gaussian
    rng = StableRNG(123),
)

# per-subject EBEs (the scatter in step 3)
ebe = empirical_bayes(model, subject, params, FOCE())

# evaluate / show the gain
ll = loglikelihood(model, pop, param, FOCE())
```

Notes: a **Gaussian** joint prior is the post-hoc moment-match on the pooled
posteriors (default, fast); the **`CouplingFlow`** is the NF (level-2). Confirm
the exact way to request the Gaussian family vs. flow in `replace_randeffs_dist`
against the installed DeepPumas version — check the docstring and the repo
scripts; the API is the source of truth, not this brief. `params_to_remove` must
match the independent-prior ω parameter names in your joint model.

## 5. Style — match the existing workshop exercises

Mirror `05_prognostic_factors.jl` and `03a_random_effect_positioning.jl` in the
**workshop repo** (`DeepPumas_workshop_2026_PAGE`):
- Heavy *narrative* comments; runnable top-to-bottom; clear section banners.
- Plots inline with **CairoMakie / AlgebraOfGraphics** (the repo's plotting deps).
- `StableRNG` seeds for reproducibility; activate the workshop project.
- Keep it tight — this is a *small* exercise (a lull-filler on Day 2), not a lab.
- Optionally leave 1-2 "your turn" gaps for participants (e.g. change the DGM
  correlation, or swap Gaussian↔flow) with answers in comments or a solutions copy.

## 6. Acceptance criteria (the definition of done)

- [ ] Runs **end-to-end in the JuliaHub DeepPumas image** with no manual fixups.
- [ ] Both independent fits converge; the joint via `replace_randeffs_dist` runs.
- [ ] **Step 3 scatter** clearly shows cross-endpoint EBE correlation.
- [ ] **Step 4** shows the joined prior captured that correlation (cov or cloud).
- [ ] **Step 5** demonstrates information transfer (A informs B) numerically
      and/or visually.
- [ ] Total runtime is workshop-friendly (target ≪ a couple of minutes for the
      core; note it if the NF extension is slower).
- [ ] Matches the workshop's exercise file style; lands as a new `.jl` in the
      workshop repo (suggest `07_model_joining.jl`, sequencing TBD with Niklas).

## 7. Watch out for

- **Shrinkage hiding the signal.** Too few obs/subject → EBEs shrink to 0 and the
  scatter looks uncorrelated. Use enough samples/subject (the toy uses ~8).
- **NF overfitting on small/linear data** — it needs regularization and can
  *underperform* the Gaussian on linear truth (this is a real result, Exp 1). So
  default the core exercise to **Gaussian**; frame the flow as the non-linear
  extension.
- **`params_to_remove`** must exactly match the joint model's independent-ω names.
- Confirm the **Gaussian-family** option of `replace_randeffs_dist` in the
  installed version (vs. only-flow). Adjust the narrative if only one is exposed.
- Keep the two endpoints **conditionally independent given η** (A1) — the join is
  purely through the prior, which is the whole point.

## 8. Hand back to Niklas

When it runs, report: final runtime, the exact models/DGM you settled on, whether
the NF extension made the cut, and any API surprises so the slides
(`09_model_joining.typ`) can be kept consistent with what the exercise actually does.

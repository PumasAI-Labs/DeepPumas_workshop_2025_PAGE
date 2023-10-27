using DeepPumas
using StableRNGs
using PumasUtilities
using CairoMakie
using Serialization
using Latexify
using PumasPlots
set_theme!(deep_light())

############################################################################################
## Generate synthetic data from an indirect response model (IDR) with complicated covariates
############################################################################################

## Define the data-generating model
datamodel = @model begin
    @param begin
        tvKa ∈ RealDomain(; lower = 0, init = 0.5)
        tvCL ∈ RealDomain(; lower = 0)
        tvVc ∈ RealDomain(; lower = 0)
        tvSmax ∈ RealDomain(; lower = 0, init = 0.9)
        tvn ∈ RealDomain(; lower = 0, init = 1.5)
        tvSC50 ∈ RealDomain(; lower = 0, init = 0.2)
        tvKout ∈ RealDomain(; lower = 0, init = 1.2)
        Ω ∈ PDiagDomain(; init = fill(0.05, 5))
        σ ∈ RealDomain(; lower = 0, init = 5e-2)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @covariates R_eq c1 c2 c3 c4 c5 c6
    @pre begin
        Smax = tvSmax * exp(η[1]) + 3 * c1 / (10.0 + c1) # exp(η[3] + exp(c3) / (1 + exp(c3)) + 0.05 * c4)
        SC50 = tvSC50 * exp(η[2] + 0.5 * (c2 / 20)^0.75)
        Ka = tvKa * exp(η[3] + 0.3 * c3 * c4)
        Vc = tvVc * exp(η[4] + 0.3 * c3)
        Kout = tvKout * exp(η[5] + 0.3 * c5 / (c6 + c5))
        Kin = R_eq * Kout
        CL = tvCL
        n = tvn
    end
    @init begin
        R = Kin / Kout
    end
    @vars begin
        cp = max(Central / Vc, 0)
        EFF = Smax * cp^n / (SC50^n + cp^n)
    end
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL / Vc) * Central
        R' = Kin * (1 + EFF) - Kout * R
    end
    @derived begin
        yPD ~ @. Normal(R, σ)
    end
end

render(latexify(datamodel, :pre))

## Generate synthetic data.
pop = synthetic_data(
    datamodel;
    covariates = (;
        R_eq = Gamma(5e2, 1/(5e2)), 
        c1 = Gamma(10, 1),
        c2 = Gamma(21, 1),
        c3 = Normal(),
        c4 = Normal(),
        c5 = Gamma(11, 1),
        c6 = Gamma(11, 1),
    ),
    nsubj = 1020,
    rng = StableRNG(123),
)

covariates_check(pop; markersize=4)
covariates_dist(pop)

## Split the data into different training/test populations
trainpop_small = pop[1:100]
trainpop_large = pop[1:1000]
testpop = pop[1001:end]

## Visualize the synthetic data and the predictions of the data-generating model.
pred_datamodel = predict(datamodel, testpop, init_params(datamodel); obstimes = 0:0.1:10);
plotgrid(pred_datamodel)


############################################################################################
## Neural-embedded NLME modeling
############################################################################################
# Here, we define a model where the PD is entirely deterimined by a neural network.
# At this point, we're not trying to explain how patient data may inform individual
# parameters

model = @model begin
    @param begin
        # Define a multi-layer perceptron (a neural network) which maps from 6 inputs (2
        # state variables + 4 individual parameters) to a single output. Apply L2
        # regularization (equivalent to a Normal prior).
        NN ∈ MLPDomain(6, 7, 6, (1, identity); reg = L2(0.1, input=false))
        tvKa ∈ RealDomain(; lower = 0)
        tvCL ∈ RealDomain(; lower = 0)
        tvVc ∈ RealDomain(; lower = 0)
        Ω ∈ PDiagDomain(2)
        σ ∈ RealDomain(; lower = 0)
    end
    @covariates R_eq
    @random begin
        η ~ MvNormal(Ω)
        η_nn ~ MvNormal(3, 0.5)
    end
    @pre begin
        Ka = tvKa * exp(η[1])
        Vc = tvVc * exp(η[2])
        CL = tvCL
        R₀ = R_eq

        # Fix individual parameters as static inputs to the NN and return an "individual"
        # neural network:
        iNN = fix(NN, R₀, η_nn)
    end
    @init begin
        R = R₀
    end
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL / Vc) * Central
        R' = iNN(Central, R)[1]
    end
    @derived begin
        yPD ~ @. Normal(R, σ)
    end
end

fpm = fit(
    model,
    trainpop_small,
    init_params(model),
    MAP(FOCE());
    # Some extra options to speed up the demo at the expense of a little accuracy:
    optim_options = (; iterations=300)
)
# Like any good TV-chef:
# serialize(@__DIR__() * "/assets/deep_pumas_fpm.jls", fpm)
# fpm = deserialize(@__DIR__() * "/assets/deep_pumas_fpm.jls")
fpm.optim

# The model has succeeded in discovering the dynamical model if the individual predictions
# match the observations well.
pred = predict(model, testpop, coef(fpm); obstimes = 0:0.1:10);
plotgrid(pred)

############################################################################################
## 'Augment' the model to predict heterogeneity from data
############################################################################################
# All patient heterogeneity of our recent model was captured by random effects and can thus
# not be predicted by the model. Here, we 'augment' that model with ML that's trained to 
# capture this heterogeneity from data.
#
# Data quantity is more important for covariate identification than it is for system
# identification. Prediction improvements could still be made with only 100 patients, but
# to correctly identify the covariate effects we need more data.

# Generate a target for the ML fitting from a Normal approximation of the posterior η
# distribution.
target = preprocess(fpm)

nn = MLPDomain(numinputs(target), 10, 10, (numoutputs(target), identity); reg=L2(1.))

fnn = fit(nn, target)
augmented_fpm = augment(fpm, fnn)

pred_augment =
    predict(augmented_fpm.model, testpop, coef(augmented_fpm); obstimes = 0:0.1:10);
plotgrid(
    pred_datamodel;
    ipred = false,
    pred = (; color = (:black, 0.4), label = "Best possible pred"),
)
plotgrid!(pred_augment; ipred=false, pred=(; color=:purple, linestyle=:dash))


# was that an appropriate regularization? We can automatically test a few
# different ones by calling hyperopt rather than fit.

ho = hyperopt(nn, target)
augmented_fpm = augment(fpm, ho)

pred_augment_ho =
    predict(augmented_fpm.model, testpop, coef(augmented_fpm); obstimes = 0:0.1:10);
plotgrid!(pred_augment_ho; ipred=false, pred=(; linestyle=:dash))


# We should now have gotten something pretty good but not fantastic. Training UDEs requires
# much less data than training covariate models. With UDEs, every observation is a data
# point. With prognostic factor model, every subject is a data point. To get much better
# here, we need more data.

target_large = preprocess(model, trainpop_large, coef(fpm), FOCE())
fnn_large = hyperopt(nn, target_large)
augmented_fpm_large = augment(fpm, fnn_large)


pred_augment_large =
    predict(augmented_fpm_large.model, testpop, coef(augmented_fpm_large); obstimes = 0:0.1:10);
plotgrid!(pred_augment_large; ipred=false)

ho = hyperopt(nn, target_large)
augmented_fpm_large_ho = augment(fpm, ho)
pred_augment_large_ho =
    predict(augmented_fpm.model, testpop, coef(augmented_fpm_large_ho); obstimes = 0:0.1:10);
plotgrid!(pred_augment_large_ho; ipred=false, pred=(; color=:orange))


############################################################################################
## Further refinement by fitting everything in concert
############################################################################################


# Running this fully would take hours, but we can show that it works
fpm_deep = fit(
  augmented_fpm.model,
  trainpop_large,
  coef(augmented_fpm),
  MAP(FOCE());
  optim_options = (; time_limit = 5*60),
  constantcoef = (; NN = coef(augmented_fpm).NN)
)
# serialize(@__DIR__() * "/assets/deep_pumas_fpm_deep.jls", fpm_deep)
# fpm_deep = deserialize(@__DIR__() * "/assets/deep_pumas_fpm_deep.jls")

pred_deep = predict(fpm_deep.model, testpop, coef(fpm_deep); obstimes = 0:0.1:10);
plotgrid(
    pred_datamodel;
    ipred = false,
    pred = (; color = (:black, 0.4), label = "Best possible pred"),
)
plotgrid!(pred_deep; ipred=false)

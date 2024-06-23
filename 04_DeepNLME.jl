using DeepPumas
using CairoMakie
using StableRNGs
using PumasPlots
set_theme!(deep_light())

############################################################################################
## Generate synthetic data from an indirect response model (IDR) 
############################################################################################

## Define the data-generating model
datamodel = @model begin
    @param begin
        tvKa ∈ RealDomain()
        tvCL ∈ RealDomain()
        tvVc ∈ RealDomain()
        tvSmax ∈ RealDomain()
        tvn ∈ RealDomain()
        tvSC50 ∈ RealDomain()
        tvKout ∈ RealDomain()
        tvKin ∈ RealDomain()
        Ω ∈ PDiagDomain(5)
        σ ∈ RealDomain()
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @pre begin
        Smax = tvSmax * exp(η[1]) 
        SC50 = tvSC50 * exp(η[2])
        Ka = tvKa * exp(η[3])
        Vc = tvVc * exp(η[4])
        Kout = tvKout * exp(η[5])
        Kin = tvKin
        CL = tvCL
        n = tvn
    end
    @init begin
        R = Kin / Kout
    end
    @vars begin
        cp = max(Central / Vc, 0.)
        EFF = Smax * cp^n / (SC50^n + cp^n)
    end
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL / Vc) * Central
        R' = Kin * (1 + EFF) - Kout * R
    end
    @derived begin
        Outcome ~ @. Normal(R, σ)
    end
end

p_data = (;
    tvKa = 0.5,
    tvCL = 1.,
    tvVc = 1., 
    tvSmax = 2.9,
    tvn = 1.5,
    tvSC50 = 0.05,
    tvKout = 2.2,
    tvKin = 0.8,
    Ω = Diagonal(fill(0.1, 5)),
    σ = 0.1                         ## <-- tune the observational noise of the data here
)

obstimes = 0:24
ntrain = 10
ntest = 12
pop = map(1:ntrain + ntest) do i
    rng = StableRNG(i)
    dose_1 = DosageRegimen(1.)
    dose_2 = DosageRegimen(1.; time=rand(rng, Gamma(40, 5/40)))
    sim = simobs(
        datamodel, 
        Subject(; id = i, events=DosageRegimen(dose_1, dose_2)), 
        p_data; 
        obstimes, 
        rng)
    Subject(sim)
end

trainpop = pop[1:ntrain]
testpop = pop[(ntrain+1):end]


## Visualize the synthetic data and the predictions of the data-generating model.
## The specified `obstimes` is just to get a denser timecourse so that plots look smooth.
pred_datamodel = predict(datamodel, testpop, p_data; obstimes = 0:0.1:24);
plotgrid(pred_datamodel)


############################################################################################
## Neural-embedded NLME modeling
############################################################################################
# Here, we define a model where the PD is entirely deterimined by a neural network.
# At this point, we're not trying to explain how patient data may inform individual
# parameters

model = @model begin
    @param begin
        # Define a multi-layer perceptron (a neural network) which maps from 5 inputs (2
        # state variables + 3 individual parameters) to a single output. Apply L2
        # regularization (equivalent to a Normal prior).
        NN ∈ MLPDomain(5, 7, 7, (1, identity); reg = L2(1.0))
        tvKa ∈ RealDomain(; lower = 0)
        tvCL ∈ RealDomain(; lower = 0)
        tvVc ∈ RealDomain(; lower = 0)
        tvR₀  ∈ RealDomain(; lower = 0)
        ωR₀  ∈ RealDomain(; lower = 0)
        Ω ∈ PDiagDomain(2)
        σ ∈ RealDomain(; lower = 0)
    end
    @random begin
        η ~ MvNormal(Ω)
        η_nn ~ MvNormal(3, 0.1)
    end
    @pre begin
        Ka = tvKa * exp(η[1])
        Vc = tvVc * exp(η[2])
        CL = tvCL
        
        # Letting the initial value of R depend on a random effect enables
        # its identification from observations. Note how we're using this 
        # random effect in both R₀ and as an input to the NN.
        # This is because the same information might be useful for both
        # determining the initial value and for adjusting the dynamics.
        R₀ = tvR₀ * exp(10 * ωR₀ * η_nn[1])

        # Fix random effects as non-dynamic inputs to the NN and return an "individual"
        # neural network:
        iNN = fix(NN, η_nn)
    end
    @init begin
        R = R₀
    end
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL / Vc) * Central
        R' = iNN(Central/Vc, R)[1]
    end
    @derived begin
        Outcome ~ @. Normal(R, σ)
    end
end

fpm = fit(
    model,
    trainpop,
    init_params(model),
    MAP(FOCE());
    # Some extra options to speed up the demo at the expense of a little accuracy:
    optim_options = (; iterations=300, f_tol=1e-6),
)
# Note that we only used 10 patients to train the model (unless you've tinkered with the code - something we encourage!).

pred_traindata = predict(fpm; obstimes = 0:0.1:24);
plotgrid(pred_traindata)

ins = inspect(fpm)
goodness_of_fit(ins)


# The model has succeeded in discovering the dynamical model if the individual predictions
# match the observations well for test data.
pred = predict(model, testpop, coef(fpm); obstimes = 0:0.1:24);
plotgrid(pred; ylabel="Outcome (Test data)")



# If we discovered the actual relationship between drug and response then we should be able
# to use the model on patients under a dosing regimen that the model was never fitted on.
# Let's see what happends if we supply three low doses, wait a while and then supply two
# high doses.
dr2 = DosageRegimen(0.3, ii=3, addl=2)
dr3 = DosageRegimen(1.5, time=25, ii=8, addl=1)
testpop2 = synthetic_data(datamodel, DosageRegimen(dr2, dr3), p_data; nsubj = 12, obstimes=0:2:48)
pred2 = predict(model, testpop2, coef(fpm); obstimes = 0:0.01:48);
plotgrid(pred2)

# We can overlay the data-generating model ipreds of this out-of-sample data
pred_truth = predict(datamodel, testpop2, p_data; obstimes = 0:0.01:48);
plotgrid!(pred_truth; pred=false, ipred=(; color=Cycled(3), label="DataModel ipred"))


#=
Exercises:

Explore freely, but if you want some ideas for what you can look at then here's a list

- How many subjects do you need for training? Re-train with different numbers of training
  subjects and see how the model performs on test data. Can you make the model better than 
  it is here? Can you break it?
  
- How noise-sensitive is this? Increase the noisiness (σ) in your training and
  test data and re-fit. Can you compensate for noise with a larger training
  population?

- Is there some out-of-sample dose regimen that the model fails for? Why?
  
- Rewrite the model to include more knowledge. Flesh out the indirect response model and let
  the NN capture only what's called EFF in the data generating model. You can stop using R as
  an input but you'll need to change the MLPDomain definition for that.
  
- Change the number of random effects that's passed to the neural network. What happens if
  the DeepNLME model has fewer random effects than the data generating model? What happens if
  it has more?

=#

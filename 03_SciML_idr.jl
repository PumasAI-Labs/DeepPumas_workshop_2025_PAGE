using DeepPumas
using CairoMakie
using StableRNGs
set_theme!(deep_dark())

# 
# TABLE OF CONTENTS
# 
# 1. INTRODUCTION
#
# 1.1. Simulate subjects A and B with different dosage regimens
# 1.2. A dummy neural network for modeling dynamics
# 
# 2. IDENTIFICATION OF MODEL DYNAMICS USING NEURAL NETWORKS
#
# 2.1. Delegate the identification of dynamics to a neural network
# 2.2. Combine existing domain knowledge and a neural network
# 2.3. Extend the analysis to a population of multiple subjects
# 2.4. Analyse the effect of very sparse data on the predictions
#

# 
# 1. INTRODUCTION
#
# 1.1. Simulate subjects A and B with different dosage regimens
# 1.2. A dummy neural network for modeling dynamics
# 

"""
Helper Pumas model to generate synthetic data. It assumes 
one compartment non-linear elimination and oral dosing.
"""
data_model = @model begin
  @param begin
    tvKa ∈ RealDomain()
    tvCL ∈ RealDomain()
    tvVc ∈ RealDomain()
    tvSmax ∈ RealDomain()
    tvn ∈ RealDomain()
    tvSC50 ∈ RealDomain()
    tvKout ∈ RealDomain()
    tvKin ∈ RealDomain()
    σ ∈ RealDomain()
  end
  @pre begin
    Smax = tvSmax
    SC50 = tvSC50
    Ka = tvKa
    Vc = tvVc
    Kout = tvKout
    Kin = tvKin
    CL = tvCL
    n = tvn
  end
  @init begin
    R = Kin / Kout
  end
  @vars begin
    cp = max(Central / Vc, 0.0)
    EFF = Smax * cp^n / (SC50^n + cp^n)
  end
  @dynamics begin
    Depot' = -Ka * Depot
    Central' = Ka * Depot - (CL / Vc) * Central
    R' = Kin * (1 + EFF) - Kout * R
  end
  @derived begin
    Outcome ~ @. Normal(R, abs(R) * σ)
  end
end

true_parameters = (;
  tvKa=0.5,
  tvCL=1.0,
  tvVc=1.0,
  tvSmax=2.9,
  tvn=1.5,
  tvSC50=0.05,
  tvKout=2.2,
  tvKin=0.8,
  σ=0.02                         ## <-- tune the observational noise of the data here
)

# 1.1. Simulate subjects A and B with different dosage regimens
data_a = synthetic_data(
  data_model,
  DosageRegimen(1.0),
  true_parameters;
  nsubj=1,
  obstimes=0:0.5:15,
  rng=StableRNG(1)
)

data_b = synthetic_data(
  data_model,
  DosageRegimen(0.2),
  true_parameters;
  nsubj=1,
  obstimes=0:0.5:10,
  rng=StableRNG(2)
)

plotgrid(data_a; data=(; label="Data (subject A)"))
plotgrid!(data_b; data=(; label="Data (subject B)"), color=:gray)

pred_datamodel_a = predict(data_model, data_a, true_parameters; obstimes=0:0.01:10)
pred_datamodel_b = predict(data_model, data_b, true_parameters; obstimes=0:0.01:10)
plotgrid(pred_datamodel_a; ipred=false)
plotgrid!(pred_datamodel_b; data=true, ipred=false)


# 1.2. A non-dynamic machine learning model for later comparison.

```
    time_model
    
A machine learning model mapping time to a noisy outcome. This is not a SciML model.
```
time_model = @model begin
  @param begin
    mlp ∈ MLPDomain(1, 6, 6, (1, identity); reg=L1(1.0; output=false))
    σ ∈ RealDomain(; lower=0.0)
  end
  @derived begin
    nn_output := first.(mlp.(t))
    # unpacking that call:
    # t       - a vector of all time points for which a subject had observations. 
    # mlp.(t) - apply the mlp on each element of t. (the . "broadcasts" the function over all elements instead of using the vector directly) 
    # first   - get the first element of the mlp output (the output is a 1-element vector)
    Outcome ~ @. Normal(nn_output, σ)
  end
end

# Strip the dose out of the subject since this simple model does not know what to do with a dose.
data_a_no_dose = read_pumas(DataFrame(data_a); observations=[:Outcome], event_data=false)
data_b_no_dose = read_pumas(DataFrame(data_b); observations=[:Outcome], event_data=false)

fpm_time = fit(time_model, data_a_no_dose, init_params(time_model), MAP(NaivePooled()))

pred_a = predict(fpm_time; obstimes=0:0.1:10);
plotgrid(
  pred_a;
  pred=(; label="Pred (subject A)"),
  data=(; label="Data (subject A)", color=:gray),
  ipred=false
)

pred_b = predict(time_model, data_b_no_dose, coef(fpm_time); obstimes=0:0.1:10);
plotgrid!(
  pred_b,
  pred=(; label="Pred (subject B)", color=:red),
  data=(; label="Data (subject A)", color=:gray),
  ipred=false,
)

# 
# 2. IDENTIFICATION OF MODEL DYNAMICS USING NEURAL NETWORKS
#
# 2.1. Delegate the identification of dynamics to a neural network
# 2.2. Combine existing domain knowledge and a neural network
# 2.3. Extend the analysis to a population of multiple subjects
# 2.4. Analyse the effect of very sparse data on the predictions
#

# 2.1. Delegate the identification of dynamics to a neural network
neural_ode_model = @model begin
  @param begin
    mlp ∈ MLPDomain(3, 8, 8, (3, identity); reg=L1(1.0; output=false))    # neural network with 2 inputs and 1 output
    tvR₀ ∈ RealDomain(; lower=0)
    σ ∈ RealDomain(; lower=0)                       # residual error
  end
  @pre begin
    mlp_ = mlp 
    R₀ = tvR₀
  end
  @init R = R₀
  @dynamics begin
    Depot' = mlp_(Depot, Central, R)[1]
    Central' = mlp_(Depot, Central, R)[2]
    R' = mlp_(Depot, Central, R)[3]
  end
  @derived begin
    Outcome ~ @. Normal(R, abs(R) * σ)
  end
end

fpm_node = fit(neural_ode_model, data_a, sample_params(neural_ode_model), MAP(NaivePooled()))

pred_a = predict(fpm_node; obstimes=0:0.01:15)
plotgrid(
  pred_a;
  pred=(; label="Pred (subject A)"),
  ipred=false,
  data=(; label="Data (subject A)", color=:gray)
)

pred_b = predict(fpm_node, data_b; obstimes=0:0.01:15)
plotgrid!(
  pred_b,
  pred=(; label="Pred (subject B)", color=:red),
  data=(; label="Data (subject B)", color=:gray),
  ipred=false,
)

# You can get pretty good results here but the generalization performance is rather brittle.
# Try changing the the parameters from init_params (deterministic) to sample_params (random
# and anything goes) and fit again a few times. How well do you fit subject A? And how well
# do you fit subject B?


# Let's encode some more knowledge, leaving less for the neural network to pick up.

ude_model = @model begin
  @param begin
    mlp ∈ MLPDomain(2, 6, 6, (1, identity); reg=L1(1.0))    # neural network with 2 inputs and 1 output
    tvKa ∈ RealDomain(; lower=0)                    # typical value of absorption rate constant
    tvCL ∈ RealDomain(; lower=0)
    tvVc ∈ RealDomain(; lower=0)
    tvR₀ ∈ RealDomain(; lower=0)
    σ ∈ RealDomain(; lower=0)                       # residual error
  end
  @pre begin
    mlp_ = only ∘ mlp # equivalent to (args...) -> only(mlp(args...))
    CL = tvCL
    Vc = tvVc
    Ka = tvKa
    R₀ = tvR₀
  end
  @init R = R₀
  @dynamics begin
    Depot' = -Ka * Depot                                # known
    Central' = Ka * Depot - (CL / Vc) * Central
    R' = mlp_(Central / Vc, R)
  end
  @derived begin
    Outcome ~ @. Normal(R, abs(R) * σ)
  end
end

fpm_ude = fit(ude_model, data_a, init_params(ude_model), MAP(NaivePooled()))

pred_a = predict(fpm_ude; obstimes=0:0.1:15);
plotgrid(
  pred_a;
  pred=(; label="Pred (subject A)"),
  ipred=false,
  data=(; label="Data (subject A)", color=:gray),
)

pred_b = predict(ude_model, data_b, coef(fpm_ude); obstimes=0:0.1:10);
plotgrid!(
  pred_b,
  pred=(; label="Pred (subject B)", color=:red),
  data=(; label="Data (subject B)", color=:gray),
  ipred=false,
)


# 2.2. Combine existing domain knowledge and a neural network


ude_model_knowledge = @model begin
  @param begin
    mlp ∈ MLPDomain(1, 6, 6, (1, identity); reg=L1(1))    # neural network with 2 inputs and 1 output
    tvKa ∈ RealDomain(; lower=0)                    # typical value of absorption rate constant
    tvCL ∈ RealDomain(; lower=0)
    tvVc ∈ RealDomain(; lower=0)
    tvKout ∈ RealDomain(; lower=0)
    tvKin ∈ RealDomain(; lower=0)
    σ ∈ RealDomain(; lower=0)                       # residual error
  end
  @pre begin
    mlp_ = only ∘ mlp
    CL = tvCL
    Vc = tvVc
    Ka = tvKa
    Kin = tvKin
    Kout = tvKout
  end
  @init R = Kin / Kout
  @dynamics begin
    Depot' = -Ka * Depot
    Central' = Ka * Depot - (CL / Vc) * Central
    R' = Kin * (1 + mlp_(Central / Vc)) - Kout * R
  end
  @derived begin
    Outcome ~ @. Normal(R, abs(R) * σ)
  end
end

fpm_knowledge = fit(
  ude_model_knowledge,
  data_a,
  # sample_params(ude_model_knowledge),
  init_params(ude_model_knowledge),
  MAP(NaivePooled());
  # optim_options = (; iterations = 10_000, show_every=100)
)

pred_a = predict(fpm_knowledge; obstimes=0:0.1:15);
plotgrid(
  pred_a;
  ipred=false,
  data=(; label="Data (subject a)", color=(:black, 0.5)),
  pred=(; label="Pred (subject a)"),
  legend=(; orientation=:horizontal, nbanks=2)
)

pred_b = predict(ude_model_knowledge, data_b, coef(fpm_knowledge); obstimes=0:0.1:10);
plotgrid!(
  pred_b;
  ipred=false,
  data=(; label="Data (subject b)", color=:black),
  pred=(; label="Pred (subject b)", color=:red)
)

plotgrid!(pred_datamodel_a; pred=(; color=(:black, 0.4), label="Datamodel"), ipred=false)
plotgrid!(pred_datamodel_b; pred=(; color=(:black, 0.4), label="Datamodel"), ipred=false)



# How did we do? Did the encoding of further knowledge (conservation of drug
# between Depot and Central) make the model better?

# 2.3. Extend the analysis to a population of multiple, heterogeneous, subjects
#

data_model_heterogeneous = @model begin
  @param begin
    tvImax ∈ RealDomain(; lower=0.0)  # typical value of maximum inhibition
    tvIC50 ∈ RealDomain(; lower=0.0)  # typical value of concentration for half-way inhibition
    tvKa ∈ RealDomain(; lower=0.0)    # typical value of absorption rate constant
    tvVc ∈ RealDomain(; lower=0)
    σ ∈ RealDomain(; lower=0.0)       # residual error
  end
  @random η ~ MvNormal(Diagonal([0.05, 0.05, 0.05]))
  @pre begin
    Imax = tvImax * exp(η[1])
    IC50 = tvIC50 * exp(η[2])
    Vc = tvVc
    Ka = tvKa * exp(η[2])
  end
  @vars begin
    cp := Central / Vc
  end
  @dynamics begin
    Depot' = -Ka * Depot
    Central' = Ka * Depot - Imax * cp / (IC50 + cp)
  end
  @derived begin
    Outcome ~ @. Normal(cp, cp * σ)
  end
end

data_model_heterogeneous = @model begin
  @param begin
    tvKa ∈ RealDomain()
    tvCL ∈ RealDomain()
    tvVc ∈ RealDomain()
    tvSmax ∈ RealDomain()
    tvn ∈ RealDomain()
    tvSC50 ∈ RealDomain()
    tvKout ∈ RealDomain()
    tvKin ∈ RealDomain()
    σ ∈ RealDomain()
  end
  @random begin
    η ~ MvNormal(5, 0.3)
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
    cp = max(Central / Vc, 0.0)
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


# 2.4. Analyse the effect of very sparse data on the predictions

sims_sparse = [
  simobs(
    data_model_heterogeneous,
    Subject(; events=DosageRegimen(1.0), id=i),
    true_parameters;
    obstimes=11 .* sort!(rand(2))
  ) for i = 1:25
]
population_sparse = Subject.(sims_sparse)
plotgrid(population_sparse)

fpm_sparse = fit(
  ude_model_knowledge,
  population_sparse,
  init_params(ude_model_knowledge),
  MAP(NaivePooled()),
)

pred = predict(fpm_sparse; obstimes=0:0.01:10);
plotgrid(pred)

# plot them all stacked ontop of oneanother
fig = Figure();
ax = Axis(fig[1, 1]; xlabel="Time", ylabel="Outcome", title="Stacked predictions")
for i in eachindex(pred)
  plotgrid!([ax], pred[i:i]; data=(; color=Cycled(i)))
end
fig

# Does it look like we've found anything reasonable?


# 2.5. Finally, what if we have multiple patients with fairly rich timecourses?

population = synthetic_data(
  data_model_heterogeneous,
  DosageRegimen(1.0),
  true_parameters; obstimes=0:1:10,
  nsubj=25,
  rng=StableRNG(1)
)
plotgrid(population)

fpm_knowledge_2 = fit(
  ude_model_knowledge,
  population,
  init_params(ude_model_knowledge),
  MAP(NaivePooled()),
)

pred = predict(fpm_knowledge_2; obstimes=0:0.1:10);
plotgrid(pred)




############################################################################################
#                                   2.6. Bonus material                                    #
############################################################################################

# The examples above illustrate the core concepts that we want to teach. However, they're a
# bit cleaner than one might expect in real life and they avoid some of the issues that a
# modeler may face when using UDEs/NeuralODEs.
# Here, we go through some of the problems one is likely to face when using UDEs in real
# projects and how to think when trying to solve them.

# 2.6.1. Akward scales 

# Most neural networks work best if the input and target outputs have value that are not too
# far from the relevant bits of our activation functions. A farily standard practice in ML
# regression is to standardize input and output to have a mean 0 and std=1 or to ensure that
# all values are between 0 and 1. With bad input/output scales, it can be hard to fit a
# model. 

scaling = 1e4
parameters_scaled = (;
  tvKa=0.5,
  tvCL=1.0,
  tvVc=1.0,
  tvSmax=1.9,
  tvn=1.5,
  tvSC50=0.05 * scaling,
  tvKout=2.2,
  tvKin=0.8 * scaling,
  σ=0.02 
)

data_hard_scale = synthetic_data(
  data_model,
  DosageRegimen(scaling),
  parameters_scaled;
  nsubj=1,
  obstimes=0:0.5:15,
  rng=StableRNG(1)
)

plotgrid(data_hard_scale)

fpm_hard_scale = fit(
  ude_model_knowledge,
  data_hard_scale,
  # init_params(ude_model_knowledge),
  sample_params(ude_model_knowledge),
  MAP(NaivePooled())
)

pred_hard_scale = predict(fpm_hard_scale; obstimes=0:0.1:10)
plotgrid(pred_hard_scale)

## Why did that fail so miserably? 
# For "Outcome" to be ≈ 1000, we'd need Central to be ≈ 1000 and these values will be
# inputs to our neural network.

# But our activation function, tanh, saturates for values much larger than 1.
x = -5:0.1:5
lines(x, tanh.(x))

using ForwardDiff: derivative
derivative(tanh, 0.0)
derivative(tanh, 1.0)
derivative(tanh, 10.0)
derivative(tanh, 100.0) # the gradient vanishes at large input.

lines(x, map(_x -> derivative(tanh, _x), x); axis=(; ylabel="Derivative", xlabel="input"))

## So, what'll happen in the first layer of the neural network?
w = rand(1, 6)
b = rand(6)
input = [1.0]
tanh.(w' * input .+ b)

input_large = 1e2
tanh.(w' * input_large .+ b) # All saturated, almost no gradient, no chance for the optimiser to work.

## So, what's the solution? Abandon tanh?

softplus(1e4)
derivative(softplus, 1e3)

# Looks fine? Here, we don't saturate and we have a non-zero gradient.
# We can try:

model_softplus = @model begin
  @param begin
    mlp ∈ MLPDomain(1, 6, 6, (1, identity, false); reg=L1(1), act=softplus)
    tvKa ∈ RealDomain(; lower=0)                    # typical value of absorption rate constant
    tvCL ∈ RealDomain(; lower=0)
    tvVc ∈ RealDomain(; lower=0)
    tvKout ∈ RealDomain(; lower=0)
    tvKin ∈ RealDomain(; lower=0)
    σ ∈ RealDomain(; lower=0)                       # residual error
  end
  @pre begin
    mlp_ = only ∘ mlp
    CL = tvCL
    Vc = tvVc
    Ka = tvKa
    Kin = tvKin
    Kout = tvKout
  end
  @init R = Kin / Kout
  @dynamics begin
    Depot' = -Ka * Depot
    Central' = Ka * Depot - (CL / Vc) * Central
    R' = Kin * (1 + mlp_(Central / Vc)) - Kout * R
  end
  @derived begin
    Outcome ~ @. Normal(R, σ)
  end
end

fpm_softplus = fit(
  model_softplus,
  data_hard_scale,
  init_params(model_softplus),
  MAP(NaivePooled());
)

plotgrid(predict(fpm_softplus; obstimes=0:0.1:10))

## looks better, right? But it's completely linear at the relevant scales of input values.
#It's still not good.

nn = only ∘ coef(fpm_softplus).mlp
lines(1:100:10_000, nn.(1:100:10_000))


# With UDEs/NeuralODEs, we don't always know exactly what input values the NN will recieve,
# but we can often figure out which order of magnitude they'll have. If we can rescale the
# NN inputs and outputs to be close to 1 then we would be in a much better place. In this
# case, we know that we're dosing with 1e4 and that there's concervation from Depot to
# Central. 


model_softplus_rescale = @model begin
  @param begin
    mlp ∈ MLPDomain(1, 6, 6, (1, identity); reg=L1(1), act=softplus)
    tvKa ∈ RealDomain(; lower=0) 
    tvCL ∈ RealDomain(; lower=0)
    tvVc ∈ RealDomain(; lower=0)
    tvKout ∈ RealDomain(; lower=0)
    tvKin ∈ RealDomain(; lower=0, init=1e3)
    σ ∈ RealDomain(; lower=0)
  end
  @pre begin
    mlp_ = only ∘ mlp
    CL = tvCL
    Vc = tvVc
    Ka = tvKa
    Kin = tvKin
    Kout = tvKout
  end
  @init R = Kin / Kout
  @dynamics begin
    Depot' = -Ka * Depot
    Central' = Ka * Depot - (CL / Vc) * Central
    R' = Kin * (1 + mlp_((Central / Vc) / 1e4)) - Kout * R
  end
  @derived begin
    Outcome ~ @. Normal(R, abs(R) * σ)
  end
end


fpm_rescale = fit(
  model_softplus_rescale,
  data_hard_scale,
  init_params(model_softplus_rescale),
  MAP(NaivePooled())
)

plotgrid(predict(fpm_rescale; obstimes=0:0.1:10))


# Now when we've rescaled like this, our switch to softplus became unnecessary, try
# switching back.


# So, be mindful of what scales you expect your nerual network to get as inputs and to need
#to get as outputs. Also, be mindful of how the regularization may be penalizing automatic
#rescaling of the input/output layer. Here, we looked at large inputs which could have been
#solved by the weights of the first neural network being small but where the later need to
#up-scale in the output layer would be penalized by the regularization. For inputs much
#smaller than 1, we get that the necessary large weights of the input layer may be
#over-regularized. It often makes sense not to regularize the input or output layer of the
#neural network. That avoids this particular problem but it does not always make it easy to
#find the soultion since initial gradients may be close to zero and the optimizer won't know
#what to do.


# 2.6.2 why deviate from the ML standard of using the relu activation function? Why tanh!?


model_relu = @model begin
  @param begin
    mlp ∈ MLPDomain(1, 6, 6, (1, identity); act=relu, reg=L2(1; output=false))
    tvKa ∈ RealDomain(; lower=0.0)                          # typical value of absorption rate constant
    σ ∈ RealDomain(; lower=0.0)                             # residual error
  end
  @pre begin
    mlp_ = only ∘ mlp
    Ka = tvKa
  end
  @dynamics begin
    Depot' = -Ka * Depot                                # known
    Central' = Ka * Depot - mlp_(Central)               # knowledge of conservation added
  end
  @derived begin
    Outcome ~ @. Normal(Central, σ)
  end
end



fpm_relu = fit(
  model_relu,
  data_a,
  init_params(model_relu),
  MAP(NaivePooled());
  checkidentification=false # needed because some parameter gradients will be zero with relu
)

plotgrid(predict(fpm_relu; obstimes=0:0.1:10))

# well, that was less bad than ideal for illustrating why not to use relu... But, the
# problem is that you'll find a piecewise linear function


nn_relu = only ∘ coef(fpm_relu).mlp
x = 0:0.001:0.3 # the range of Central, just look at the y axis of the plot to see this.
lines(x, nn_relu.(x); axis=(; ylabel="Central elimination", xlabel="Central"))


# With large neural networks, such piecewise linear functions end up being pretty
# expressive. But for our relatively small networks they tend to lead to bad artefacts and
# they can be hard to fit. I just avoid relu in any small networks.




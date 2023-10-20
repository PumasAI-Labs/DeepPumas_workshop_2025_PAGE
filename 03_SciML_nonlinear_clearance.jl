using DeepPumas
using CairoMakie
using StableRNGs
set_theme!(deep_light())

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
    tvImax ∈ RealDomain(; lower=0)  # typical value of maximum inhibition
    tvIC50 ∈ RealDomain(; lower=0)  # typical value of concentration for half-way inhibition
    tvKa ∈ RealDomain(; lower=0)    # typical value of absorption rate constant
    tvVc ∈ RealDomain(; lower=0)
    σ ∈ RealDomain(; lower=0)       # residual error
  end
  @pre begin
    Imax = tvImax                       # per subject value = typical value,
    IC50 = tvIC50                       # that is, no subject deviations, or,
    Vc = tvVc
    Ka = tvKa                           # in other words, no random effects
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

true_parameters = (; tvImax=0.2, tvIC50=0.08, tvKa=0.3, σ=0.05, tvVc=0.8)

# 1.1. Simulate subjects A and B with different dosage regimens

data_a = synthetic_data(
  data_model,
  DosageRegimen(1.0),
  true_parameters;
  nsubj=1,
  obstimes=0:1:10,
  rng=StableRNG(1)
)
data_b = synthetic_data(
  data_model,
  DosageRegimen(0.5),
  true_parameters;
  nsubj=1,
  obstimes=0:1:10,
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
    mlp ∈ MLPDomain(1, 6, 6, (1, identity); reg=L1(1.; output=false))
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
  ipred=false,
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

ude_model = @model begin
  @param begin
    mlp ∈ MLPDomain(2, 6, 6, (1, identity); reg=L2(1.; output=false))    # neural network with 2 inputs and 1 output
    tvKa ∈ RealDomain(; lower=0.0)                    # typical value of absorption rate constant
    σ ∈ RealDomain(; lower=0.0)                       # residual error
  end
  @pre begin
    mlp_ = only ∘ mlp # equivalent to (args...) -> only(mlp(args...))
    Ka = tvKa
  end
  @dynamics begin
    Depot' = -Ka * Depot                                # known
    Central' = mlp_(Depot, Central)                     # left as function of `Depot` and `Central`
  end
  @derived begin
    Outcome ~ @. Normal(Central, σ)
  end
end

fpm_ude = fit(ude_model, data_a, init_params(ude_model), MAP(NaivePooled()))

pred_a = predict(fpm_ude; obstimes=0:0.1:10);
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
    mlp ∈ MLPDomain(1, 6, 6, (1, identity); reg=L2(1; output=false))    # neural network with 1 inputs and 1 output
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

fpm_knowledge = fit(
  ude_model_knowledge,
  data_a,
  init_params(ude_model_knowledge),
  MAP(NaivePooled());
)

pred_a = predict(fpm_knowledge; obstimes=0:0.1:10);
plotgrid(
  pred_a;
  ipred=false,
  data=(; label="Data (subject a)", color=(:black, 0.5)),
  pred=(; label="Pred (subject a)"),
  legend =(; orientation=:horizontal, nbanks=2)
)

pred_b = predict(ude_model_knowledge, data_b, coef(fpm_knowledge); obstimes=0:0.1:10);
plotgrid!(
  pred_b;
  ipred=false,
  data=(; label="Data (subject b)", color=:black),
  pred=(; label="Pred (subject b)", color=:red)
)

plotgrid!(pred_datamodel_a; pred=(;  color=(:black, 0.4), label = "Datamodel"), ipred=false)
plotgrid!(pred_datamodel_b; pred=(;  color=(:black, 0.4), label = "Datamodel"), ipred=false)



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
  rng=StableRNG(1),
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
parameters_scaled = (; tvImax=0.2 * scaling, tvIC50=0.08 * scaling, tvKa=0.3, σ=0.05, tvVc=0.8)

data_hard_scale = synthetic_data(
  data_model,
  DosageRegimen(scaling),
  parameters_scaled;
  nsubj=1,
  obstimes=0:1:10,
  rng=StableRNG(1)
)

plotgrid(data_hard_scale)

fpm_hard_scale = fit(
  ude_model_knowledge,
  data_hard_scale,
  init_params(ude_model_knowledge),
  MAP(NaivePooled())
)

pred_hard_scale = predict(fpm_hard_scale)
plotgrid(pred_hard_scale)

## Why did that fail so miserably? 
# For "Outcome" to be ≈ 1000, we'd need Central to be ≈ 1000 and these values will be
# inputs to our neural network.

# But our activation function, tanh, saturates for values much larger than 1.
x = -5:0.1:5
lines(x, tanh.(x))

using ForwardDiff: derivative
derivative(tanh, 0.)
derivative(tanh, 1.)
derivative(tanh, 10.)
derivative(tanh, 1e4) # the gradient vanishes at large input.

lines(x, map(_x -> derivative(tanh, _x), x); axis=(; ylabel="Derivative", xlabel="input"))

## So, what'll happen in the first layer of the neural network?
w = rand(1,6)
b = rand(6)
input = [1.]
tanh.(w' * input .+ b)

input_large = 1e4
tanh.(w' * input_large .+ b) # All saturated, no gradient, no chance for the optimiser to work.

## So, what's the solution? Abandon tanh?

softplus(1e4)
derivative(softplus, 1e4)

# Looks fine? Here, we don't saturate and we have a non-zero gradient.
# We can try:

model_softplus = @model begin
  @param begin
    mlp ∈ MLPDomain(1, 6, 6, (1, identity); act = softplus, reg=L2(0.1))    # neural network with 1 inputs and 1 output
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

fpm_softplus = fit(
  model_softplus,
  data_hard_scale,
  init_params(model_softplus),
  MAP(NaivePooled())
)

plotgrid(predict(fpm_softplus; obstimes=0:0.1:10))

## looks better, right? But it's completely linear at the relevant scales of input values
#(Central needs to be ~ 1000 to get the right outcomes). The model misses the non-linear
#clearance.

nn = only ∘ coef(fpm_softplus).mlp
lines(1:1000, nn.(1:1000))


## So why did the non-linear softplus model turn linear? One reason is that the mlp output
#should be on the scale of 1e3 but we're regularizing the parameter values so that scale is
#best achieved by just passing along the input without ever scaling it down towards the
#non-linear part of softplus

x = -2:0.01:2
lines(x, softplus.(x)) # non-linearity brings expressivity to our neural network
x = 0:1000 # The scale of Central values in our data
lines(x, softplus.(x)) # now we're just linear

# we could have avoided regularizing the output layer of the neurual network and hoped that
# the fit found that the first layer should scale down the inputs and the last layer should
# scale up the output. You can try this ( 
# mlp ∈ MLPDomain(1, 6, 6, (1, identity); act=softplus, reg = L1(0.1; input=false, output=false))
# ). 
# Then, we'd not penalize that solution and it _should_ work, but it's hard to get it to fit
# well because the gradient towards the nonlinear parts of the neural network is so small.


# A better approach is to know your scales and rescale towards values between [-2, 2] or [0,
# 1] or something like that.

maximum(mapreduce(vcat, data_hard_scale) do subject
  subject.observations.Outcome
end)

maximum(data_hard_scale) do subject
  subject.time[end]
end

# Our largest observation of "Outcome" is ~2500. The error model is designed such that this
# should also be the scale of Central. The time range of our data is 0 to 10 and at t=10 we
# should have degraded most of the drug. This means that we need to enable an average
# degradation rate of ~ 2500/10 and one could suspect that the peak degradation rate is even
# a bit higher than that. So, if we rescaled the input and output of the neural network
# accordingly then the neural network itself would deal with values closer to 1 which is
# what it does best.


model_softplus_rescale = @model begin
  @param begin
    mlp ∈ MLPDomain(1, 6, 6, (1, identity); act = softplus, reg=L2(0.1))
    tvKa ∈ RealDomain(; lower=0.0)
    σ ∈ RealDomain(; lower=0.0)
  end
  @pre begin
    mlp_ = only ∘ mlp
    Ka = tvKa
  end
  @dynamics begin
    Depot' = -Ka * Depot
    Central' = Ka * Depot - mlp_(Central/2500)*250
  end
  @derived begin
    Outcome ~ @. Normal(Central, σ)
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

# Instead of hard-coding these constants into our models, we could also have applied this
# scaling in the data wrangling process. Rescaling outcomes and doses such that we expect
# the dynamic variables to range from around 0 to 1 and rescaling time to [0, 1] is usually
# sufficient to make fitting much easier. You can then apply inverse transforms during
# plotting, etc. In this particular case, we have some conservation between the dose and our
# measurement so scale them with the same scaling constant! In many PD cases, the model
# would be able to rescale at will and then the dose can often just be scaled such that the
# largest dose is 1.

df = DataFrame(data_hard_scale)
yscale = 2500
tscale = 10
dose_scale = yscale
df.Outcome ./= yscale
df.time ./= tscale
df.amt ./= dose_scale

data_nice_scale = read_pumas(df; observations=[:Outcome])

fpm_rescaled_data = fit(
  model_softplus,
  data_nice_scale,
  init_params(model_softplus),
  MAP(NaivePooled())
)

# We can return to our original scale when plotting, etc, by passing the inverse transform
plotgrid(
  predict(fpm_rescaled_data; obstimes=0:0.01:1); 
  ytransform = x -> x * yscale,
  xtransform = x -> x * tscale,
)


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
    mlp ∈ MLPDomain(1, 6, 6, (1, identity); act = relu, reg=L2(1; output=false))  
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

# And why tanh? Well, we've just found it consistently pretty good but feel free to explore
# these use cases and write a paper on it. softplus is also one of our go-to activation
# functions. Note that tanh and softplus extrapolate differently for inputs larger that the
# network was trained on. tanh tends towards saturation while softplus tends towards
# linearity.



############################################################################################
## Explore how images can be used as covariates in longitudinal NLME models.
##
## Note that this is a working proof-of-concept and not representative of the more polished 
## image processing feature we want to release.
############################################################################################

using DeepPumas
using SimpleChains
using SimpleChains: relu
using Images: Gray
using CairoMakie
using Serialization
set_theme!(deep_light())

mnist = deserialize(@__DIR__() * "/mnist.jls")
num_train = 60_000
xtrain = mnist.x[:,:,:,1:num_train]
xtest = mnist.x[:,:,:,num_train+1:end]
ytrain = mnist.y[1:num_train]
ytest = mnist.y[num_train+1:end]

## Have a peek at a random image from the data set
Gray.(xtrain[:,:,1,rand(1:end)])

############################################################################################
## Generate synthetic data based on the MNIST annotations.
############################################################################################
_trainpop = [Subject(; id=i, covariates=(; n=ytrain[i], img=xtrain[:, :, :, i]), ) for i in eachindex(ytrain)]
_testpop = [Subject(; id=i, covariates=(; n=ytest[i], img=xtest[:, :, :, i])) for i in eachindex(ytest)]

_trainpop[3].covariates()
Gray.(_trainpop[3].covariates().img[:,:,1])

mnist_datamodel = @model begin
  @param begin
    σ ∈ RealDomain(; lower=0.0)
  end
  @covariates n
  @pre begin
    α = n
    r = 3 * (n + 1) / (5 + n + 1)
  end
  @init y = 5.0
  @dynamics y' = r * (α - y)
  @derived begin
    Y = @. Normal(y, σ)
  end
end

# Mimic that data is not sampled at perfect time intervals and that all patients don't have
# the same number of samples
sample_times() = vcat(0, cumsum(rand(Gamma(7, 0.2/(7-1)), rand(10:20))))
p_truth = (; σ = 0.5)

mnist_trainpop = [Subject(simobs(mnist_datamodel, subj, p_truth; obstimes= sample_times())) for subj in _trainpop]
mnist_testpop = [Subject(simobs(mnist_datamodel, subj, p_truth; obstimes= sample_times())) for subj in _testpop]

plotgrid(mnist_trainpop[1:12])

############################################################################################
## Fit an NLME model where η slurps heterogeniety
############################################################################################
ηmodel = @model begin
  @param begin
    σ ∈ RealDomain(; lower=0.0)
    tvr ∈ RealDomain(; lower=0.0)
    tvα ∈ RealDomain(; lower=0.0)
    Ω ∈ PSDDomain(2)
  end
  @random begin
    η ~ MvNormal(Ω)
  end
  @pre begin
    α = tvα + η[1]
    r = tvr * exp(η[2])
  end
  @init y = 5.0
  @dynamics y' = r * (α - y)
  @derived begin
    Y ~ @. Normal(y, σ)
  end
end

@time fpmη = fit(ηmodel, mnist_trainpop, init_params(ηmodel), FOCE()) # takes ~6 minutes

# For the impatient, load a pre-fitted model:
# fpmη = deserialize(@__DIR__() * "/fits/mnist_ηmodel_fpm.jls")
# mnist_trainpop = fpmη.data

predη = predict(ηmodel, mnist_testpop[1:25], coef(fpmη); obstimes=0:0.01:4);
plotgrid(predη; figure=(; resolution=(1200,1200)))


############################################################################################
## Train the CNN model towards our target η posterior
############################################################################################

target = preprocess(fpmη; covs=(:img,), standardize=false)

## Define a convolutional neural network - a variant of the 'LeNet-5' model.
lenet = SimpleChain(
  (static(28), static(28), static(1)),
  Conv(relu, (5, 5), 6),
  MaxPool(2, 2),
  Conv(relu, (5, 5), 16),
  MaxPool(2, 2),
  Flatten(3),
  TurboDense(relu, 120),
  TurboDense(relu, 84),
  TurboDense(identity, numoutputs(target)),
)

#=
Here, we're changing the output from being a classifier of unordered data to being a
regressor of ordered data. This comes at the cost of a little predictive accuracy but it
often makes sense in a pharmacometric scenario where we typically expect patients to be on a
continuum. It also ensures that when an image is very unclear then it's better to fall back
towards the mean prediction rather than just picking whether the scribble should be
interpreted as a 1 or a 7. 
=#


## Fit using SimpleChains machinery
lenetloss = SimpleChains.add_loss(lenet, AbsoluteLoss(target.y))
G = SimpleChains.alloc_threaded_grad(lenetloss)
p = SimpleChains.init_params(lenet)
@time SimpleChains.train_batched!(G, p, lenetloss, xtrain, SimpleChains.ADAM(3e-4), 100);

############################################################################################
## Embed the fitted CNN in an NLME model - replacing the effect that the η had before
############################################################################################


deep_model = let _p = Float64.(p), _lenet = lenet, ytrsf = target.ytrsf
  @model begin
    @param begin
      NN ∈ NeuralDomain(_lenet, nothing, _p)
      tvr ∈ RealDomain(; lower=0.0)
      tvα ∈ RealDomain(; lower=0.0)
      σ ∈ RealDomain(; lower=0.0, init=5e-1)
    end
    @covariates img
    @pre begin
      nn_pred := only(ytrsf(NN(img))).η
      α = tvα + nn_pred[1]
      r = tvr * exp(nn_pred[2])
    end
    @init y = 5.0
    @dynamics y' = r * (α - y)
    @derived begin
      Y = @. Normal(y, σ)
    end
  end
end

param = (;
  tvr = coef(fpmη).tvr,
  tvα = coef(fpmη).tvα,
  σ = coef(fpmη).σ,
  NN = init_params(deep_model).NN,
)

plotpop = sample(mnist_testpop, 25; replace=false)
pred = predict(deep_model, plotpop, param; obstimes=0:0.05:4); 

begin
  fig = plotgrid(
    pred[1:25]; 
    figure=(; resolution=(1200, 1200)),
    pred = (; label = "Baseline (t=0) prediction"),
    ipred=false,
  )
  axs = fig.content.axes
  for i in 1:length(axs)
    image!(axs[i], 1 .+[2.5, 4.], [6, 10.], permutedims(pred[i].subject.covariates().img[end:-1:1,:,1]))
  end
  fig
end



############################################################################################
## We can go even further and fit everything in concert but this is too slow to do live
############################################################################################

# fpm_deep = fit(deep_model, mnist_trainpop, param, NaivePooled())

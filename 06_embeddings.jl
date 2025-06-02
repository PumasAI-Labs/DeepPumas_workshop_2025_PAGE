using Pkg
Pkg.activate(@__DIR__())
Pkg.resolve()

using DeepPumas

using AlgebraOfGraphics
using CairoMakie
using CSV
using DataFrames
using DataFramesMeta
using DeepPumas
using Flux
using PumasPlots
using Latexify
using Random
using StatsBase
using Tables
using Transformers
using Transformers.HuggingFace
using Transformers.TextEncoders
using TSne


# Load the patient data
patient_data = CSV.read(@__DIR__() * "/data_prognostic_text.csv", DataFrame);

@rtransform! patient_data :Description = replace(:Description, "\""=>"")

pop = read_pumas(
    patient_data; 
    observations = [:yPK, :yPD],
    covariates = [:Description, :Score],
)

train_pop = pop[1:100]
test_pop = pop[101:200]

plotgrid(train_pop[1:6]; observation = :yPK)
plotgrid(train_pop[1:6]; observation = :yPD)

get_text(s::Pumas.Subject) = s.covariates().Description
get_text(train_pop[1])
get_text(train_pop[2])


### Load the Embedding Model
# Load the pre-trained embedding model from Hugging Face
loaded_model = hgf"avsolatorio/NoInstruct-small-Embedding-v0"

const encoder = loaded_model[1]
const llm = loaded_model[2]

# Define how to get a patient's embedding
get_embedding(subj::DeepPumas.Pumas.Subject) = get_embedding(subj.covariates(0).Description)
get_embedding(pop::DeepPumas.Pumas.Population) = mapreduce(get_embedding, hcat, pop)
function get_embedding(context)
    enc = encode(encoder, context)
    out = llm(enc)
    return out.pooled
end

# Get the embeddings for all patients and put it in a matrix
X_train = mapreduce(get_embedding, hcat, train_pop)
X_test = mapreduce(get_embedding, hcat, test_pop)




## t-SNE is a stochastic dimension reduction technique for visualizing spatial patterns of the embeddings. You'll get different result each time you run this.
Y = tsne(X_train', 2, 0, 10000, 8.0) # 2D t-SNE embedding of the training data
scatter(Y)


begin
    plt = scatter(Y)
    id = 1
    Makie.text!(
        -100,  # Tweak the x position
        -1000,   # Tweak the y position
        text=get_text(train_pop[id]),
        fontsize=12,
        word_wrap_width=200,
        offset=(5,5),
    )
    scatter!(Y[id, 1], Y[id, 2]; color=Cycled(2), markersize=25, strokewidth=2)
    plt
end

## We could also do a PCA but that might miss nonlinear patterns.
pca = fit(PCA, X_train, maxoutdim=2)
x_pca = predict(pca, X_train)

scatter(x_pca)

begin
    plt = scatter(x_pca)
    id = 6
    Makie.text!(
        -0.3,  # Tweak the x position
        -0.4,   # Tweak the y position
        text=get_text(train_pop[id]),
        fontsize=12,
        word_wrap_width=200,
        offset=(5,5),
    )
    s = scatter!(x_pca[1, id], x_pca[2, id]; color=Cycled(2), markersize=25, strokewidth=2)
    plt
end


## Conclusion:
# The patient "wellness" quantification is a central component to explaining between subject variability in this data set. 


# Let's go to the NLME modelling then!


embedding_df = DataFrame(id = getfield.(train_pop, :id), Embeddings_pc = eachcol(x_pca))
test_embedding_df = DataFrame(id = getfield.(test_pop, :id), Embeddings_pc = eachcol(predict(pca, X_test)))

embedding_df = DataFrame(id = getfield.(train_pop, :id), embeddings = get_embedding.(train_pop))
test_embedding_df = DataFrame(id = getfield.(test_pop, :id), embeddings = get_embedding.(test_pop))

pop_embeddings = read_pumas(
    innerjoin(patient_data, embedding_df; on=:id);
    observations = [:yPK, :yPD],
    covariates = [:Description, :Score, :embeddings],
)

test_pop_embeddings = read_pumas(
    innerjoin(patient_data, test_embedding_df; on=:id);
    observations = [:yPK, :yPD],
    covariates = [:Description, :Score, :embeddings],
)


model = @model begin
  @param begin
    NN ∈ MLPDomain(5, 6, 5, (1, identity); reg=L2(1.0))
    tvKa ∈ RealDomain(; lower=0)
    tvCL ∈ RealDomain(; lower=0)
    tvVc ∈ RealDomain(; lower=0)
    tvR₀ ∈ RealDomain(; lower=0)
    ωR₀ ∈ RealDomain(; lower=0)
    Ω ∈ PDiagDomain(2)
    Ω_nn ∈ PDiagDomain(3)
    σ ∈ RealDomain(; lower=0)
    σ_pk ∈ RealDomain(; lower=0)
  end
  @random begin
    η ~ MvNormal(Ω)
    η_nn ~ MvNormal(Ω_nn)
  end
  @pre begin
    Ka = tvKa * exp(η[1])
    Vc = tvVc * exp(η[2])
    CL = tvCL
    R₀ = tvR₀ * exp(10 * ωR₀ * η_nn[1])
    iNN = fix(NN, η_nn)
  end
  @init begin
    R = R₀
  end
  @dynamics begin
    Depot' = -Ka * Depot
    Central' = Ka * Depot - (CL / Vc) * Central
    R' = iNN(Central / Vc, R)[1]
  end
  @derived begin
    yPK ~ @. Normal(Central/Vc, σ_pk)
    yPD ~ @. Normal(R, σ)
  end
end

fpm = fit(
  model,
  pop_embeddings,
  init_params(model),
  MAP(FOCE());
  optim_options=(; iterations = 300),
  constantcoef = (; Ω_nn = I(3))
)

target = preprocess(fpm; covs = [:embeddings])
nn = MLPDomain(numinputs(target), 55, 40, (numoutputs(target), identity); backend=:simplechains, act=tanh, reg=L2(5))
fnn = fit(nn, target; optim_options = (; loss = DeepPumas.l2), training_fraction=1.0)

nn = MLPDomain(numinputs(target), 9, 9, (numoutputs(target), identity); reg=L2(10.0))

fnn = fit(nn, target; training_fraction=0.9, optim_options = (; loss = l2))

augmented_fpm = augment(fpm, fnn)

pred_embedding = simobs(fpm.model, test_pop_embeddings, coef(fpm), fnn(test_pop_embeddings))

pred_augment =
  predict(augmented_fpm.model, test_pop_embeddings, coef(augmented_fpm); obstimes=0:0.1:24);

pred_original = predict(fpm, test_pop_embeddings; obstimes = 0:0.1:24)
plotgrid(pred_original[1:6]; ipred=false, pred=(; color=(:red, 0.2), label="No covariate pred"), observation=:yPD)
plotgrid!(pred_augment[1:6]; ipred=false, pred=(; linestyle=:dash, label = "Embedding pred"), observation = :yPD)


### We actually have the data generating model here so we can compare.

datamodel = @model begin
  @param begin
      tvKa ∈ RealDomain()
      tvVc ∈ RealDomain()
      tvSmax ∈ RealDomain()
      tvSC50 ∈ RealDomain()
      tvKout ∈ RealDomain()
      Kin ∈ RealDomain()
      CL ∈ RealDomain()
      n ∈ RealDomain()
      Ω ∈ PDiagDomain(5)
      σ_pk ∈ RealDomain()
      σ_pd ∈ RealDomain()
  end
  @random η ~ MvNormal(Ω)
  @covariates Score
  @pre begin
      # s = (Score - 5) / 10
      Smax = tvSmax * exp(η[1]) + 3 * Score / 5.
      # Smax = tvSmax * exp((1-c) * η[1] + c * s) 
      # SC50 = tvSC50 * exp(η[2] + 0.3 * (Score / 5)^0.75)
      SC50 = tvSC50 * exp(η[2])
      Ka = tvKa * exp(η[3] + 0.3 * (Score/5)^2 )
      Vc = tvVc * exp(η[4])
      Kout = tvKout * exp(η[5] + 0.5 * Score/5)
  end
  @init R = Kin / Kout
  @vars begin
      cp = abs(Central / Vc)
      EFF = Smax * cp^n / (SC50^n + cp^n)
  end
  @dynamics begin
      Depot' = -Ka * Depot
      Central' = Ka * Depot - (CL / Vc) * Central
      R' = Kin * (1 + EFF) - Kout * R
  end
  @derived begin
      yPK ~ @. Normal(Central ./ Vc, σ_pk)
      yPD ~ @. Normal(R, σ_pd)
  end
end

data_params = (;
  tvKa = 0.5,
  tvVc = 1.0,
  tvSmax = 0.9,
  tvSC50 = 0.02,
  tvKout = 1.2,
  Kin = 1.2,
  CL = 1.0,
  n = 1.0,
  Ω = Diagonal(fill(1e-2, 5)),
  σ_pk = 3e-2,
  σ_pd = 1e-1,
)

pred_data = predict(datamodel, test_pop_embeddings, data_params; obstimes = 0:0.1:25)
plotgrid!(pred_data; ipred=false, pred = (; label = "DataModel pred", color=:grey), observation = :yPD)


## Convert predictions to dataframes for some custom plotting
df_pred_data = DataFrame(predict(datamodel, test_pop_embeddings, data_params))
df_pred_original = DataFrame(predict(fpm, test_pop_embeddings))
df_pred_embeddings = DataFrame(predict(augmented_fpm, test_pop_embeddings))


_df = vcat(df_pred_embeddings, df_pred_original, df_pred_data; source = :Model => [:Embedding, :Original, :DataGenerating], cols=:union)
__df = @by dropmissing(_df, :yPD) :Model :r2 = cor(:yPD, :yPD_pred).^2

begin
  spec = data(_df) * mapping(:yPD_pred => "Population prediction", :yPD=> "PD Observation")
  spec2 = data(hcat(__df, DataFrame(;x=fill(0.5,3), y=fill(3.,3)))) * mapping(:x, :y; text = :r2 => (x -> verbatim("r²: $(round(x, digits=2))" ))) * visual(Makie.Text)
  layoutspec = mapping(col=:Model=>sorter(:Original, :Embedding, :DataGenerating))
  fig = draw((spec + spec2)*layoutspec; axis = (; width=200, height=200))
  Label(fig.figure[2,:], "Predicted yPD")
  Label(fig.figure[:,0], "Observed yPD", rotation=pi/2)
  Makie.resize_to_layout!(fig.figure)
  fig
end
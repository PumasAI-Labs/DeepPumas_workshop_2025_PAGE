
using DeepPumas


using AlgebraOfGraphics
using CairoMakie
using CSV
using DataFrames
using DataFramesMeta
using DeepPumas
using Flux
using Markdown
using MultivariateStats
using PairPlots
using PrettyTables
using PumasPlots
using Pumas.Latexify
using Random
using Tables
using Transformers
using Transformers.HuggingFace
using Transformers.TextEncoders
using QuartoTools
using TSne
using QuartoTools: @cache
const AoG = AlgebraOfGraphics



using Transformers
using Transformers.HuggingFace
using Transformers.TextEncoders

using CSV
using DataFrames

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


using StatsBase


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


x_pca[:,1]


embedding_df = DataFrame(id = getfield.(:id, train_pop))

@rtransform! patient_data :Embeddings_pc = 
predict(pca, get_embedding(:Description))

pop = read_pumas(
    patient_data; 
    observations = [:yPK, :yPD],
    covariates = [:Description, :Score, :Embeddings_pc],
)


Subject(train_pop[1], covariates = (; embedding = x_pca[:, 1]))


model = @model begin
  @param begin
    NN ∈ MLPDomain(5, 6, 5, (1, identity); reg=L2(1.0))
    # tvKa ∈ RealDomain(; lower=0)
    # tvCL ∈ RealDomain(; lower=0)
    # tvVc ∈ RealDomain(; lower=0)
    # tvR₀ ∈ RealDomain(; lower=0)
    # ωR₀ ∈ RealDomain(; lower=0)
    # Ω ∈ PDiagDomain(2)
    # Ω_nn ∈ PDiagDomain(3)
    # σ ∈ RealDomain(; lower=0)
    # σ_pk ∈ RealDomain(; lower=0)
    # 
    tvKa ∈ RealDomain()
    tvCL ∈ RealDomain()
    tvVc ∈ RealDomain()
    tvR₀ ∈ RealDomain()
    ωR₀ ∈ RealDomain()
    Ω ∈ PDiagDomain(2)
    Ω_nn ∈ PDiagDomain(3)
    σ ∈ RealDomain()
    σ_pk ∈ RealDomain()
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
  trainpop_small,
  init_params(model),
  MAP(FOCE());
  optim_options=(; iterations=300),
  constantcoef = (; Ω_nn = Diagonal(fill(0.1, 3)))
)


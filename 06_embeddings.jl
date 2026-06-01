using Pkg
Pkg.activate(@__DIR__())
Pkg.resolve()

using DeepPumas

using AlgebraOfGraphics
using CairoMakie
using CSV
using DataFrames
using DataFramesMeta
using PumasPlots
using Random
using StableRNGs
using StatsBase
using Tables
using TSne
using MultivariateStats

using ONNXRunTime
using HuggingFaceTokenizers
using Downloads
using LinearAlgebra

set_mlp_backend(:simplechains)


############################################################################################
## Load the data: a PK time series (yPK) plus each patient's free-text Description and a
## wellness Score. We jump straight into the modelling — the data-generating model that
## produced data_pk.csv lives near the bottom of this script.
############################################################################################

patient_data = CSV.read(@__DIR__() * "/data_pk.csv", DataFrame)

pop = read_pumas(
    patient_data;
    observations = [:yPK],
    covariates = [:Description, :Score],
)

train_pop = pop[1:100]
test_pop = pop[101:200]
scores = [s.covariates(0.).Score for s in train_pop]   # wellness score per training subject

plotgrid(train_pop[1:6]; observation = :yPK)

get_text(s::Pumas.Subject) = s.covariates(0.).Description
get_text(train_pop[1])
get_text(train_pop[2])


### Load the embedding model: all-MiniLM-L6-v2 (384-dim sentence embeddings).
# Download the ONNX export once (cached under assets/, ~90 MB) and load it with
# the ONNXRunTime C runtime; the matching tokenizer comes from HuggingFaceTokenizers.
const EMB_REPO = "sentence-transformers/all-MiniLM-L6-v2"
const EMB_ONNX = joinpath(@__DIR__(), "assets", "all-MiniLM-L6-v2.onnx")
if !isfile(EMB_ONNX)
    mkpath(dirname(EMB_ONNX))
    Downloads.download("https://huggingface.co/$(EMB_REPO)/resolve/main/onnx/model.onnx", EMB_ONNX)
end
const emb_model = ONNXRunTime.load_inference(EMB_ONNX)
const emb_tokenizer = HuggingFaceTokenizers.from_pretrained(HuggingFaceTokenizers.Tokenizer, EMB_REPO)

# A patient's embedding: tokenize the Description, run the transformer, then
# mean-pool the token vectors and L2-normalize → one 384-vector per patient.
function get_embedding(context::AbstractString)
    ids = HuggingFaceTokenizers.encode(emb_tokenizer, context).ids
    L = length(ids)
    out = emb_model(Dict(
        "input_ids"      => reshape(Int64.(ids), 1, L),
        "attention_mask" => reshape(ones(Int64, L), 1, L),
        "token_type_ids" => reshape(zeros(Int64, L), 1, L),
    ))
    tok_emb = out["last_hidden_state"]               # (1, L, 384)
    pooled = vec(sum(@view(tok_emb[1, :, :]); dims = 1)) ./ L
    return pooled ./ norm(pooled)
end
get_embedding(subj::DeepPumas.Pumas.Subject) = get_embedding(subj.covariates(0).Description)
get_embedding(pop::DeepPumas.Pumas.Population) = mapreduce(get_embedding, hcat, pop)

# Get the embeddings for all patients and put it in a matrix
X_train = mapreduce(get_embedding, hcat, train_pop)
X_test = mapreduce(get_embedding, hcat, test_pop)


## t-SNE is a stochastic dimension reduction technique for visualizing spatial patterns of the embeddings. You'll get different result each time you run this.
Y = tsne(X_train', 2, 0, 10000, 25.0) # 2D t-SNE embedding of the training data
begin
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect = 1)
    sc = scatter!(ax, Y; color = scores)
    lims = (minimum(Y), maximum(Y))
    limits!(ax, lims, lims)  # link the x and y axes to a shared range
    Colorbar(fig[1, 2], sc; label = "Wellness score")
    fig
end

begin
    plt = scatter(Y; color = scores)
    id = 1
    Makie.text!(
        -10,  # Tweak the x position
        -50,   # Tweak the y position
        text = get_text(train_pop[id]),
        fontsize = 12,
        word_wrap_width = 200,
        offset = (5, 5),
    )
    scatter!(Y[id, 1], Y[id, 2]; color = Cycled(2), markersize = 25, strokewidth = 2)
    plt
end


## We could also do a PCA but that might miss nonlinear patterns.
pca = fit(PCA, X_train, maxoutdim = 2)
x_pca = predict(pca, X_train)

begin
    plt = scatter(x_pca, color = scores)
    id = 1
    Makie.text!(
        -0.3,  # Tweak the x position
        -0.4,   # Tweak the y position
        text = get_text(train_pop[id]),
        fontsize = 12,
        word_wrap_width = 200,
        offset = (5, 5),
    )
    scatter!(x_pca[1, id], x_pca[2, id]; color = Cycled(2), markersize = 25, strokewidth = 2)
    plt
end


## Conclusion:
# The patient "wellness" quantification is a central component to explaining between subject variability in this data set.


# Let's go to the NLME modelling then!


# Workshop shortcut for speed: project the 384-dim embeddings onto their top-10
# principal components (PCA fit on TRAIN only, then applied to test). ~95% of the
# wellness signal survives at a fraction of the width, so the covariate NN shrinks
# ~10× and the joint fit is much lighter. Standardize the scores so the NN sees
# inputs near unit scale.
pca10 = fit(PCA, X_train; maxoutdim = 10)
z_train = predict(pca10, X_train)
pc_μ, pc_σ = mean(z_train; dims = 2), std(z_train; dims = 2)
pcs_train = (z_train .- pc_μ) ./ pc_σ
pcs_test  = (predict(pca10, X_test) .- pc_μ) ./ pc_σ

embedding_df = DataFrame(
    id = getfield.(train_pop, :id),
    embeddings = get_embedding.(train_pop),
    pcs = [pcs_train[:, i] for i in axes(pcs_train, 2)],
)
test_embedding_df = DataFrame(
    id = getfield.(test_pop, :id),
    embeddings = get_embedding.(test_pop),
    pcs = [pcs_test[:, i] for i in axes(pcs_test, 2)],
)

pop_embeddings = read_pumas(
    innerjoin(patient_data, embedding_df; on = :id);
    observations = [:yPK],
    covariates = [:Description, :Score, :embeddings, :pcs],
)

test_pop_embeddings = read_pumas(
    innerjoin(patient_data, test_embedding_df; on = :id);
    observations = [:yPK],
    covariates = [:Description, :Score, :embeddings, :pcs],
)


############################################################################################
## Base model — the same PK structure, but with NO covariate. Between-subject variability
## (including the Score-driven clearance effect) is soaked up by the random effects.
############################################################################################

base_model = @model begin
  @param begin
    tvKa   ∈ RealDomain(; lower = 0, init = 1.0)
    tvVc   ∈ RealDomain(; lower = 0, init = 5.0)
    tvVmax ∈ RealDomain(; lower = 0, init = 40.0)
    tvKm   ∈ RealDomain(; lower = 0, init = 3.0)
    Ω      ∈ PDiagDomain(2)
    σ      ∈ RealDomain(; lower = 0, init = 0.3)
  end
  @random η ~ MvNormal(Ω)
  @pre begin
    Ka   = tvKa
    Vc   = tvVc   * exp(η[2])
    Vmax = tvVmax * exp(η[1])
    Km   = tvKm
  end
  @dynamics begin
    Depot'   = -Ka * Depot
    Central' =  Ka * Depot - Vmax * (Central / Vc) / (Km + Central / Vc)
  end
  @derived begin
    yPK ~ @. Normal(Central / Vc, σ)
  end
end

fpm = fit(base_model, pop_embeddings, init_params(base_model), MAP(FOCE());
          optim_options = (; iterations = 300))


############################################################################################
## Post-hoc covariate integration: fit an ML model from the embedding PCs to the random
## effects, then use it to predict η for new patients from their text alone.
############################################################################################

target = preprocess(fpm; covs = [:pcs])
nn = MLPDomain(numinputs(target), 16, (numoutputs(target), identity); reg = L2(3.0))
fnn = fit(nn, target; optim_options = (; loss = l2), training_fraction = 0.8)

# `augment` would fold `fnn` back into the model, but it re-marginalizes the whole ODE per
# subject and is slow. A quicker alternative is to feed fnn's predicted random effects 
# directly:
# augmented_fpm = augment(fpm, fnn)
pred_embedding = predict(base_model, test_pop_embeddings, coef(fpm),
                         fnn(test_pop_embeddings); obstimes = 0:0.05:8)

pred_original = predict(fpm, test_pop_embeddings; obstimes = 0:0.05:8)
plotgrid(pred_original[1:6]; ipred = false, pred = (; color = (:red, 0.3), label = "No covariate pred"))
plotgrid!(pred_embedding[1:6]; pred = false, ipred = (; linestyle = :dash, label = "Embedding pred"))


############################################################################################
## The data-generating model — and how data_pk.csv was produced.
## 1-cmpt oral PK with Michaelis–Menten (nonlinear) clearance; a single observation (yPK).
## The wellness Score adds to the η on Vmax, so clearance is what the text covariate explains.
############################################################################################

datamodel = @model begin
  @param begin
    tvKa   ∈ RealDomain(; lower = 0, init = 1.0)
    tvVc   ∈ RealDomain(; lower = 0, init = 5.0)
    tvVmax ∈ RealDomain(; lower = 0, init = 40.0)
    tvKm   ∈ RealDomain(; lower = 0, init = 3.0)
    Ω      ∈ PDiagDomain(2)
    σ      ∈ RealDomain(; lower = 0, init = 0.3)
  end
  @random η ~ MvNormal(Ω)
  @covariates Score
  @pre begin
    s    = (Score - 5) / 5
    Ka   = tvKa
    Vc   = tvVc   * exp(η[2])
    Vmax = tvVmax * exp(η[1] + 0.6 * s)     # Score adds to the η on Vmax
    Km   = tvKm
  end
  @dynamics begin
    Depot'   = -Ka * Depot
    Central' =  Ka * Depot - Vmax * (Central / Vc) / (Km + Central / Vc)
  end
  @derived begin
    yPK ~ @. Normal(Central / Vc, σ)
  end
end

data_params = (; tvKa = 1.0, tvVc = 5.0, tvVmax = 40.0, tvKm = 3.0,
                 Ω = Diagonal([0.1, 0.1]), σ = 0.3)

# data_pk.csv ships with the repo. To regenerate it (e.g. after editing the DGM above),
# delete the file and run this block, then re-run from the top.
DATA_PK = @__DIR__() * "/data_pk.csv"
if !isfile(DATA_PK)
  src = unique(CSV.read(@__DIR__() * "/data_text.csv", DataFrame), :id)[1:200, :]
  gen_subjects = [
    Subject(; id = row.id, events = DosageRegimen(100.0; cmt = :Depot),
            covariates = (; Score = row.Score, Description = row.Description))
    for row in eachrow(src)
  ]
  gen_sims = simobs(datamodel, gen_subjects, data_params;
                    obstimes = [0.25, 0.5, 1, 2, 4, 8], rng = StableRNG(1))
  CSV.write(DATA_PK, DataFrame(gen_sims))
end


############################################################################################
## Compare predictions: no-covariate baseline vs embedding-informed vs the true datamodel.
############################################################################################

pred_data = predict(datamodel, test_pop_embeddings, data_params; obstimes = 0:0.05:8)
plotgrid!(pred_data; ipred = false, pred = (; label = "DataModel pred", color = :grey))

# Each model's best prediction from covariate info: the no-covariate baseline can only use
# its population prediction (yPK_pred); the embedding model uses the fnn-predicted random
# effects (yPK_ipred); the true datamodel's population prediction already uses Score.
function _pred_df(df, col, name)
    d = dropmissing(df, [:yPK, col])
    DataFrame(yPK = d.yPK, prediction = d[!, col], Model = name)
end

_df = vcat(
    _pred_df(DataFrame(predict(fpm, test_pop_embeddings)), :yPK_pred, "Original"),
    _pred_df(DataFrame(predict(base_model, test_pop_embeddings, coef(fpm), fnn(test_pop_embeddings))),
             :yPK_ipred, "Embedding"),
    _pred_df(DataFrame(predict(datamodel, test_pop_embeddings, data_params)), :yPK_pred, "DataGenerating"),
)
r2_df = @by _df :Model :r2 = cor(:yPK, :prediction) .^ 2

begin
  spec = data(_df) * mapping(:prediction => "Prediction", :yPK => "PK observation")
  spec2 = data(hcat(r2_df, DataFrame(; x = fill(2.0, 3), y = fill(15.0, 3)))) *
          mapping(:x, :y; text = :r2 => (x -> verbatim("r²: $(round(x, digits=2))"))) * visual(Makie.Text)
  layoutspec = mapping(col = :Model => sorter("Original", "Embedding", "DataGenerating"))
  fig = draw((spec + spec2) * layoutspec; axis = (; width = 200, height = 200))
  Makie.resize_to_layout!(fig.figure)
  fig
end

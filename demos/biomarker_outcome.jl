#=
Here, we're not actually fitting or running the models, just defining them so
that we can have a look and reason around what they would do. 
=#

using DeepPumas

model_direct = @model begin
  @param begin
    NN ∈ MLPDomain(6, 10, 10, (1, identity); reg=L2(1.))
    NN_bm ∈ MLPDomain(7, 10, 10, (1, identity); reg=L2(1.))
    tvVc ∈ RealDomain(; lower=0.)
    tvCL ∈ RealDomain(; lower=0.)
    ωVc ∈ RealDomain(; lower=0.)
    Ω_bm ∈ PDiagDomain(3)
    Ω ∈ PDiagDomain(3)
    σ ∈ RealDomain(; lower=0.)
    σ_bm ∈ RealDomain(; lower=0.)
  end
  @random begin
    ηVc ~ Normal(0., ωVc)
    η_bm ~ MvNormal(Ω_bm)
    η ~ MvNormal(Ω)
  end
  @pre begin
    CL = tvCL
    Vc = tvVc * exp(ηVc)
    iNN = fix(NN, η)
    iNN_bm = fix(NN_bm, η_bm)
  end
  @dynamics begin
    Central' = - CL/Vc * Central
    latent_bm' = iNN_bm(Central, latent_bm, t)[1]
    y' = iNN(Central, latent_bm, y, t)[1]
  end
  @derived begin
    Biomarker ~ @. Normal(latent_bm, σ_bm)
    PrimaryEndpoint ~ @. Normal(y, σ)
  end
end


model_shared_random = @model begin
  @param begin
    NN ∈ MLPDomain(6, 10, 10, (1, identity); reg=L2(1.))
    NN_bm ∈ MLPDomain(7, 10, 10, (1, identity); reg=L2(1.))
    NN_merge ∈ MLPDomain(3, 5, 5, (3, identity); reg=L2(1.))
    tvVc ∈ RealDomain(; lower=0.)
    tvCL ∈ RealDomain(; lower=0.)
    ωVc ∈ RealDomain(; lower=0.)
    Ω_bm ∈ PDiagDomain(3)
    Ω ∈ PDiagDomain(3)
    σ ∈ RealDomain(; lower=0.)
    σ_bm ∈ RealDomain(; lower=0.)
  end
  @random begin
    ηVc ~ Normal(0., ωVc)
    η_bm ~ MvNormal(Ω_bm)
    η ~ MvNormal(Ω)
  end
  @pre begin
    CL = tvCL
    Vc = tvVc * exp(ηVc)
    iNN = fix(NN, η)
    iNN_bm = fix(NN_bm, η_bm .+ NN_merge(η))
  end
  @dynamics begin
    Central' = - CL/Vc * Central
    latent_bm' = iNN_bm(Central, latent_bm, t)[1]
    y' = iNN(Central, y, t)[1]
  end
  @derived begin
    Biomarker ~ @. Normal(latent_bm, σ_bm)
    PrimaryEndpoint ~ @. Normal(y, σ)
  end
end


model_covariance = @model begin
  @param begin
    NN ∈ MLPDomain(6, 10, 10, (1, identity); reg=L2(1.))
    NN_bm ∈ MLPDomain(7, 10, 10, (1, identity); reg=L2(1.))
    NN_merge ∈ MLPDomain(3, 5, 5, (3, identity); reg=L2(1.))
    tvVc ∈ RealDomain(; lower=0.)
    tvCL ∈ RealDomain(; lower=0.)
    ωVc ∈ RealDomain(; lower=0.)
    Ω ∈ PSDDomain(3+3)
    σ ∈ RealDomain(; lower=0.)
    σ_bm ∈ RealDomain(; lower=0.)
  end
  @random begin
    ηVc ~ Normal(0., ωVc)
    η ~ MvNormal(Ω)
  end
  @pre begin
    CL = tvCL
    Vc = tvVc * exp(ηVc)
    iNN = fix(NN, η[1:3])
    iNN_bm = fix(NN_bm, η[4:6])
  end
  @dynamics begin
    Central' = - CL/Vc * Central
    latent_bm' = iNN_bm(Central, latent_bm, t)[1]
    y' = iNN(Central, y, t)[1]
  end
  @derived begin
    Biomarker ~ @. Normal(latent_bm, σ_bm)
    PrimaryEndpoint ~ @. Normal(y, σ)
  end
end
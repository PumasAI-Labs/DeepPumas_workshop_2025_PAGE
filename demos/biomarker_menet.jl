using DeepPumas

bm_menet= @model begin
  @param begin
    NN ∈ MLPDomain(5, 10, 10, (1, identity); reg=L2(1.))
    Ω ∈ PDiagDomain(4)
    σ ∈ RealDomain(; lower=0.)
  end
  @random begin
    η ~ MvNormal(Ω)
  end
  @derived begin
    iNN = fix(NN, η)
    Biomarker ~ @. Normal(iNN(t), σ)
  end
end

using DeepPumas

bm_ude = @model begin
  @param begin
    NN ∈ MLPDomain(5, 10, 10, (1, identity); reg=L2(1.))
    tvVc ∈ RealDomain(; lower=0.)
    tvCL ∈ RealDomain(; lower=0.)
    ωVc ∈ RealDomain(; lower=0.)
    Ω_nn ∈ PDiagDomain(3)
    σ ∈ RealDomain(; lower=0.)
  end
  @random begin
    ηVc ~ Normal(0., ωVc)
    η_nn ~ MvNormal(Ω_nn)
  end
  @pre begin
    CL = tvCL
    Vc = tvVc * exp(ηVc)
    iNN = fix(NN, η_nn)
  end
  @dynamics begin
    Central' = - CL/Vc * Central
    latent_bm' = iNN(Central, latent_bm)[1]
  end
  @derived begin
    Biomarker ~ @. Normal(latent_bm, σ)
  end
end

bm_ude_time = @model begin
  @param begin
    NN ∈ MLPDomain(6, 10, 10, (1, identity); reg=L2(1.))
    tvVc ∈ RealDomain(; lower=0.)
    tvCL ∈ RealDomain(; lower=0.)
    ωVc ∈ RealDomain(; lower=0.)
    Ω_nn ∈ PDiagDomain(3)
    σ ∈ RealDomain(; lower=0.)
  end
  @random begin
    ηVc ~ Normal(0., ωVc)
    η_nn ~ MvNormal(Ω_nn)
  end
  @pre begin
    CL = tvCL
    Vc = tvVc * exp(ηVc)
    iNN = fix(NN, η_nn)
  end
  @dynamics begin
    Central' = - CL/Vc * Central
    latent_bm' = iNN(Central, latent_bm, t)[1]
  end
  @derived begin
    Biomarker ~ @. Normal(latent_bm, σ)
  end
end
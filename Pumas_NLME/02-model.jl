include("01-population.jl")

# Model definition
model = @model begin
    @metadata begin
        desc = "1-compartment model"
        timeu = u"hr"
    end

    @param begin
        # here we define the parameters of the model
        tvcl ∈ RealDomain(; lower = 0)
        tvvc ∈ RealDomain(; lower = 0)
        Ω ∈ PDiagDomain(2)
        σ_add ∈ RealDomain(; lower = 0)
        σ_prop ∈ RealDomain(; lower = 0)
    end

    @random begin
        # here we define random effects
        η ~ MvNormal(Ω)
    end

    @covariates AGE SEX

    @pre begin
        # pre computations and other statistical transformations
        CL = tvcl * exp(η[1])
        Vc = tvvc * exp(η[2])
    end

    @dynamics begin
        # here we define compartments and dynamics
        Central' = -CL / Vc * Central
    end

    @derived begin
        # here is where we calculate concentration and add residual error
        # tilde (~) means "distributed as"
        CONC := @. Central / Vc
        """
        Drug Concentration (ng/mL)
        """
        DV ~ @. Normal(CONC, sqrt(CONC^2 * σ_prop^2 + σ_add^2))
    end
end

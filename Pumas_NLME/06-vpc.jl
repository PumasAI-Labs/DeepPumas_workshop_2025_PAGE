include("05-compare.jl")

# You can do visual predictive checks (VPC) with any fitted Pumas model
# First, use the vpc function
vpc_1cmt = vpc(fit_1cmt)
vpc_2cmt = vpc(fit_2cmt)

# Then you can plot the vpc result with vpc_plot
vpc_plot(vpc_1cmt)
vpc_plot(vpc_2cmt)

# You can also use a prediction-corrected VPC with the keyword argument prediction_correction
vpc_1cmt_pc = vpc(fit_1cmt; prediction_correction = true)
vpc_2cmt_pc = vpc(fit_2cmt; prediction_correction = true)
vpc_plot(vpc_1cmt_pc)
vpc_plot(vpc_2cmt_pc)

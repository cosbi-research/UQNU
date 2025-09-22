cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

gr()

include("../../ConfidenceEllipse.jl")
using .ConfidenceEllipse

threshold = 0.01

#set the seed
seed = 0
rng = StableRNG(seed)

in_dim = 2
out_dim = 2

activation_function_fun = gelu
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=0.01)

ensembles = []
for i in 1:10
    maximized_ensemble_folder = "../../results_maximized/lorenz/result_lorenz$i/results.jld"
    if !isfile(maximized_ensemble_folder)
        continue
    end
    tmp_results = deserialize(maximized_ensemble_folder)
    tmp_ensemble = tmp_results.naive_ensemble_reference
    push!(ensembles, tmp_ensemble)
end

σ_values = vcat([model.σ for ens in ensembles for model in ens]...)
r_values = vcat([model.r for ens in ensembles for model in ens]...)

#initialization bounds 
original_σ = 10
original_r = 28

σ_bounds = (original_σ * 0.5, original_σ * 2.0)    
r_bounds = (original_r * 0.5, original_r * 2.0)

variation_coefficient_σ = std(σ_values) / mean(σ_values)
variation_coefficient_r = std(r_values) / mean(r_values) 

using StatsPlots

#plot an histogram of sigma values
binwidth = (σ_bounds[2] - σ_bounds[1])/20
StatsPlots.histogram(σ_values, bins=σ_bounds[1]:binwidth:σ_bounds[2], title="", xlabel="σ", ylabel="Frequency", legend=false, dpi=400)
#vline for the gound truth value
vline!([original_σ], color=:red, label="", linewidth=2)
#dashed vline for the bounds
vline!([σ_bounds[1], σ_bounds[2]], color=:black, linestyle=:dash, label="", linewidth=1)
Plots.savefig("histogram_σ_values.sensitivity_matrix_first_trajectory.svg")
#plot an histogram of r values
binwidth = (r_bounds[2] - r_bounds[1])/20
StatsPlots.histogram(r_values, bins=r_bounds[1]:binwidth:r_bounds[2], title="", xlabel="r", ylabel="Frequency", legend=false, dpi=400)
#vline for the gound truth value
vline!([original_r], color=:red, label="", linewidth=2)
#dashed vline for the bounds
vline!([r_bounds[1], r_bounds[2]], color=:black, linestyle=:dash, label="", linewidth=1)
Plots.savefig("histogram_r_values.svg")

#save the variation coefficients
variation_coefficients = Dict("sigma" => variation_coefficient_σ, "r" => variation_coefficient_r)
CSV.write("variation_coefficients.csv", DataFrame(variation_coefficients))
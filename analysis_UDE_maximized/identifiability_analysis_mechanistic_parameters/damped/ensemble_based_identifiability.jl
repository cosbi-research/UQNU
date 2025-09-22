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
    maximized_ensemble_folder = "../../results_maximized/damped/result_damped$i/results.jld"
    if !isfile(maximized_ensemble_folder)
        continue
    end
    tmp_results = deserialize(maximized_ensemble_folder)
    tmp_ensemble = tmp_results.naive_ensemble_reference
    push!(ensembles, tmp_ensemble)
end

α_values = vcat([model.α for ens in ensembles for model in ens]...)

#initialization bounds 
original_α = 0.1

α_bounds = (original_α * 0.5, original_α * 2.0)

variation_coefficient_α = std(α_values) / mean(α_values)

#plot an histogram of alpha values 
binwidth = (α_bounds[2] - α_bounds[1])/20
Plots.histogram(α_values, bins=α_bounds[1]:binwidth:α_bounds[2], title="", xlabel="α", ylabel="Frequency", legend=false, dpi=400)
#vline for the gound truth value
vline!([original_α], color=:red, label="", linewidth=2)
#dashed vline for the bounds
vline!([α_bounds[1], α_bounds[2]], color=:black, linestyle=:dash, label="", linewidth=1)
Plots.savefig("histogram_α_values.svg")
#plot an histogram of delta values


#save the variation coefficients
variation_coefficients = Dict("alpha" => variation_coefficient_α,)
CSV.write("variation_coefficients.csv", DataFrame(variation_coefficients))  

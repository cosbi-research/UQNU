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

#bounding box of interest 
boundig_box_vect_field = deserialize("../../data_generator/lotka_volterra_in_silico_data_bounding_box.jld")
experimental_data = deserialize("../../data_generator/lotka_volterra_in_silico_data_no_noise.jld")

in_dim = 2
out_dim = 2

activation_function_fun = gelu
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=0.01)

#gound truth model
# parameters for Lotka Volterra and initial state
original_parameters = Float64[1.3, 0.9, 0.8, 1.8]
#function to generate the Data
function lotka_volterra_gound_truth(u)
    α, β, γ, δ = original_parameters
    du = similar(u)
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
    return du
end

ensembles = []
for i in 1:10
    maximized_ensemble_folder = "../../results_maximized/lv/result_lv$i/results.jld"
    if !isfile(maximized_ensemble_folder)
        continue
    end
    tmp_results = deserialize(maximized_ensemble_folder)
    tmp_ensemble = tmp_results.naive_ensemble_reference
    push!(ensembles, tmp_ensemble)
end

α_values = vcat([model.α for ens in ensembles for model in ens]...)
δ_values = vcat([model.δ for ens in ensembles for model in ens]...)

#initialization bounds 
original_α = 1.3
original_δ = 1.8

α_bounds = (original_α * 0.5, original_α * 2.0)
δ_bounds = (original_δ * 0.5, original_δ * 2.0)

variation_coefficient_α = std(α_values) / mean(α_values)
variation_coefficient_δ = std(δ_values) / mean(δ_values) 

#plot an histogram of alpha values 
binwidth = (α_bounds[2] - α_bounds[1])/20
Plots.histogram(α_values, bins=α_bounds[1]:binwidth:α_bounds[2], title="", xlabel="α", ylabel="Frequency", legend=false, dpi=400)
#vline for the gound truth value
vline!([original_α], color=:red, label="", linewidth=2)
#dashed vline for the bounds
vline!([α_bounds[1], α_bounds[2]], color=:black, linestyle=:dash, label="", linewidth=1)
Plots.savefig("histogram_α_values.svg")
#plot an histogram of delta values

binwidth = (δ_bounds[2] - δ_bounds[1])/20
Plots.histogram(δ_values, bins=δ_bounds[1]:binwidth:δ_bounds[2], title="", xlabel="δ", ylabel="Frequency", legend=false, dpi=400)
#vline for the gound truth value
vline!([original_δ], color=:red, label="", linewidth=2)
#dashed vline for the bounds
vline!([δ_bounds[1], δ_bounds[2]], color=:black, linestyle=:dash, label="", linewidth=1)
Plots.savefig("histogram_δ_values.svg")

#save the variation coefficients
variation_coefficients = Dict("alpha" => variation_coefficient_α, "delta" =>
    variation_coefficient_δ)
CSV.write("variation_coefficients.csv", DataFrame(variation_coefficients))  

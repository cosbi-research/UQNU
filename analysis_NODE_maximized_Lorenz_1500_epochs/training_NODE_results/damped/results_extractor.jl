cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions
using PlotlyJS

results_1 = deserialize("ensemble_results_model_3_gain_1.0_reg_0_0.0_to_keep.jld")

results_1_with_seed = []
# add a field seed with an increasing identificator
for (i, res) in enumerate(results_1)
    if res.status == "success"
        new_res = (
            elapsed_time = res.elapsed_time,
            training_res = res.training_res,
            net_status = res.net_status,
            learning_rate_adam = res.learning_rate_adam,
            neural_network_dimension = res.neural_network_dimension,
            activation_function = res.activation_function,
            regularization = res.regularization,
            regularization_coefficient_1 = res.regularization_coefficient_1,
            regularization_coefficient_2 = res.regularization_coefficient_2,
            gain = res.gain,
            error_level = res.error_level,
            validation_likelihood = res.validation_likelihood,
            status = res.status,
            seed = i
        )
        push!(results_1_with_seed, new_res)
    else
        new_res = (
            status = res.status,
            seed = i
        )
        push!(results_1_with_seed, new_res)
    end
end

#save the results with seed
serialize("ensemble_results_model_3_with_seed.jld", results_1_with_seed)

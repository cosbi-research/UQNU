cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions
using PlotlyJS

#experimental_data = deserialize("../../../data_generator/lorenz_in_silico_data_no_noise.jld")
training_data_structure = deserialize("../../../data_generator/lorenz_training_ds_err_1.jld")

trajectories = [1, 2, 3]

experimental_data_1 = training_data_structure.solution_dataframes[1]
experimental_data_2 = training_data_structure.solution_dataframes[2]
experimental_data_3 = training_data_structure.solution_dataframes[3]

experimental_datas = [experimental_data_1, experimental_data_2, experimental_data_3]

max_oscillations_1 = training_data_structure.max_oscillations[1]
max_oscillations_2 = training_data_structure.max_oscillations[2]
max_oscillations_3 = training_data_structure.max_oscillations[3]

max_oscillations = [max_oscillations_1, max_oscillations_2, max_oscillations_3]

initial_time_training = 0.0f0
end_time_training = 2.0f0

integrator = Vern7()
abstol = 1e-7
reltol = 1e-6
sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))

get_uode_model_function = function (appr_neural_network, state)
    #generates the function with the parameters
    f(du, u, p, t) =
        let appr_neural_network = appr_neural_network, st = state
            û = appr_neural_network(u, p, st)[1]
            @inbounds du[1] = û[1]
            @inbounds du[2] = û[2]
            @inbounds du[3] = û[3]
        end
end

results_1 = deserialize("ensemble_results_model_2_gain_1.0_reg_0_first_ensembles.jld")
results_2 = deserialize("ensemble_results_model_2_gain_1.0_reg_0_second_ensembles.jld")

result = vcat(results_1, results_2)
serialize("ensemble_results_model_2_gain_1.0_reg_0.jld", result)
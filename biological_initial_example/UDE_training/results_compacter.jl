cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, DiffEqFlux , Distributions


neural_network_dimension = 32
activation_function = 4

########################################################### general configurations ######################################################################
rng = Random.default_rng()
Random.seed!(rng, 0)

# numerical integrator
integrator = TRBDF2(autodiff=true);
abstol = 1e-7
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

#load the training results 
results = deserialize("cell_apop_UDE_results/ensemble_results.jld")
results_51 = deserialize("cell_apop_UDE_results/ensemble_results_prova.jld")
results_44  = deserialize("cell_apop_UDE_results/ensemble_results_prova_1.jld")
results[51] = results_51[1]
results[44] = results_44[1]

serialize("cell_apop_UDE_results/compacted_ensemble_results.jld", results)

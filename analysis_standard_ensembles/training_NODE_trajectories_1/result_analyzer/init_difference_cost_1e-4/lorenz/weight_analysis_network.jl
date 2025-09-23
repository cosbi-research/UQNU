cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions
using PlotlyJS

gr()

results_1 = deserialize("ensemble_results_model_2_gain_1.0_reg_0.jld")
results_01 = deserialize("ensemble_results_model_2_gain_0.1_reg_0.jld")
results_001 = deserialize("ensemble_results_model_2_gain_0.01_reg_0.jld")

threshold = 0.01

results_1 = [res for res in results_1 if res.status == "success" && res.validation_likelihood < threshold]
results_01 = [res for res in results_01 if res.status == "success" && res.validation_likelihood < threshold]
results_001 = [res for res in results_001 if res.status == "success" && res.validation_likelihood < threshold]

# plot the variability according to the regulazation metrics
function get_reg_cost(res, regularization)
    pars = res.training_res.p
    weights = vcat([vec(pars[layer_name].weight) for layer_name in keys(pars)]...)
    if regularization == 1
      return sum(abs, weights)
    elseif regularization == 2
      return sum(abs2, weights)
    elseif regularization == 3
      return sum(abs2, weights) + sum(abs, weights)
    end
    return 0.0
end

L1_cost_1 = [get_reg_cost(res, 1) for res in results_1 if res.status == "success"]
L1_cost_01 = [get_reg_cost(res, 1) for res in results_01 if res.status == "success"]
L1_cost_001 = [get_reg_cost(res, 1) for res in results_001 if res.status == "success"]

#histogram of the L1 cost
using StatsPlots
plt = StatsPlots.density(log10.(L1_cost_1), label="1.0", xlabel="L1 cost", ylabel="Frequency", title="L1 cost distribution", alpha=0.5,  color=:auto, lw = 2)
StatsPlots.density!(plt, log10.(L1_cost_01), label="0.1", alpha=0.5,  color=:auto, lw = 2)
StatsPlots.density!(plt,log10.(L1_cost_001), label="0.01", alpha=0.5,  color=:auto, lw = 2)

Plots.savefig(plt, "L1_cost_distribution.png")

L2_cost_1 = [get_reg_cost(res, 2) for res in results_1 if res.status == "success"]
L2_cost_01 = [get_reg_cost(res, 2) for res in results_01 if res.status == "success"]
L2_cost_001 = [get_reg_cost(res, 2) for res in results_001 if res.status == "success"]

#histogram of the L1 cost
plt = StatsPlots.density(log10.(L2_cost_1), label="1", xlabel="L1 cost", ylabel="Frequency", title="L2 cost distribution", alpha=0.5,  color=:auto, lw = 2)
StatsPlots.density!(plt, log10.(L2_cost_01), label="0.1", alpha=0.5,  color=:auto, lw = 2)
StatsPlots.density!(plt,log10.(L2_cost_001), label="0.01", alpha=0.5,  color=:auto, lw = 2)

Plots.savefig(plt, "L2_cost_distribution.png")



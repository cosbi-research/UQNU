cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions
using Logging, StatsBase, Infiltrator

loglevel = Logging.Info
global_logger(ConsoleLogger(stderr, loglevel))
debug_folder = "debug_lv"
result_folder = "result_lv"

trajectories = deserialize("trajectories_lv.jld")

plt = Plots.plot()

counter = 0
for traj in trajectories

    counter += 1

    alpha_path = [res.α for res in traj]
    delta_path = [res.δ for res in traj]
    
    Plots.plot!(plt, alpha_path, delta_path, label="trajectory "*string(counter), alpha=0.5)
end

Plots.plot!(plt, [1.0], [1.0], label="original estimate point", marker=:circle, markersize=5, color=:black, legend = false)
#set the x label and y label
Plots.xlabel!(plt, L"$\alpha$", fontsize=12)
Plots.ylabel!(plt, L"$\delta$", fontsize=12)
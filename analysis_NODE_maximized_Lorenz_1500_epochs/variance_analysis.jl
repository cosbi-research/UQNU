cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions
using Logging

variances = deserialize("variances.jld")

plt = Plots.plot()
last_iteration = 0
for ensemble_member in axes(variances, 1)
  variances_ensemble = variances[ensemble_member]
  iterations = 1:length(variances_ensemble)
  iterations = iterations .+ last_iteration

  last_iteration = last_iteration + length(variances_ensemble)

  Plots.plot!(plt, iterations, variances_ensemble, label = "Ensemble member $ensemble_member")
end

display(plt)

savefig(plt, "variances.png")
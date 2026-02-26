cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

gr()

include("../ConfidenceEllipse.jl")
using .ConfidenceEllipse

### vector field comparison #####
mc_vector_field_standard = deserialize("cicps_distribution_naive.jld")
mc_vector_field_maximized = deserialize("cicps_distribution.jld")

#calculate the mean of cp and the sem 
mean_cp_standard = mean(mc_vector_field_standard)
sem_cp_standard = std(mc_vector_field_standard) / sqrt(length(mc_vector_field_standard))
mean_cp_maximized = mean(mc_vector_field_maximized)
sem_cp_maximized = std(mc_vector_field_maximized) / sqrt(length(mc_vector_field_maximized))

#compare the distributions with wilcoxon signed-rank test
using HypothesisTests
mc_vector_field_maximized_diff_theory = abs.(mc_vector_field_maximized .- 0.95)
mc_vector_field_standard_diff_theory = abs.(mc_vector_field_standard .- 0.95)
test = HypothesisTests.ApproximateSignedRankTest(mc_vector_field_maximized_diff_theory, mc_vector_field_standard_diff_theory)
pval = pvalue(test, tail=:left)

### trajectory comparison #####
c_trajectory_standard = deserialize("trajectory_results_total_naive.jld")
c_trajectory_maximized = deserialize("trajectory_results_total.jld")

mcp_standard = [mean(trajectory) for trajectory in c_trajectory_standard]
mcp_maximized = [mean(trajectory) for trajectory in c_trajectory_maximized]

#calculate the mean and sem of the cp
mean_cp_trajectory_standard = mean(mcp_standard)
mean_cp_trajectory_maximized = mean(mcp_maximized)
sem_cp_trajectory_standard = std(mcp_standard) / sqrt(length(mcp_standard))
sem_cp_trajectory_maximized = std(mcp_maximized) / sqrt(length(mcp_maximized))

#compare the distributions with wilcoxon signed-rank test
mcp_maximized_diff_theory = abs.(mcp_maximized .- 0.95)
mcp_standard_diff_theory = abs.(mcp_standard .- 0.95)
test_trajectory = HypothesisTests.ApproximateSignedRankTest(mcp_maximized_diff_theory, mcp_standard_diff_theory)
pval_trajectory = pvalue(test_trajectory, tail=:left)

#write the results in two named tuples
results_vector_field = (mean_cp_standard = mean_cp_standard, sem_cp_standard = sem_cp_standard, mean_cp_maximized = mean_cp_maximized, sem_cp_maximized = sem_cp_maximized, comparison_p_val = pval)
results_trajectory = (mean_cp_trajectory_standard = mean_cp_trajectory_standard, sem_cp_trajectory_standard = sem_cp_trajectory_standard, mean_cp_trajectory_maximized = mean_cp_trajectory_maximized, sem_cp_trajectory_maximized = sem_cp_trajectory_maximized, comparison_p_val = pval_trajectory)


#serialize in a file
serialize("comparison_results_lorenz.jld", (results_vector_field=results_vector_field, results_trajectory=results_trajectory))
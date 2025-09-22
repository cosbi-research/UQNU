cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

gr()

include("../ConfidenceEllipse.jl")
using .ConfidenceEllipse

experimental_points_projected_to_plot = deserialize("experimental_points_projected_to_plot.jld")
alphas = experimental_points_projected_to_plot.alphas
experimental_points_centered_basis = experimental_points_projected_to_plot.experimental_points_centered_basis


results_naive_deserialized = deserialize("results_analysis_projected_naive.jld")
results_naive_ensemble = results_naive_deserialized.results_ensemble
points = results_naive_deserialized.points

results_maximized_deserialized = deserialize("results_analysis_projected.jld")
results_maximized_ensemble = results_maximized_deserialized.results_ensemble

mean_areas_in_maximized_ensemble = ([mean([results_maximized_ensemble[j].ellipse_areas[i] for j in axes(results_maximized_ensemble, 1)]) for i in axes(points, 1)])
mean_areas_in_naive = [mean([results_naive_ensemble[j].ellipse_areas[i] for j in axes(results_naive_ensemble, 1)]) for i in axes(points, 1)]

mean_areas_in_maximized_ensemble = max.(0.01, mean_areas_in_maximized_ensemble)
mean_areas_in_naive = max.(0.01, mean_areas_in_naive)

max_area = max(maximum(mean_areas_in_maximized_ensemble), maximum(mean_areas_in_naive))
min_area = min(minimum(mean_areas_in_maximized_ensemble), minimum(mean_areas_in_naive))

#add two fake points to force the visualization between 0 and 1
points_to_plot = deepcopy(points)
push!(points_to_plot, points[1])
push!(points_to_plot, points[1])
push!(mean_areas_in_maximized_ensemble, min_area)
push!(mean_areas_in_maximized_ensemble, max_area)
push!(mean_areas_in_naive, min_area)
push!(mean_areas_in_naive, max_area)

points_to_plot = reverse(points_to_plot)
mean_areas_in_maximized_ensemble = reverse(mean_areas_in_maximized_ensemble)
mean_areas_in_naive = reverse(mean_areas_in_naive)

min_x = minimum([minimum(p[1]) for p in points_to_plot])
max_x = maximum([maximum(p[1]) for p in points_to_plot])
min_y = minimum([minimum(p[2]) for p in points_to_plot])
max_y = maximum([maximum(p[2]) for p in points_to_plot])

plt = Plots.scatter([s[1] for s in experimental_points_centered_basis], [s[2] for s in experimental_points_centered_basis], label="training data", color=:orange)
Plots.scatter!(plt, [p[1] for p in points_to_plot], [p[2] for p in points_to_plot], label="", xlabel="v1", ylabel="v2", zcolor=log10.(mean_areas_in_maximized_ensemble), color=:viridis, markerstrokewidth=0, markersize=4)
Plots.scatter!(plt, [s[1] for s in experimental_points_centered_basis], [s[2] for s in experimental_points_centered_basis], label="", alpha=alphas, color=:orange)
Plots.plot!(plt, xlims=(min_x, max_x), ylims=(min_y, max_y))
Plots.plot!(plt, title="MOD", legend=:topleft)
Plots.plot!(plt, 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),     # Increase y-axis label font size
    legendfont=font(10),
    right_margin = 10px
)

Plots.savefig(plt, "mean_area_maximized.png")
Plots.savefig(plt, "mean_area_maximized.svg")

plt = Plots.scatter([s[1] for s in experimental_points_centered_basis], [s[2] for s in experimental_points_centered_basis], label="training data", color=:orange)
Plots.scatter!(plt, [p[1] for p in points_to_plot], [p[2] for p in points_to_plot], label="", xlabel="v1", ylabel="v2", zcolor=log10.(mean_areas_in_naive), color=:viridis, markerstrokewidth=0, markersize=4)
Plots.scatter!(plt, [s[1] for s in experimental_points_centered_basis], [s[2] for s in experimental_points_centered_basis], label="", alpha=alphas, color=:orange)
Plots.plot!(plt, xlims=(min_x, max_x), ylims=(min_y, max_y))
Plots.plot!(plt, title="Standard", legend=:topleft)
Plots.plot!(plt, 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),     # Increase y-axis label font size
    legendfont=font(10),
    #left margin
    right_margin = 10px
)

Plots.savefig(plt, "mean_area_naive.png")
Plots.savefig(plt, "mean_area_naive.svg")
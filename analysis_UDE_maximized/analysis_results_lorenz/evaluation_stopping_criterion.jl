cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

gr()

trajectory_number = 100
attempts = 1000

include("../ConfidenceEllipse.jl")
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
variances = []
cicps = []
trajectories = []
for i in 1:10
    maximized_ensemble_folder_results = "../results_maximized/lorenz/result_lorenz$i/results.jld"
    variance_ensembles = "../results_maximized/lorenz/result_lorenz$i/variances_lorenz.jld"
    cicp_ensembles = "../results_maximized/lorenz/result_lorenz$i/cicps_lorenz.jld"
    traj_ensemble = "../results_maximized/lorenz/result_lorenz$i/trajectories_lorenz.jld"
    if !isfile(maximized_ensemble_folder_results)
        continue
    end
    tmp_results = deserialize(maximized_ensemble_folder_results)
    tmp_ensemble = tmp_results.ensemble_reprojected
    tmp_variances = deserialize(variance_ensembles)
    tmp_cicps = deserialize(cicp_ensembles)
    tmp_trajectories = deserialize(traj_ensemble)

    push!(ensembles, tmp_ensemble)
    push!(variances, tmp_variances)
    push!(cicps, tmp_cicps)
    push!(trajectories, tmp_trajectories)
end

second_model_plot = Plots.plot()
#plot the training trajectories for reference
third_model_plot = Plots.plot()
fourth_model_plot = Plots.plot()
fifth_model_plot = Plots.plot()

second_model_trajectories = Plots.plot()
third_model_trajectories = Plots.plot()
fourth_model_trajectories = Plots.plot()
fifth_model_trajectories = Plots.plot()

third_model_cicps = Plots.plot()
fourth_model_cicps = Plots.plot()
fifth_model_cicps = Plots.plot()

plot_comparison_cicp_variance_increase = Plots.plot()

#generate 10 colors rabnddomly 
colors =  ["red", "blue", "green", "purple", "orange", "cyan", "magenta", "yellow", "brown", "pink"] 

ensemble_index = 0
for variance in variances

    ensemble_index += 1
    for model_count in 1:length(variance)
        model_variance = variance[model_count]

        cicps_traj = nothing
        if model_count > 2
            cicps_traj = cicps[ensemble_index][model_count]
        end

        traj = trajectories[ensemble_index][model_count][1:(end-2)]
        
        first_point = traj[1]
        distances = [norm(first_point .- point)^2 for point in traj]

        if model_count == 1
            Plots.plot!(second_model_plot, 1:length(model_variance), model_variance; label="MOD ensemble $ensemble_index", legend=:topleft, markersize=3, color = colors[ensemble_index])
            Plots.plot!(second_model_trajectories, (1:length(distances)) .*2, distances; label="MOD ensemble $ensemble_index", legend=:topleft, markersize=3, color = colors[ensemble_index])
        elseif model_count == 2
            Plots.plot!(third_model_plot, 1:length(model_variance), model_variance; label="MOD ensemble $ensemble_index", legend=:topleft, markersize=3, color = colors[ensemble_index])
            Plots.plot!(third_model_trajectories,(1:length(distances)) .*2, distances; label="MOD ensemble $ensemble_index", legend=:topleft, markersize=3, color = colors[ensemble_index])
        elseif model_count == 3
            Plots.plot!(fourth_model_plot, 1:length(model_variance), model_variance; label="MOD ensemble $ensemble_index", legend=:topleft, markersize=3, color = colors[ensemble_index])
            Plots.plot!(fourth_model_trajectories, (1:length(distances)) .*2, distances; label="MOD ensemble $ensemble_index", legend=:topleft, markersize=3, color = colors[ensemble_index])
            Plots.plot!(fourth_model_cicps, 1:length(cicps_traj), cicps_traj; label="MOD ensemble $ensemble_index", legend=:topleft, markersize=3, color = colors[ensemble_index])
        else
            Plots.plot!(fifth_model_plot, 1:length(model_variance), model_variance; label="MOD ensemble $ensemble_index", legend=:topleft, markersize=3, color = colors[ensemble_index])
            Plots.plot!(fifth_model_trajectories, (1:length(distances)) .*2, distances; label="MOD ensemble $ensemble_index", legend=:topleft, markersize=3, color = colors[ensemble_index])
            Plots.plot!(fifth_model_cicps, 1:length(cicps_traj), cicps_traj; label="MOD ensemble $ensemble_index", legend=:topleft, markersize=3, color = colors[ensemble_index])
        end
    end
end 

Plots.plot!(second_model_plot,
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),
    legendfont=font(10),
    legend=:topleft,
)

Plots.plot!(second_model_trajectories,
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),
    legendfont=font(10),
    legend=:topleft,
)

#save as svg 
Plots.savefig(second_model_plot, "ensemble_variance_increase_second.svg")
Plots.savefig(second_model_trajectories, "ensemble_variance_increase_trajectories_second.svg")

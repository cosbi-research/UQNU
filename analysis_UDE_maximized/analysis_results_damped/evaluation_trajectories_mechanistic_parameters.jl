cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

gr()

result_folder = "trajectory_mechanistic_parameters"
if !isdir(result_folder)
    mkdir(result_folder)
end

include("../ConfidenceEllipse.jl")
using .ConfidenceEllipse


in_dim = 2
out_dim = 2

activation_function_fun = gelu
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=0.01)

original_ude_parameters = [0.1] 
lower_parameter_boundaries_vec = original_ude_parameters * 0.5
upper_parameter_boundaries_vec = original_ude_parameters * 2.0

ensemble_trajectories = []
initial_points_original =  []
for i in 1:10
    maximized_ensemble_trajectories = "../results_maximized/damped/result_damped$i/trajectories_damped.jld"
    if !isfile(maximized_ensemble_trajectories)
        continue
    end
    tmp_trajectories= deserialize(maximized_ensemble_trajectories)
    maximized_ensemble_results = deserialize("../results_maximized/damped/result_damped$i/results.jld")
    starting_point = maximized_ensemble_results.ensemble_reprojected[1]

    starting_point_α = starting_point.α

    push!(ensemble_trajectories, tmp_trajectories)
    push!(initial_points_original, starting_point_α)
end

#for each ensemble
counter = 0
for ensemble_trajectory in ensemble_trajectories

  counter += 1

  first_trajectory_delta = [res.α for res in ensemble_trajectory[1]][1:end-1] .* initial_points_original[counter]
  second_trajectory_delta = [res.α for res in ensemble_trajectory[2]][1:end-1].* initial_points_original[counter]
  third_trajectory_delta = [res.α for res in ensemble_trajectory[3]][1:end-1].* initial_points_original[counter]
  fourth_trajectory_delta = [res.α for res in ensemble_trajectory[4]][1:end-1].* initial_points_original[counter]

  #plot the trajectories
  plt = Plots.scatter([1], [initial_points_original[counter]], label="Initial model", color=:black, markersize=5)
  Plots.plot!(plt, 1:length(first_trajectory_delta), first_trajectory_delta, label="Training model 2", color=:blue, linewidth=2, legend=:topright)
  Plots.plot!(plt, 1:length(second_trajectory_delta), second_trajectory_delta, label="Training model 3", color=:red, linewidth=2)
  Plots.plot!(plt, 1:length(third_trajectory_delta), third_trajectory_delta, label="Training model 4", color=:green, linewidth=2)
  Plots.plot!(plt, 1:length(fourth_trajectory_delta), fourth_trajectory_delta, label="Training model 5", color=:orange, linewidth=2)
  

  #draw a dashed rectangle around the bounding box
  min_y1, max_y1 = lower_parameter_boundaries_vec[1], upper_parameter_boundaries_vec[1]

  Plots.plot!(plt, ylims=(min_y1, max_y1))

  ground_truth_α = original_ude_parameters[1]
  #draw dashed lines on the true values
  Plots.plot!(plt, [1, length(first_trajectory_delta)], [ground_truth_α, ground_truth_α], label="", color=:grey, linestyle=:dash, linewidth=2)


  Plots.plot!(xlabel=L"iteration", ylabel=L"α", title="", legend=:topright)
  Plots.plot!(plt, 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(18),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),     # Increase y-axis label font size
    legendfont=font(10)
  )


  #save the plot
  plot_path = joinpath(result_folder, "ensemble_trajectory_$counter.png")
  Plots.savefig(plt, plot_path)
  Plots.savefig(plt, replace(plot_path, ".png" => ".svg"))
end

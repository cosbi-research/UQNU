cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

gr()

#deserialize results maximixed on trajectories
maximized_trajectory_results = deserialize("trajectory_results_total.jld")
naive_trajectory_results = deserialize("trajectory_results_total_naive.jld")

maximized_trajectory_results = [mean(trajectory) for trajectory in maximized_trajectory_results]
naive_trajectory_results = [mean(trajectory) for trajectory in naive_trajectory_results]

# Create boxplots for the two sets of results
plt = Plots.boxplot(["Standard" for _ in naive_trajectory_results], naive_trajectory_results, label="Standard", alpha=0.5, color=:lightblue, xlabel="", ylabel="CP")
Plots.boxplot!(plt, ["Maximized OOD disagreement" for _ in maximized_trajectory_results], maximized_trajectory_results, label="Maximized OOD disagreement", alpha=0.5, color=:lightblue)
Plots.yaxis!(plt, (0, 1.1))

using HypothesisTests
maximized_trajectory_results = abs.(maximized_trajectory_results .- 0.95)
naive_trajectory_results = abs.(naive_trajectory_results .- 0.95)
test = HypothesisTests.ApproximateSignedRankTest(maximized_trajectory_results, naive_trajectory_results)
pval = pvalue(test, tail = :left)

Plots.plot!(plt, title="p-value: "*string(round(pval, sigdigits=2)))
Plots.plot!(plt, 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(18),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),     # Increase y-axis label font size
    legendfont=font(10),
    legend=false,
    rightmargin=30px
)

Plots.savefig(plt, "trajectory_results_comparison.png")
Plots.savefig(plt, "trajectory_results_comparison.svg")


#plot the trajectories
trajectory_folder = "trajectories"
if !isdir(trajectory_folder)
    mkdir(trajectory_folder)
end

#deserialize the pointwise solutions for the trajectories
maximized_trajectory_results = deserialize("pointwise_solutions_total.jld")
naive_trajectory_results = deserialize("pointwise_solutions_total_naive.jld")

for trajectory_index in axes(maximized_trajectory_results[2], 1)
    #plot the ground truth solution
    ground_truth_solution = maximized_trajectory_results[1][trajectory_index]

    plt1 = Plots.plot()
    plt2 = Plots.plot()

    n_ensemble = 5
    t_value = quantile(TDist(n_ensemble - 1), 0.975)

    multiplier = t_value * sqrt(1+1/n_ensemble)

    ensemble_results = maximized_trajectory_results[1][trajectory_index]
    #for ensemble_results in [maximized_trajectory_results[i][trajectory_index] for i in axes(maximized_trajectory_results, 1)]
        
        #if ensemble_results.sd_x1[end] > 1e5
        #    continue
        #end
        
        Plots.plot!(plt1, ensemble_results.t, ensemble_results.mean_x1, ribbon=multiplier .* ensemble_results.sd_x1, fillalpha=0.1, color=:blue, linealpha=0, label="")
        Plots.plot!(plt2, ensemble_results.t, ensemble_results.mean_x2, ribbon=multiplier .* ensemble_results.sd_x2, fillalpha=0.1, color=:blue, linealpha=0, label="")
    #end

    ensemble_results = naive_trajectory_results[1][trajectory_index]
    #for ensemble_results in [naive_trajectory_results[i][trajectory_index] for i in axes(naive_trajectory_results, 1)]
        Plots.plot!(plt1, ensemble_results.t, ensemble_results.mean_x1, ribbon=multiplier .* ensemble_results.sd_x1, fillalpha=0.5, color=:red, linealpha=0, label="")
        Plots.plot!(plt2, ensemble_results.t, ensemble_results.mean_x2, ribbon=multiplier .* ensemble_results.sd_x2, fillalpha=0.5, color=:red, linealpha=0, label="")
    #end

    Plots.plot!(plt1, ground_truth_solution.t, ground_truth_solution.x1, label="Ground truth x", xlabel="t", ylabel="x", title="", color=:black)
    Plots.plot!(plt2, ground_truth_solution.t, ground_truth_solution.x2, label="Ground truth y", xlabel="t", ylabel="y", title="", color=:black)

    max_y1 = maximum([maximum(ensemble_results.mean_x1 .+ 1.96 .* ensemble_results.sd_x1), maximum(ground_truth_solution.x1)])
    max_y2 = maximum([maximum(ensemble_results.mean_x2 .+ 1.96 .* ensemble_results.sd_x2), maximum(ground_truth_solution.x2)])
    min_y1 = minimum([minimum(ensemble_results.mean_x1 .- 1.96 .* ensemble_results.sd_x1), minimum(ground_truth_solution.x1)])
    min_y2 = minimum([minimum(ensemble_results.mean_x2 .- 1.96 .* ensemble_results.sd_x2), minimum(ground_truth_solution.x2)])

    Plots.plot!(plt1, 
        xguidefont=font(18),    # Increase x-axis label font size
        yguidefont=font(18),
        titlefont=font(18),
        xtickfont=font(12),     # Increase x-axis tick font size
        ytickfont=font(12),     # Increase y-axis label font size
        legendfont=font(10),
        legend=false

    )

    Plots.plot!(plt2, 
        xguidefont=font(18),    # Increase x-axis label font size
        yguidefont=font(18),
        titlefont=font(18),
        xtickfont=font(12),     # Increase x-axis tick font size
        ytickfont=font(12),     # Increase y-axis label font size
        legendfont=font(10),
        legend=false

    )

    #Plots.yaxis!(plt1, (max(0, min_y1), max_y1))
    #Plots.yaxis!(plt2, (max(0, min_y2), max_y2))

    Plots.savefig(plt1, joinpath(trajectory_folder, "trajectory_$trajectory_index._x.png"))
    Plots.savefig(plt2, joinpath(trajectory_folder, "trajectory_$trajectory_index._y.png"))
end
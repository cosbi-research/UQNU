cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

gr()

# compute for each point the confidence interval
number_of_iteration_performed = []
for i in 1:10
    maximized_ensemble_folder = "../results_maximized_1500_epochs/lorenz/result_lorenz$i/results.jld"
    if !isfile(maximized_ensemble_folder)
        continue
    end
    tmp_results = deserialize(maximized_ensemble_folder)
    iteration_performed = min(size(tmp_results.validation_costs[1],1), 1500)
    push!(number_of_iteration_performed, iteration_performed)
    iteration_performed = min(size(tmp_results.validation_costs[2],1), 1500)
    push!(number_of_iteration_performed, iteration_performed)
    iteration_performed = min(size(tmp_results.validation_costs[3],1), 1500)
    push!(number_of_iteration_performed, iteration_performed)
    iteration_performed = min(size(tmp_results.validation_costs[4],1), 1500)
    push!(number_of_iteration_performed, iteration_performed)
end


#plot as an histogram the costs
plt = Plots.plot(dpi=1000)
bins = range(0, 1500, length=30)  # adjust length as needed

Plots.histogram!(plt, number_of_iteration_performed, label="", color="blue", alpha=0.5, bins=bins)
#vline in 10^-3
Plots.vline!(plt, [1500], color="black", linestyle=:dash, label ="")
Plots.plot!(plt, 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(18),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),     # Increase y-axis label font size
    legendfont=font(10),
    xlabel="Epochs performed",
    ylabel="",
    legend = :topright
)

#save the plot
Plots.savefig(plt, "number_of_epochs_performed.png")
Plots.savefig(plt, "number_of_epochs_performed.svg")
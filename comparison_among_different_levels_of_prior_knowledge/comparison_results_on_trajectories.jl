cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions
using HypothesisTests

gr()

### trajectories comparison #####

test_cases = ["lotka-volterra", "damped", "lorenz"]
titles = ["\nLotka-Volterra", "\nDamped oscillator", "\nLorenz system"]

plots = []


for test_case_index in axes(test_cases,1)

    test_case = test_cases[test_case_index]
    title_case = titles[test_case_index]

    cicps_trajectories_standard_NODE =
        deserialize("./results_on_trajectories/full_reconstruction/" * test_case * "/trajectory_results_total.jld")
    cicps_trajectories_standard_UDE =
        deserialize("./results_on_trajectories/part_reconstruction/" * test_case * "/trajectory_results_total.jld")
    cicps_trajectories_standard_UDE_fixed =
        deserialize("./results_on_trajectories/part_reconstruction_with_known_parameters/" * test_case * "/trajectory_results_total.jld")

    cicps_trajectories_standard_NODE = [mean(x) for x in cicps_trajectories_standard_NODE]
    cicps_trajectories_standard_UDE = [mean(x) for x in cicps_trajectories_standard_UDE]
    cicps_trajectories_standard_UDE_fixed = [mean(x) for x in cicps_trajectories_standard_UDE_fixed]

    # boxplot of the three distributions
    data = DataFrame(
        cicps = vcat(
            cicps_trajectories_standard_NODE,
            cicps_trajectories_standard_UDE,
            cicps_trajectories_standard_UDE_fixed
        ),
        method = vcat(
            fill("NODE", length(cicps_trajectories_standard_NODE)),
            fill("UDE", length(cicps_trajectories_standard_UDE)),
            fill("UDE (fixed mech. pars)", length(cicps_trajectories_standard_UDE_fixed))
        )
    )

    plt_tmp = @df data boxplot(
        :method,
        :cicps,
        legend = false,
        xlabel = "",
        ylabel = "CP",
        title = title_case,
        xtickfont = font(13)
    )

    plot!(
        xrotation = 45,
        bottom_margin = 25Plots.mm,
        top_margin = 20Plots.mm,
        left_margin = 15Plots.mm
    )

    push!(plots, plt_tmp)

    # statistical tests
    test_NODE_UDE =
        MannWhitneyUTest(cicps_trajectories_standard_NODE, cicps_trajectories_standard_UDE)
    test_NODE_UDE_fixed =
        MannWhitneyUTest(cicps_trajectories_standard_NODE, cicps_trajectories_standard_UDE_fixed)
    test_UDE_UDE_fixed =
        MannWhitneyUTest(cicps_trajectories_standard_UDE, cicps_trajectories_standard_UDE_fixed)

    # save statistics
    open("comparison_results_trajectories_standard_" * test_case * ".txt", "w") do f
        write(f, "Comparison of CICPS distributions on trajectories reconstruction\n")
        write(f, "Test NODE vs UDE: p-value = $(pvalue(test_NODE_UDE))\n")
        write(f, "Test NODE vs UDE with known parameters: p-value = $(pvalue(test_NODE_UDE_fixed))\n")
        write(f, "Test UDE vs UDE with known parameters: p-value = $(pvalue(test_UDE_UDE_fixed))\n")
    end

end

# combine all plots
plt = Plots.plot(
    plots...,
    layout = (1, 3),
    size = (1200, 600),
    plot_title = "Comparison of the distributions of CP on trajectories reconstruction"
)

savefig("./comparison_cicps_distributions_trajectories.svg")

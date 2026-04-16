cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions
using HypothesisTests

gr()

### vector field comparison #####

test_cases = ["lotka-volterra", "damped", "lorenz"]
titles = ["\nLotka-Volterra", "\nDamped oscillator", "\nLorenz system"]

plots = []
for test_case_index in axes(test_cases,1)

    test_case = test_cases[test_case_index]
    title_case = titles[test_case_index]

    cicps_vector_filed_standard_NODE = deserialize("./results_on_vector_field/full_reconstruction/" * test_case * "/cicps_distribution.jld")
    cicps_vector_filed_standard_UDE = deserialize("./results_on_vector_field/part_reconstruction/" * test_case * "/cicps_distribution.jld")
    cicps_vector_field_standard_UDE_fixed = deserialize("./results_on_vector_field/part_reconstruction_with_known_parameters/" * test_case * "/cicps_distribution.jld")

    #boxlot of the three distributions
    data = DataFrame(cicps=vcat(cicps_vector_filed_standard_NODE, cicps_vector_filed_standard_UDE, cicps_vector_field_standard_UDE_fixed),
        method=vcat(fill("NODE", length(cicps_vector_filed_standard_NODE)), fill("UDE", length(cicps_vector_filed_standard_UDE)), fill("UDE (fixed mech. pars)", length(cicps_vector_field_standard_UDE_fixed))))

    plt_tmp = @df data boxplot(:method, :cicps, legend=false, ylabel="CP", title=title_case)
    Plots.plot!(xrotation = 45)
    Plots.plot!(
        bottom_margin = 25Plots.mm,
        top_margin = 20Plots.mm,
        left_margin = 15Plots.mm,
        xtickfont = font(13)
    )

    push!(plots, plt_tmp)
    #save the plot

    #perform the wilcoxon signed-rank test between the three distributions
    test_NODE_UDE = HypothesisTests.MannWhitneyUTest(cicps_vector_filed_standard_NODE, cicps_vector_filed_standard_UDE)
    pval_NODE_UDE = pvalue(test_NODE_UDE, tail=:both)

    test_NODE_UDE_fixed = HypothesisTests.MannWhitneyUTest(cicps_vector_filed_standard_NODE, cicps_vector_field_standard_UDE_fixed)
    pval_NODE_UDE_fixed = pvalue(test_NODE_UDE_fixed, tail=:both)

    test_UDE_UDE_fixed = HypothesisTests.MannWhitneyUTest(cicps_vector_filed_standard_UDE, cicps_vector_field_standard_UDE_fixed)
    pval_UDE_UDE_fixed = pvalue(test_UDE_UDE_fixed, tail=:both)


    #print in a text file
    open("comparison_results_vector_field_standard_" * test_case * ".txt", "w") do f
        write(f, "Comparison of CICPS distributions on vector field reconstruction\n")
        write(f, "Test NODE vs UDE: p-value = " * string(pval_NODE_UDE) * "\n")
        write(f, "Test NODE vs UDE with known parameters: p-value = " * string(pval_NODE_UDE_fixed) * "\n")
        write(f, "Test UDE vs UDE with known parameters: p-value = " * string(pval_UDE_UDE_fixed) * "\n")
    end

end

#put all the plots together
plt = Plots.plot(plots..., layout=(1,3), size=(1250,600), plot_title ="Comparison of the distributions of CP on vector field reconstruction")
savefig("./comparison_cicps_distributions_vector_field.svg")





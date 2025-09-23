cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions, HypothesisTests

result_image_folder = "ensemble_5_diagnostic"
if !isdir(result_image_folder)
    mkpath(result_image_folder)
end

gr()

trajectory_number = 100
include("../utils/ConfidenceEllipse.jl")
using .ConfidenceEllipse

threshold = 0.01

#set the seed
seed = 0
rng = StableRNG(seed)

#bounding box of interest 
boundig_box_vect_field = deserialize("../data_generator/lorenz_in_silico_data_bounding_box.jld")
experimental_data = deserialize("../data_generator/lorenz_in_silico_data_no_noise.jld")

#generate randomly the initial points by pertubing by a gaussian of 10% one of the points in the original trajectory (experimental data)
initial_points = []
for i in 1:trajectory_number
    index = rand(rng, 1:3)
    experimental_data_filtered = experimental_data[[1,102,203],:]
    point = experimental_data_filtered[index, :]
    perturbation_1 = rand(rng, Normal(1, 0.5), 3)
    push!(initial_points, [point.x1 * perturbation_1[1], point.x2 * perturbation_1[2], point.x3 * perturbation_1[3]])
end

calibrations = deserialize("ensemble_results_model_2_gain_1.0_reg_0_0.0_to_keep.jld")
ensemble_dimension = 5
ensembles = [calibrations[(k*ensemble_dimension + 1):(k*ensemble_dimension+ensemble_dimension)] for k in 0:9]

in_dim = 3
out_dim = 3

activation_function_fun = gelu
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=0.01)

#gound truth model
# parameters for Lotka Volterra and initial state
original_parameters = Float64[10, 28, 8/3]
#function to generate the Data
function lorenz!(du, u, p, t)
    σ, r, b = Float64[10, 28, 8/3]
    du[1] = σ*(u[2] - u[1]) 
    du[2] = u[1]*(r-u[3])- u[2]
    du[3] = u[1]*u[2] - b*u[3]
end

function sample_uniform_3d(a, b, c, d, e, f)
    x = rand(rng) * (b - a) + a  # Sample x-coordinate
    y = rand(rng) * (d - c) + c  # Sample y-coordinate
    z = rand(rng) * (f - e) + e  # Sample z-coordinate
    return (x, y, z)
end

min_y1, max_y_1, min_y2, max_y2, min_y3, max_y3 = boundig_box_vect_field

simulations = []
for initial_point in initial_points
    u0 = initial_point
    tspan = (0.0, 0.1)
    #define the problem
    prob = ODEProblem(lorenz!, [u0[1], u0[2], u0[3]], tspan, original_parameters)

    sol = solve(prob, Tsit5(), saveat=0.001)

    df = DataFrame(sol, [:t, :x1, :x2, :x3])

    #restrict the simulation to the bounding box
    push!(simulations, df)
end


######################### simulations with NODE #####################
neural_network_dimension = 32
approximating_neural_network = Lux.Chain(
    Lux.Dense(in_dim, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform),
    Lux.Dense(neural_network_dimension, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform),
    Lux.Dense(neural_network_dimension, out_dim; init_weight=my_glorot_uniform),
)
get_uode_model_function = function (appr_neural_network, state)
    #generates the function with the parameters
    f(du, u, p, t) =
        let appr_neural_network = appr_neural_network, st = state
            û = appr_neural_network(u, p, st)[1]
            du[1] = û[1]
            du[2] = û[2]
            du[3] = û[3]
        end
end

local_rng = StableRNG(seed)
p_net, st = Lux.setup(local_rng, approximating_neural_network)
model_function = get_uode_model_function(approximating_neural_network, st)

simulations_node_total_df = []
seed = 0
rng = StableRNG(seed)

for ensemble in ensembles
    simulations_node = []
    for i in axes(initial_points, 1)
        initial_point = initial_points[i]
        u0 = [initial_point[1], initial_point[2], initial_point[3]]

        times = simulations[i].t
        tspan = extrema(times)

        esemble_simulations = []
        for member in ensemble
            prob = ODEProblem(model_function, u0, tspan, member.training_res.p)
            simulation = solve(prob, Tsit5(), saveat=times)

            if simulation.retcode != :Success
                println("Simulation failed")
                continue
            end

            ensemble_simulation = DataFrame(simulation, [:t, :x1, :x2, :x3])
            push!(esemble_simulations, ensemble_simulation)
        end

        @info "number of successfuull simulations " * string(length(esemble_simulations))

        # for each time get the mean and the confidence interval
        mean_y_1 = []
        mean_y_2 = []
        mean_y_3 = []
        sd_y1 = []  
        sd_y2 = []
        sd_y3 = []

        for time in times
            y1 = []
            y2 = []
            y3 = []
            for ensemble_simulation in esemble_simulations
                push!(y1, ensemble_simulation[ensemble_simulation.t.==time, :x1][1])
                push!(y2, ensemble_simulation[ensemble_simulation.t.==time, :x2][1])
                push!(y3, ensemble_simulation[ensemble_simulation.t.==time, :x3][1])
            end
            push!(mean_y_1, mean(y1))
            push!(mean_y_2, mean(y2))
            push!(mean_y_3, mean(y3))
            push!(sd_y1, std(y1))
            push!(sd_y2, std(y2))
            push!(sd_y3, std(y3))
        end

        # dataframe with the mean and the confidence interval
        df = DataFrame(t=times, x1=mean_y_1, x2=mean_y_2, x3=mean_y_3, sd_x1=sd_y1, sd_x2=sd_y2, sd_x3 = sd_y3)

        push!(simulations_node, df)
    end
    push!(simulations_node_total_df, simulations_node)
end

### analyze trajectory for trajectory if the ground truth is inside the confidence interval
trajectory_results_total = []
pointwise_solutions_total = []
traversed_non_captured_regions_total = []

for ensemble_index in axes(ensembles, 1)
    simulations_node = simulations_node_total_df[ensemble_index]
    trajectory_results = []
    pointwise_solutions = []
    traversed_non_captured_regions = []
    for i in axes(initial_points, 1)
        simulation = simulations[i]
        simulation_node = simulations_node[i]

        # for each time get the mean and the confidence interval
        mean_y_1 = simulation_node.x1
        mean_y_2 = simulation_node.x2
        mean_y_3 = simulation_node.x3
        sd_y1 = simulation_node.sd_x1
        sd_y2 = simulation_node.sd_x2
        sd_y3 = simulation_node.sd_x3

        n_ensemble = length(ensembles[ensemble_index])
        t_value = quantile(TDist(n_ensemble - 1), 0.975)

        sup_y1 = mean_y_1 .+ t_value .* sd_y1 .* sqrt(1+1/n_ensemble)
        inf_y1 = mean_y_1 .- t_value .* sd_y1 .* sqrt(1+1/n_ensemble)
        sup_y2 = mean_y_2 .+ t_value .* sd_y2 .* sqrt(1+1/n_ensemble)
        inf_y2 = mean_y_2 .- t_value .* sd_y2 .* sqrt(1+1/n_ensemble)
        sup_y3 = mean_y_3 .+ t_value .* sd_y3 .* sqrt(1+1/n_ensemble)
        inf_y3 = mean_y_3 .- t_value .* sd_y3 .* sqrt(1+1/n_ensemble)

        ground_truth_y_1_in_ci = simulation.x1 .> inf_y1 .&& simulation.x1 .< sup_y1
        ground_truth_y_2_in_ci = simulation.x2 .> inf_y2 .&& simulation.x2 .< sup_y2
        ground_truth_y_3_in_ci = simulation.x3 .> inf_y3 .&& simulation.x3 .< sup_y3

        ground_truth_in_ci = ground_truth_y_1_in_ci .& ground_truth_y_2_in_ci .& ground_truth_y_3_in_ci

        solution_pointwise = DataFrame(t=simulation.t, x1=simulation.x1, x2=simulation.x2, x3=simulation.x3, mean_x1=mean_y_1, mean_x2=mean_y_2, mean_x3=mean_y_3, sd_x1=sd_y1, sd_x2=sd_y2, sd_x3=sd_y3, sup_x1=sup_y1, inf_x1=inf_y1, sup_x2=sup_y2, inf_x2=inf_y2, sup_x3=sup_y3, inf_x3=inf_y3, ground_truth_in_ci=ground_truth_in_ci)
    
        percentage = sum(ground_truth_in_ci) / length(ground_truth_in_ci)

        push!(trajectory_results, percentage)
        push!(pointwise_solutions, solution_pointwise)
    end
    push!(trajectory_results_total, trajectory_results)
    push!(pointwise_solutions_total, pointwise_solutions)
end

#deserialize results maximixed on trajectories
naive_trajectory_results = trajectory_results_total
mean_contained = [mean(res) for res in naive_trajectory_results]
mean_PICP = round(mean(mean_contained), digits=2)

# Compute the differences from the fixed value
differences = mean_contained .- 0.95
# Perform the Wilcoxon signed-rank test (two-tailed)
test = HypothesisTests.SignedRankTest(differences)
pvalue_test = pvalue(test; tail = :left)

#write the p-value on a text file
open(result_image_folder * "/p_value_trajectoreis.txt", "w") do file
    write(file, "P-value: $pvalue_test\n")
end

# Create boxplots for the two sets of results
plt = Plots.plot()
Plots.boxplot!(plt, [1], mean_contained, color=:lightblue)
Plots.plot!(plt, ylims=(0, 1), legend =false)
Plots.plot!(plt, title="Mean MPICP: $mean_PICP", xlabel="", ylabel="MPICP")

Plots.savefig(plt, result_image_folder*"/trajectory_results_comparison.png")

results = (mean_contained=mean_contained, mean_PICP=mean_PICP)

serialize(result_image_folder * "/results_total_trajectory.jld", results)

#plot the trajectories
trajectory_folder = result_image_folder * "/trajectories"
if !isdir(trajectory_folder)
    mkdir(trajectory_folder)
end

#deserialize the pointwise solutions for the trajectories
naive_trajectory_results = pointwise_solutions_total

for trajectory_index in axes(naive_trajectory_results[1], 1)
    #plot the ground truth solution
    ground_truth_solution = naive_trajectory_results[1][trajectory_index]

    plt1 = Plots.plot()
    plt2 = Plots.plot()
    plt3 = Plots.plot()

    for k in axes(naive_trajectory_results, 1)
        ensemble_results = naive_trajectory_results[k][trajectory_index]

        #if ensemble_results.sd_x1[end] > 1e5
        #    continue
        #end

        n_ensemble = 5
        t_value = quantile(TDist(n_ensemble - 1), 0.975)

        multiplier = t_value * sqrt(1+1/n_ensemble)

        Plots.plot!(plt1, ensemble_results.t, ensemble_results.mean_x1, ribbon=multiplier .* ensemble_results.sd_x1, fillalpha=0.1, color=:red, linealpha=0, label="")
        Plots.plot!(plt2, ensemble_results.t, ensemble_results.mean_x2, ribbon=multiplier .* ensemble_results.sd_x2, fillalpha=0.1, color=:red, linealpha=0, label="")
        Plots.plot!(plt3, ensemble_results.t, ensemble_results.mean_x3, ribbon=multiplier .* ensemble_results.sd_x3, fillalpha=0.1, color=:red, linealpha=0, label="")
    end

    Plots.plot!(plt1, ground_truth_solution.t, ground_truth_solution.x1, label="Ground truth x", xlabel="t", ylabel="x", title="", color=:black, legend = false, left_margin=20mm)
    Plots.plot!(plt2, ground_truth_solution.t, ground_truth_solution.x2, label="Ground truth y", xlabel="t", ylabel="y", title="", color=:black, legend = false, left_margin=20mm)
    Plots.plot!(plt3, ground_truth_solution.t, ground_truth_solution.x3, label="Ground truth z", xlabel="t", ylabel="z", title="", color=:black, legend = false, left_margin=20mm)

    Plots.plot!(plt1, 
        xguidefont=font(18),    # Increase x-axis label font size
        yguidefont=font(18),
        titlefont=font(18),
        xtickfont=font(12),     # Increase x-axis tick font size
        ytickfont=font(12),     # Increase y-axis label font size
        legendfont=font(10)
    )
    Plots.plot!(plt2, 
        xguidefont=font(18),    # Increase x-axis label font size
        yguidefont=font(18),
        titlefont=font(18),
        xtickfont=font(12),     # Increase x-axis tick font size
        ytickfont=font(12),     # Increase y-axis label font size
        legendfont=font(10)
    )
    Plots.plot!(plt3, 
        xguidefont=font(18),    # Increase x-axis label font size
        yguidefont=font(18),
        titlefont=font(18),
        xtickfont=font(12),     # Increase x-axis tick font size
        ytickfont=font(12),     # Increase y-axis label font size
        legendfont=font(10)
    )

    Plots.savefig(plt1, joinpath(trajectory_folder, "trajectory_$trajectory_index._x.svg"))
    Plots.savefig(plt2, joinpath(trajectory_folder, "trajectory_$trajectory_index._y.svg"))
    Plots.savefig(plt3, joinpath(trajectory_folder, "trajectory_$trajectory_index._z.svg"))
end

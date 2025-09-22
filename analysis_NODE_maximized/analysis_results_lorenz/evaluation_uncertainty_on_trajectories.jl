cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

gr()


trajectory_number = 100
include("../ConfidenceEllipse.jl")
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

ensembles = []
for i in 1:10
    maximized_ensemble_folder = "../results_maximized/lorenz/result_lorenz$i/results.jld"
    if !isfile(maximized_ensemble_folder)
        continue
    end
    tmp_results = deserialize(maximized_ensemble_folder)
    tmp_ensemble = tmp_results.ensemble_reprojected
    push!(ensembles, tmp_ensemble)
end

function sample_uniform_3d(a, b, c, d, e, f)
    x = rand(rng) * (b - a) + a  # Sample x-coordinate
    y = rand(rng) * (d - c) + c  # Sample y-coordinate
    z = rand(rng) * (f - e) + e  # Sample z-coordinate
    return (x, y, z)
end

simulations = []
for initial_point in initial_points
    u0 = initial_point
    tspan = (0.0, 0.1)
    #define the problem
    prob = ODEProblem(lorenz!, [u0[1], u0[2], u0[3]], tspan, original_parameters)

    sol = solve(prob, Vern7(), saveat=0.001, abstol=1e-6, reltol=1e-5)

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
            prob = ODEProblem(model_function, u0, tspan, member)
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


serialize("trajectory_results_total.jld", trajectory_results_total)
serialize("pointwise_solutions_total.jld", pointwise_solutions_total)


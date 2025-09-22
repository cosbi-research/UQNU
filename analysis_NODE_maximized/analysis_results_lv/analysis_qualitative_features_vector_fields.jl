cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

gr()

trajectory_number = 2
attempts = 1000

include("../ConfidenceEllipse.jl")
using .ConfidenceEllipse

threshold = 0.01

#set the seed
seed = 1
rng = StableRNG(seed)

#bounding box of interest 
boundig_box_vect_field = deserialize("../data_generator/lotka_volterra_in_silico_data_bounding_box.jld")
experimental_data = deserialize("../data_generator/lotka_volterra_in_silico_data_no_noise.jld")

#generate randomly the initial points by pertubing by a gaussian of 10% one of the points in the original trajectory (experimental data)
initial_points = []
vcs = [0.01, 0.5, 0.75]


for vc in vcs
    tmp_initial_points = []
    for i in 1:attempts
        index = rand(rng, 1:3)
        experimental_data_filtered = experimental_data[[1, 102, 203], :]
        point = experimental_data_filtered[index, :]
        perturbation_1 = rand(rng, Normal(0, 1), 2)

        perturbation_1 = perturbation_1 ./ norm(perturbation_1)  # normalize the perturbation
        perturbation_1 = perturbation_1 * vc  # scale the perturbation

        initial_x1 = point.x1 * (1 + perturbation_1[1])
        initial_x2 = point.x2 * (1 + perturbation_1[2])

        #ensure they are greater than 0
        if initial_x1 < 0 || initial_x2 < 0
            continue
        end

        push!(tmp_initial_points, [initial_x1, initial_x2])
    end
    append!(initial_points, tmp_initial_points[1:trajectory_number])  # limit to trajectory_number points
end

in_dim = 2
out_dim = 2

activation_function_fun = gelu
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=0.01)

#gound truth model
# parameters for Lotka Volterra and initial state
original_parameters = Float64[1.3, 0.9, 0.8, 1.8]
#function to generate the Data
function lotka_volterra_gound_truth(u)
    α, β, γ, δ = original_parameters
    du = similar(u)
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
    return du
end

function lokta_volterra!(du, u, p, t)
    α, β, γ, δ = original_parameters
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

ensembles = []
for i in 1:10
    maximized_ensemble_folder = "../results_maximized/lv/result_lv$i/results.jld"
    if !isfile(maximized_ensemble_folder)
        continue
    end
    tmp_results = deserialize(maximized_ensemble_folder)
    tmp_ensemble = tmp_results.ensemble_reprojected
    push!(ensembles, tmp_ensemble)
end

function sample_uniform_2d(a, b, c, d)
    x = rand(rng) * (b - a) + a  # Sample x-coordinate
    y = rand(rng) * (d - c) + c  # Sample y-coordinate
    return (x, y)
end

min_y1, max_y_1, min_y2, max_y2 = boundig_box_vect_field

# generates 100 initial points for the trajectories 

simulations = []
for initial_point in initial_points
    u0 = initial_point
    tspan = (0.0, 10.0)
    #define the problem
    prob = ODEProblem(lokta_volterra!, [u0[1], u0[2]], tspan, original_parameters)

    sol = solve(prob, Tsit5(), saveat=0.1)

    df = DataFrame(sol, [:t, :x1, :x2])
    # restrict the simulation to the bounding box (first time)
    # find the first time the simulation goes outside the bounding box
    #=     outside_training_box = df[df.x1.<min_y1.||df.x1.>max_y_1.||df.x2.<min_y2.||df.x2.>max_y2, :]

        if size(outside_training_box, 1) > 0
            min_time_outside = outside_training_box[1, :t]
            df = df[df.t.<min_time_outside, :]
        end =#

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
        end
end

local_rng = StableRNG(seed)
p_net, st = Lux.setup(local_rng, approximating_neural_network)
model_function = get_uode_model_function(approximating_neural_network, st)

simulations_nodes = []
seed = 0
rng = StableRNG(seed)


for i in axes(initial_points, 1)

    counter_esemble = 0
    simulation_nodes = []
    for ensemble in ensembles

        println("Simulating ensemble $counter_esemble for initial point $i")

        counter_esemble += 1

        initial_point = initial_points[i]
        u0 = [initial_point[1], initial_point[2]]

        times = simulations[i].t
        tspan = extrema(times)

        esemble_simulations = []
        for member in ensemble

            prob = ODEProblem(model_function, u0, tspan, member)
            simulation = solve(prob, Tsit5(), saveat=times)

            if simulation.retcode != ReturnCode.Success
                println("Error in the simulation")
                continue
            end

            ensemble_simulation = DataFrame(simulation, [:t, :x1, :x2])
            push!(esemble_simulations, ensemble_simulation)
        end

        # for each time get the mean and the confidence interval
        y1s = []
        y2s = []
        models = []
        times = []

        counter = 1
        for ensemble_simulation in esemble_simulations
            append!(y1s, ensemble_simulation.x1)
            append!(y2s, ensemble_simulation.x2)
            model_id = (counter_esemble - 1) * 5 + counter
            append!(models, fill(model_id, length(ensemble_simulation.x1)))
            append!(times, ensemble_simulation.t)
            counter += 1
        end

        # dataframe with the mean and the confidence interval
        df = DataFrame(
            t=times,
            x1=y1s,
            x2=y2s,
            model=models
        )

        push!(simulation_nodes, deepcopy(df))
    end
    push!(simulations_nodes, simulation_nodes)
end


#plot in the phase space the first trajectoreis for the MODE nod esenmbles


for ensemble_index in 1:3

    gr()
    plt = Plots.plot()

    #plot the training trajectories for reference
    Plots.scatter!(plt, experimental_data.x1, experimental_data.x2, label="Training data", linestyle=:dash, color=:orange, markersize=3)


    index_initial_point = 1
    i = ensemble_index
    for model in simulations_nodes[index_initial_point][i].model
        tmp_df = simulations_nodes[index_initial_point][i][simulations_nodes[index_initial_point][i].model.==model, :]
        Plots.plot!(plt, tmp_df.x1, tmp_df.x2, label="", xlabel="x", ylabel="y", title="", dpi=600, legend=:bottomright, color=:blue, linewidth=1)
    end
    Plots.scatter!(plt, simulations[index_initial_point].x1[1:1], simulations[index_initial_point].x2[1:1], color=:blue, markersize=6, label="")

    index_initial_point = 2
    i = ensemble_index
    for model in simulations_nodes[index_initial_point][i].model
        tmp_df = simulations_nodes[index_initial_point][i][simulations_nodes[index_initial_point][i].model.==model, :]
        Plots.plot!(plt, tmp_df.x1, tmp_df.x2, label="", xlabel="x", ylabel="y", title="", dpi=600, legend=:bottomright, color=:blue, linewidth=0.5)
    end
    Plots.scatter!(plt, simulations[index_initial_point].x1[1:1], simulations[index_initial_point].x2[1:1], color=:blue, markersize=6, label="")


    index_initial_point = 3
    i = ensemble_index
    for model in simulations_nodes[index_initial_point][i].model
        tmp_df = simulations_nodes[index_initial_point][i][simulations_nodes[index_initial_point][i].model.==model, :]
        Plots.plot!(plt, tmp_df.x1, tmp_df.x2, label="", xlabel="x", ylabel="y", title="", dpi=600, legend=:bottomright, color=:violet, linewidth=0.5)
    end
    Plots.scatter!(plt, simulations[index_initial_point].x1[1:1], simulations[index_initial_point].x2[1:1], color=:violet, markersize=6, label="")

    index_initial_point = 4
    i = ensemble_index
    for model in simulations_nodes[index_initial_point][i].model
        tmp_df = simulations_nodes[index_initial_point][i][simulations_nodes[index_initial_point][i].model.==model, :]
        Plots.plot!(plt, tmp_df.x1, tmp_df.x2, label="", xlabel="x", ylabel="y", title="", dpi=600, legend=:bottomright, color=:violet, linewidth=0.5)
    end
    Plots.scatter!(plt, simulations[index_initial_point].x1[1:1], simulations[index_initial_point].x2[1:1], color=:violet, markersize=6, label="")

    index_initial_point = 5
    i = ensemble_index
    for model in simulations_nodes[index_initial_point][i].model
        tmp_df = simulations_nodes[index_initial_point][i][simulations_nodes[index_initial_point][i].model.==model, :]
        Plots.plot!(plt, tmp_df.x1, tmp_df.x2, label="", xlabel="x", ylabel="y", title="", dpi=600, legend=:bottomright, color=:green, linewidth=0.5)
    end
    Plots.scatter!(plt, simulations[index_initial_point].x1[1:1], simulations[index_initial_point].x2[1:1], color=:green, markersize=6, label="")

    index_initial_point = 6
    i = ensemble_index
    for model in simulations_nodes[index_initial_point][i].model
        tmp_df = simulations_nodes[index_initial_point][i][simulations_nodes[index_initial_point][i].model.==model, :]
        Plots.plot!(plt, tmp_df.x1, tmp_df.x2, label="", xlabel="x", ylabel="y", title="", dpi=600, legend=:bottomright, color=:green, linewidth=0.5)
    end
    Plots.scatter!(plt, simulations[index_initial_point].x1[1:1], simulations[index_initial_point].x2[1:1], color=:green, markersize=6, label="")


    Plots.plot!(plt,
        xguidefont=font(18),    # Increase x-axis label font size
        yguidefont=font(18),
        titlefont=font(16),
        xtickfont=font(12),     # Increase x-axis tick font size
        ytickfont=font(12),
        legendfont=font(10),
        legend=:bottomright,
        title = "MOD ensemble "* string(ensemble_index)
    )

    Plots.plot!(plt, 
        xlims=(boundig_box_vect_field[1]*0.1, boundig_box_vect_field[2]*1.5), 
        ylims=(boundig_box_vect_field[3]*0.1, boundig_box_vect_field[4]*1.5),
    )

    #save figure 
    Plots.savefig(plt, "ensemble_simulations_phase_space" * string(ensemble_index) * ".svg")

end


#Plots.scatter!(plt, simulations[index_initial_point].x1, simulations[index_initial_point].x2, label="Ground truth", linestyle=:dash, color=:red, linewidth=4)

#Plots.scatter!(plt, simulations[index_initial_point].x1, simulations[index_initial_point].x2, label="Ground truth", linestyle=:dash, color=:red, linewidth=4)


#= ### analyze trajectory for trajectory if the ground truth is inside the confidence interval
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
        sd_y1 = simulation_node.sd_x1
        sd_y2 = simulation_node.sd_x2

        n_ensemble = length(ensembles[ensemble_index])
        t_value = quantile(TDist(n_ensemble - 1), 0.975)

        sup_y1 = mean_y_1 .+ t_value .* sd_y1 .* sqrt(1+1/n_ensemble)
        inf_y1 = mean_y_1 .- t_value .* sd_y1 .* sqrt(1+1/n_ensemble)
        sup_y2 = mean_y_2 .+ t_value .* sd_y2 .* sqrt(1+1/n_ensemble)
        inf_y2 = mean_y_2 .- t_value .* sd_y2 .* sqrt(1+1/n_ensemble)

        ground_truth_y_1_in_ci = simulation.x1 .> inf_y1 .&& simulation.x1 .< sup_y1
        ground_truth_y_2_in_ci = simulation.x2 .> inf_y2 .&& simulation.x2 .< sup_y2

        ground_truth_in_ci = ground_truth_y_1_in_ci .& ground_truth_y_2_in_ci

        #for each t, compute time that the trajectory traversed the non captured region
        # of the ground truth
        traversed_non_captured_region = false
        for j in 1:length(simulation.t)
            x_1 = simulation.x1[j]
            x_2 = simulation.x2[j]

            grid_x_1 = Int(round(((x_1 - min_y1) / (max_y_1 - min_y1) * 100)))
            grid_x_2 = Int(round(((x_2 - min_y2) / (max_y2 - min_y2) * 100)))

            grid_x_1 = max(1, min(grid_x_1, 100))
            grid_x_2 = max(1, min(grid_x_2, 100))
        end

        solution_pointwise = DataFrame(t=simulation.t, x1=simulation.x1, x2=simulation.x2, mean_x1=mean_y_1, mean_x2=mean_y_2, sd_x1=sd_y1, sd_x2=sd_y2, sup_x1=sup_y1, inf_x1=inf_y1, sup_x2=sup_y2, inf_x2=inf_y2, ground_truth_in_ci=ground_truth_in_ci)

        percentage = sum(ground_truth_in_ci) / length(ground_truth_in_ci)

        push!(pointwise_solutions, solution_pointwise)
        push!(trajectory_results, percentage)
        push!(traversed_non_captured_regions, traversed_non_captured_region)
    end
    push!(trajectory_results_total, trajectory_results)
    push!(pointwise_solutions_total, pointwise_solutions)
    push!(traversed_non_captured_regions_total, traversed_non_captured_regions)
end
 =#
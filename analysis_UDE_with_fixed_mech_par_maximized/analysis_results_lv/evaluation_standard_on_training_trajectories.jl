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
boundig_box_vect_field = deserialize("../data_generator/lotka_volterra_in_silico_data_bounding_box.jld")
experimental_data = deserialize("../data_generator/lotka_volterra_in_silico_data_no_noise.jld")

#generate randomly the initial points by pertubing by a gaussian of 10% one of the points in the original trajectory (experimental data)
trajectories = unique(experimental_data.traj)

tspan = extrema(experimental_data.t)

in_dim = 2
out_dim = 2

activation_function_fun = gelu
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=0.01)


ensembles = []
for i in 1:10
    maximized_ensemble_folder = "../results_maximized/lv/result_lv$i/results.jld"
    if !isfile(maximized_ensemble_folder)
        continue
    end
    tmp_results = deserialize(maximized_ensemble_folder)
    tmp_ensemble = tmp_results.naive_ensemble_reference
    push!(ensembles, tmp_ensemble)
end

min_y1, max_y_1, min_y2, max_y2 = boundig_box_vect_field

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
            α = 1.3
            δ = 1.8
            du[1] = α * u[1] + û[1]
            du[2] = û[2] - δ * u[2]
        end
end

local_rng = StableRNG(seed)
p_net, st = Lux.setup(local_rng, approximating_neural_network)
model_function = get_uode_model_function(approximating_neural_network, st)

simulations_node_total_df = []
seed = 0
rng = StableRNG(seed)

training_trajectories_folder = "evaluation_on_training_trajectories"
if !isdir(training_trajectories_folder)
    mkdir(training_trajectories_folder)
end

#Plot the simulations against the experimental data

for i in trajectories

    p1_plot = Plots.plot()
    p2_plot = Plots.plot()
    for j in axes(ensembles, 1)
        ensemble = ensembles[j]

        experimental_data_for_trajectory = experimental_data[experimental_data.traj.==i, :]
        initial_point = experimental_data_for_trajectory[1, :]
        initial_point = [initial_point.x1, initial_point.x2]

        u0 = [initial_point[1], initial_point[2]]

        esemble_simulations = []
        for member in ensemble

            prob = ODEProblem(model_function, u0, tspan, member)
            simulation = solve(prob, Vern7(), saveat=0.001, reltol=1e-5, abstol=1e-6)

            if simulation.retcode != ReturnCode.Success
                println("Error in the simulation")
                continue
            end

            ensemble_simulation = DataFrame(simulation, [:t, :x1, :x2])
            push!(esemble_simulations, ensemble_simulation)
        end

        for simulation in esemble_simulations
            Plots.plot!(p1_plot, simulation.t, simulation.x1, label="x", color="blue", leged=false)
            Plots.plot!(p2_plot, simulation.t, simulation.x2, label="y", color="blue", leged=false)
        end

        #plot the experimental data
        Plots.scatter!(p1_plot, experimental_data_for_trajectory.t, experimental_data_for_trajectory.x1, label="", color="red", markersize=5, leged=false, xlabel="t", ylabel="x")
        Plots.scatter!(p2_plot, experimental_data_for_trajectory.t, experimental_data_for_trajectory.x2, label="", color="red", markersize=5, legend=false, xlabel="t", ylabel="y")
    end


    Plots.plot!(p1_plot, 
        xguidefont=font(18),    # Increase x-axis label font size
        yguidefont=font(18),
        titlefont=font(18),
        xtickfont=font(12),     # Increase x-axis tick font size
        ytickfont=font(12),     # Increase y-axis label font size
        legendfont=font(10),
        legend=false,
        leftmargin=30px
    )

    Plots.plot!(p2_plot, 
        xguidefont=font(18),    # Increase x-axis label font size
        yguidefont=font(18),
        titlefont=font(18),
        xtickfont=font(12),     # Increase x-axis tick font size
        ytickfont=font(12),     # Increase y-axis label font size
        legendfont=font(10),
        legend=false,
        leftmargin=30px
    )

    plt = Plots.plot(p1_plot, p2_plot, layout=(1, 2), size=(1200, 400))
    Plots.savefig(plt, joinpath(training_trajectories_folder, "ensemble_training_trajectories_naive_$i.png"))
    Plots.savefig(plt, joinpath(training_trajectories_folder, "ensemble_training_trajectories_naive_$i.svg"))
end




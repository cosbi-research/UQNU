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
seed = 1
rng = StableRNG(seed)

#bounding box of interest 
boundig_box_vect_field = deserialize("../data_generator/damped_oscillator_in_silico_data_bounding_box.jld")
experimental_data = deserialize("../data_generator/damped_oscillator_in_silico_data_no_noise.jld")

#generate randomly the initial points by pertubing by a gaussian of 10% one of the points in the original trajectory (experimental data)
initial_points = []
vcs = [0.01, 0.5, 0.75]


function sample_uniform_2d(a, b, c, d)
    x = rand(rng) * (b - a) + a  # Sample x-coordinate
    y = rand(rng) * (d - c) + c  # Sample y-coordinate
    return (x, y)
end

initial_points = [sample_uniform_2d(boundig_box_vect_field[1], boundig_box_vect_field[2], boundig_box_vect_field[3], boundig_box_vect_field[4]) for _ in 1:trajectory_number]
initial_points_1 = [sample_uniform_2d(boundig_box_vect_field[1] * 10, boundig_box_vect_field[2] * 10, boundig_box_vect_field[3] * 10, boundig_box_vect_field[4] * 10) for _ in 1:10000]
#select the ones not in initial_points
initial_points_1 = filter(p -> !(p[1] > boundig_box_vect_field[1] && p[1] < boundig_box_vect_field[2] && p[2] > boundig_box_vect_field[3] && p[2] < boundig_box_vect_field[4]), initial_points_1)
initial_points_1 = initial_points_1[1:trajectory_number]
#select another one with * 10
initial_points_2 = [sample_uniform_2d(boundig_box_vect_field[1] * 20, boundig_box_vect_field[2] * 20, boundig_box_vect_field[3] * 20, boundig_box_vect_field[4] * 20) for _ in 1:10000]
#select the ones not in initial_points
initial_points_2 = filter(p -> !(p[1] > boundig_box_vect_field[1] * 10 && p[1] < boundig_box_vect_field[2] * 10 && p[2] > boundig_box_vect_field[3] * 10 && p[2] < boundig_box_vect_field[4] * 10), initial_points_2)
initial_points_2 = initial_points_2[1:trajectory_number]

initial_points = vcat(initial_points, initial_points_1, initial_points_2)

#scatter plot of the different intiial points
plt = scatter(initial_points, label="Region 1", xlabel="x1", ylabel="x2", title="Selected initial points", legend=:topright, color=:blue, alpha=0.5)
scatter!(plt, initial_points_1, label="Region 2", color=:red, alpha=0.5)
scatter!(plt, initial_points_2, label="Region 3", color=:green, alpha=0.5)
Plots.plot!(plt,
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),
    legendfont=font(10),
    legend=:bottomright,
)
Plots.savefig(plt, "initial_points_bounding_box.svg")


in_dim = 2
out_dim = 2

activation_function_fun = gelu
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=0.01)

#gound truth model
# parameters for Lotka Volterra and initial state
original_parameters = Float64[1.3, 0.9, 0.8, 1.8]
#function to generate the Data
function damped_ground_truth_function(u)
    α, β = original_parameters
    du = similar(u)
    du[1] = -α * u[1]^3 - β * u[2]^3
    du[2] = β * u[1]^3 - α * u[2]^3
    return du
end

function damped!(du, u, p, t)
    α, β = original_parameters
    du[1] = -α * u[1]^3 - β * u[2]^3
    du[2] = β * u[1]^3 - α * u[2]^3
end

ensembles = []
for i in 1:10
    maximized_ensemble_folder = "../results_maximized/damped/result_damped$i/results.jld"
    if !isfile(maximized_ensemble_folder)
        continue
    end
    tmp_results = deserialize(maximized_ensemble_folder)
    tmp_ensemble = tmp_results.ensemble_reprojected
    push!(ensembles, tmp_ensemble)
end

min_y1, max_y_1, min_y2, max_y2 = boundig_box_vect_field

# generates 100 initial points for the trajectories 

simulations = []
for initial_point in initial_points
    u0 = initial_point
    tspan = (0.0, 100.0)
    #define the problem
    prob = ODEProblem(damped!, [u0[1], u0[2]], tspan, original_parameters)

    sol = solve(prob, Tsit5(), saveat=10.0)

    df = DataFrame(sol, [:t, :x1, :x2])

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
            û = appr_neural_network(u, p.p_net, st)[1]
            du[1] = -p.α*u[1]^3 + û[1]
            du[2] = û[2] - p.α*u[2]^3
        end
end

local_rng = StableRNG(seed)
p_net, st = Lux.setup(local_rng, approximating_neural_network)
model_function = get_uode_model_function(approximating_neural_network, st)

simulations_nodes = []
seed = 0
rng = StableRNG(seed)

simulation_failed = []

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

        member_ensemble = 0
        for member in ensemble

            member_ensemble += 1

            prob = ODEProblem(model_function, u0, tspan, member)
            simulation = solve(prob, Tsit5(), saveat=times)

            if simulation.retcode != ReturnCode.Success
                push!(simulation_failed, (i, counter_esemble, member_ensemble))
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

last_points = []
for i in axes(simulations_nodes, 1)
    for df in simulations_nodes[i]
        if !isempty(df)
            last_point = df[end, [:x1, :x2]]
            last_point = [last_point.x1, last_point.x2]
            push!(last_points, last_point)
        end
    end
end

norm_last_point_original_simulations = [norm([simulations[i].x1[end], simulations[i].x2[end]]) for i in axes(simulations, 1)]
norm_last_point_original_simulations = min.(100.0, norm_last_point_original_simulations)

proximity_index = 0
norm_last_point = [norm(last_point) for last_point in last_points[(proximity_index*trajectory_number*5+1):((proximity_index+1)*5*trajectory_number)]]
norm_last_point = min.(100.0, norm_last_point)
norm_last_point_original = norm_last_point_original_simulations[(proximity_index*trajectory_number+1):((proximity_index+1)*trajectory_number)]

#histogram of the last points
plt = histogram(log10.(norm_last_point), bins=-3:0.1:2.1, label="NODE", xlabel="Last state norm (log10)", ylabel="Density", title="Region 1", legend=:topright, color=:blue, alpha=0.5, normalize=true)
histogram!(log10.(norm_last_point_original), bins=-3:0.1:(2.1), label="Ground truth", color=:grey, alpha=0.5, normalize=true)
Plots.plot!(plt,
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),
    legendfont=font(10),
    legend=:topleft,
)
#save the plot 
Plots.savefig(plt, "last_state_norm_region_1.svg")

proximity_index = 1
norm_last_point = [norm(last_point) for last_point in last_points[(proximity_index*trajectory_number*5+1):((proximity_index+1)*5*trajectory_number)]]
norm_last_point = min.(100.0, norm_last_point)
norm_last_point_original = norm_last_point_original_simulations[(proximity_index*trajectory_number+1):((proximity_index+1)*trajectory_number)]

#histogram of the last points
plt = histogram(log10.(norm_last_point), bins=-3:0.1:2.1, label="NODE", xlabel="Last state norm (log10)", ylabel="Density", title="Region 2", legend=:topright, color=:blue, alpha=0.5, normalize=true)
histogram!(log10.(norm_last_point_original), bins=-3:0.1:2.1, label="Ground truth", color=:grey, alpha=0.5, normalize=true)
Plots.plot!(plt,
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),
    legendfont=font(10),
    legend=:topleft,
)
#save the plot 
Plots.savefig(plt, "last_state_norm_region_2.svg")

proximity_index = 2
norm_last_point = [norm(last_point) for last_point in last_points[(proximity_index*trajectory_number*5+1):((proximity_index+1)*5*trajectory_number)]]
norm_last_point = min.(100.0, norm_last_point)
norm_last_point_original = norm_last_point_original_simulations[(proximity_index*trajectory_number+1):((proximity_index+1)*trajectory_number)]

#histogram of the last points
plt = histogram(log10.(norm_last_point), bins=-3:0.1:2.1, label="NODE", xlabel="Last state norm (log10)", ylabel="Density", title="Region 3", legend=:topright, color=:blue, alpha=0.5, normalize=true)
histogram!(log10.(norm_last_point_original), bins=-3:0.1:2.1, label="Ground truth", color=:grey, alpha=0.5, normalize=true)
Plots.plot!(plt,
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(16),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),
    legendfont=font(10),
    legend=:topleft,
)
#save the plot
Plots.savefig(plt, "last_state_norm_region_3.svg")

#get the failed simulations between 100 and 200
simulation_failed_second_region = filter(p -> p[1] >= 100 && p[1] < 200, simulation_failed)
total_second_region = 100 * 5
percentage_failed_second_region = length(simulation_failed_second_region) / total_second_region * 100

simulation_failed_third_region = filter(p -> p[1] >= 200 && p[1] <= 300, simulation_failed)
total_third_region = 100 * 5
percentage_failed_third_region = length(simulation_failed_third_region) / total_third_region * 100

#write the failed simulations percentage to a file
open("simulation_failed_percentage.txt", "w") do file
    write(file, "Percentage of failed simulations in the second region: $percentage_failed_second_region%\n")
    write(file, "Percentage of failed simulations in the third region: $percentage_failed_third_region%\n")
end
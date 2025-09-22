cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions

gr()

integrator = Vern7()
abstol = 1e-6
reltol = 1e-5

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
training_data_structure = deserialize("../data_generator/lotka_volterra_training_data_structure_err_1.jld")

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
    tmp_ensemble = tmp_results.ensemble_reprojected
    push!(ensembles, tmp_ensemble)
end

naive_ensembles = []
for i in 1:10
    maximized_ensemble_folder = "../results_maximized/lv/result_lv$i/results.jld"
    if !isfile(maximized_ensemble_folder)
        continue
    end
    tmp_results = deserialize(maximized_ensemble_folder)
    tmp_ensemble = tmp_results.naive_ensemble_reference
    push!(naive_ensembles, tmp_ensemble)
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

tspan = extrema(training_data_structure.solution_dataframes[1].t)
uode_derivative_function = get_uode_model_function(approximating_neural_network, st)
prob_uode_pred = ODEProblem{true}(uode_derivative_function, Array(training_data_structure.solution_dataframes[1][1, 2:(end-1)]), tspan)

training_trajectories_folder = "evaluation_on_training_trajectories"
if !isdir(training_trajectories_folder)
    mkdir(training_trajectories_folder)
end

function model_simulation(θ, t, trajectory, initial_states)
  if trajectory == 1
    trajectory_sol = solve(
      remake(
        prob_uode_pred;
        p=θ,
        tspan=extrema(t),
        u0=initial_states[1, :, 1]
      ),
      integrator;
      saveat=t,
      reltol=reltol,
      abstol=abstol
    )
  elseif trajectory == 2
    trajectory_sol = solve(
      remake(
        prob_uode_pred;
        p=θ,
        tspan=extrema(t),
        u0=initial_states[2, :, 1]
      ),
      integrator;
      saveat=t,
      reltol=reltol,
      abstol=abstol
    )
  elseif trajectory == 3
    trajectory_sol = solve(
      remake(
        prob_uode_pred;
        p=θ,
        tspan=extrema(t),
        u0=initial_states[3, :, 1]
      ),
      integrator;
      saveat=t,
      reltol=reltol,
      abstol=abstol
    )
  end

  if trajectory_sol.retcode != :Success
    return Inf
  end

  return Array(trajectory_sol)
end

function costFunctionOnSingleTraj(par, i)
  original_solutions = training_data_structure.solution_dataframes[i]
  original_times = original_solutions.t
  simulation = model_simulation(par, original_times, i, initial_states)

  if simulation == Inf
    return Inf
  end

  simulation = simulation[:, 1:end]
  cost_trajectory = 1 / size(original_solutions, 1) * (sum((simulation[1, :] - original_solutions.x1) .^ 2 ./ training_data_structure.max_oscillations[i][1]^2) + sum((simulation[2, :] - original_solutions.x2) .^ 2 ./ training_data_structure.max_oscillations[i][2]^2))

  return cost_trajectory
end

function costFunction(par)
  cost = 0.0
  for i in 1:3
    cost += costFunctionOnSingleTraj(par, i)
  end
  return cost
end


# to obtain initial states
trained_naive_ensemble = deserialize("../training_UDE_results/lv/ensemble_results_model_1_with_seed.jld")
single_parameter_training = trained_naive_ensemble[1]
initial_states = deepcopy(single_parameter_training.training_res.u0)

costs = []
costs_naive = []

for ensemble in ensembles 
    for element in ensemble
        push!(costs, costFunction(element))
    end
end

for ensemble in naive_ensembles 
    for element in ensemble
        push!(costs_naive, costFunction(element))
    end
end

#plot as an histogram the costs
plt = Plots.plot(dpi=1000)
all_costs_log = vcat(log10.(costs), log10.(costs_naive))
bins = range(floor(minimum(all_costs_log)), ceil(maximum(all_costs_log)), length=30)  # adjust length as needed

Plots.histogram!(plt, log10.(costs), label="MOD", color="blue", alpha=0.5, bins=bins)
Plots.histogram!(plt, log10.(costs_naive), label="Standard", color="red", alpha=0.5, bins=bins)
#vline in 10^-3
Plots.vline!(plt, [log10(0.001)], color="black", linestyle=:dash, label ="")
Plots.plot!(plt, 
    xguidefont=font(18),    # Increase x-axis label font size
    yguidefont=font(18),
    titlefont=font(18),
    xtickfont=font(12),     # Increase x-axis tick font size
    ytickfont=font(12),     # Increase y-axis label font size
    legendfont=font(10),
    xlabel="log10(loss)",
    ylabel="",
    legend = :topright
)

Plots.savefig(plt, joinpath(training_trajectories_folder, "costs_training_trajectories.png"))
Plots.savefig(plt, joinpath(training_trajectories_folder, "costs_training_trajectories.svg"))

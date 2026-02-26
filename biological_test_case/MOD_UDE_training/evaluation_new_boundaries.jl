cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions
using Logging, StatsBase

loglevel = Logging.Info
global_logger(ConsoleLogger(stderr, loglevel))
debug_folder = "debug_ca"
result_folder = "result_ca"

maxiters = 1000

observables = [4,]

#parse the starting point index 
starting_point_index = 10
#TODO ripristinare
#starting_point_index = parse(Int, ARGS[1])
@info "Starting point index: " starting_point_index

result_folder = result_folder * string(starting_point_index)
debug_folder = debug_folder * string(starting_point_index)

#if the folder doesn't exist, create it
if !isdir(result_folder)
  mkdir(result_folder)
end


if !isdir(debug_folder)
  mkdir(debug_folder)
end

rng = Random.default_rng()
Random.seed!(rng, 123)

ensemble_selected = starting_point_index
ensemble_interval_begin = (starting_point_index - 1) * 5 + 1
ensemble_interval_end = (starting_point_index - 1) * 5 + 1 + 4
# Extract one random number from the interval
starting_point_index = rand(ensemble_interval_begin:ensemble_interval_end)


include("ConfidenceEllipse.jl")
using .ConfidenceEllipse

include("diagnostic_training_set.jl")
using .diagnostic_training_set

include("configurations_ca.jl")

################################### loads the data ##############################################
training_data_structure = deserialize("../data_generator/cell_apoptosis_training_data_structure.jld")

################################### loads the single-result #####################################
#load the result of the single-parameter training
trained_ensemble = deserialize("./training_UDE_results/ensemble_results.jld")
#sort them according to status and validation likelihood
#get only the non-failed ones 
trained_ensemble = filter(res -> res.status == "success", trained_ensemble)

trained_ensemble = [res for res in trained_ensemble]
single_parameter_training = trained_ensemble[starting_point_index]

lower_bounds = single_parameter_training.lower_parameter_boundaries
upper_bounds = single_parameter_training.upper_parameter_boundaries

parameters = deepcopy(single_parameter_training.training_res.p)
original_parameters_ude = deepcopy(parameters.ode_par .* (upper_bounds .- lower_bounds) .+ lower_bounds)
parameters.ode_par .= 1.0

naive_ensemble_reference = [res.training_res.p for res in trained_ensemble[ensemble_interval_begin:ensemble_interval_end]]
naive_ensemble_reference_initial_states = [res.training_res.u0 for res in trained_ensemble[ensemble_interval_begin:ensemble_interval_end]]

################################### separate in the required structure ##########################
p_net, st = Lux.setup(rng, approximating_neural_network)
tspan = extrema(training_data_structure.solution_dataframes[1].t)

lower_bounds = single_parameter_training.lower_parameter_boundaries
upper_bounds = single_parameter_training.upper_parameter_boundaries

uode_derivative_function = get_uode_model_function(approximating_neural_network, st, original_parameters_ude)
vector_field_function = get_vector_field_function(approximating_neural_network, st, original_parameters_ude)

prob_uode_pred = ODEProblem{true}(uode_derivative_function, Array(training_data_structure.solution_dataframes[1][1, 2:end]), tspan)

################### DUBBIO SU COSA SIA ###########################
initial_states = deepcopy(single_parameter_training.training_res.u0)

################################### instantiate the module for the analysis OOD #################
include("out_of_domain_variability_nd.jl")
using .out_of_domain_variability_nd

#get experimental points]
experimental_points = []
df = training_data_structure.solution_dataframes[1]
for j in 1:size(df, 1)
  global experimental_points
  experimental_points = push!(experimental_points, collect(df[j, 2:end]))
end

function model_simulation(θ, t, trajectory, initial_states, tmp_observables=nothing, integrator=integrator, sensealg=sensealg, prob_uode_pred=prob_uode_pred)
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
      abstol=abstol,
      sensealg=sensealg,
      maxiters=maxiters
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
      abstol=abstol,
      sensealg=sensealg,
      maxiters=maxiters
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
      abstol=abstol,
      sensealg=sensealg,
      maxiters=maxiters
    )
  end

  if trajectory_sol.retcode != :Success
    return Inf
  end

  res = Array(trajectory_sol)
  if tmp_observables != nothing
    return res[tmp_observables, :]
  else
    return res
  end
end

#predict in the new scenario (basically)
extremes = training_data_structure.solution_dataframes[1].t[[1, end]]
times = collect(extremes[1]:0.01:extremes[2])

initial_states_survival = deepcopy(initial_states)
initial_states_survival[end-1] = initial_states_survival[end-1]/ 10.0

simulation = model_simulation(parameters, times, 1, initial_states_survival)

Plots.plot(times, simulation[1, :], label="x1", color="blue")
Plots.plot!(times, simulation[2, :], label="x2", color="orange")
Plots.plot!(times, simulation[3, :], label="x3", color="green")
Plots.plot!(times, simulation[4, :], label="x4", color="violet")
Plots.plot!(times, simulation[5, :], label="x5", color="brown")
Plots.plot!(times, simulation[6, :], label="x6", color="pink")
Plots.plot!(times, simulation[7, :], label="x7", color="cyan")
Plots.plot!(times, simulation[8, :], label="x8", color="black")

ranges = [extrema(simulation[i, :]) for i in 1:size(simulation, 1)]

# add 10% to the ranges
ranges_extended = [(r[1] - 0.5 * abs(r[2] - r[1]), r[2] + 0.5 * abs(r[2] - r[1])) for r in ranges]



bounding_box_df = deserialize("../data_generator/cell_apoptosis_silico_data_bounding_box.jld")
#do not take the bounding box from the reference, but try and simulate the model in the new configurations and
#take that bounding box
ranges = [(bounding_box_df[i, 2], bounding_box_df[i, 3]) for i in 1:size(bounding_box_df, 1)]

#print the two ranges on a text file
open("ranges_$ensemble_selected.txt", "w") do f
  println(f, "Ranges old: ")
  for i in 1:length(ranges)
    println(f, "Range for variable $i: ", ranges[i])
  end
  println(f, "Ranges extended: ")
  for i in 1:length(ranges_extended)
    println(f, "Range for variable $i: ", ranges_extended[i])
  end
end


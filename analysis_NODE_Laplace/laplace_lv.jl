cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, Distributions
using Logging, StatsBase

loglevel = Logging.Info
global_logger(ConsoleLogger(stderr, loglevel))
result_folder = "result_lv_laplace"

maxiters = 10000

#parse the starting point index 
#starting_point_index = 1
starting_point_index = parse(Int, ARGS[1])
@info "Starting point index: " starting_point_index

result_folder = result_folder * string(starting_point_index)

#if the folder doesn't exist, create it
if !isdir(result_folder)
  mkdir(result_folder)
end

rng = Random.default_rng()
Random.seed!(rng, 0)

ensemble_selected = starting_point_index
ensemble_interval_begin = (starting_point_index - 1) * 5 + 1
ensemble_interval_end = (starting_point_index - 1) * 5 + 1 + 4
# Extract one random number from the interval
starting_point_index = rand(ensemble_interval_begin:ensemble_interval_end)


include("ConfidenceEllipse.jl")
using .ConfidenceEllipse

include("diagnostic_training_set.jl")
using .diagnostic_training_set

include("configurations_lv.jl")

################################### loads the data ##############################################
training_data_structure = deserialize("./data_generator/lotka_volterra_training_data_structure_err_1.jld")


bounding_box = deserialize("./data_generator/lotka_volterra_in_silico_data_bounding_box.jld")
#generate a grid 100*100
xrange_bounding_box = range(bounding_box[1], bounding_box[2], length=10)
yrange_bounding_box = range(bounding_box[3], bounding_box[4], length=10)

################################### loads the single-result #####################################
#load the result of the single-parameter training
trained_ensemble = deserialize("training_NODE_results/lv/ensemble_results_model_1_with_seed.jld")
#sort them according to status and validation likelihood
trained_ensemble = [res for res in trained_ensemble]
single_parameter_training = trained_ensemble[starting_point_index]

parameters = deepcopy(single_parameter_training.training_res.p)

naive_ensemble_reference = [res.training_res.p for res in trained_ensemble[ensemble_interval_begin:ensemble_interval_end]]

################################### separate in the required structure ##########################
p_net, st = Lux.setup(rng, approximating_neural_network)
tspan = extrema(training_data_structure.solution_dataframes[1].t)

uode_derivative_function = get_uode_model_function(approximating_neural_network, st)
vector_field_function = get_vector_field_function(approximating_neural_network, st)

prob_uode_pred = ODEProblem{true}(uode_derivative_function, Array(training_data_structure.solution_dataframes[1][1, 2:(end-1)]), tspan)
initial_states = deepcopy(single_parameter_training.training_res.u0)

################################### instantiate the module for the analysis OOD #################
include("out_of_domain_variability.jl")
using .out_of_domain_variability

#get experimental points]
experimental_points = []
for i in 1:3
  df = training_data_structure.solution_dataframes[i]
  for j in 1:size(df, 1)
    global experimental_points
    experimental_points = push!(experimental_points, collect(df[j, 2:(end-1)]))
  end
end

ood_analyzer = out_of_domain_variability.out_of_domain_var(xrange_bounding_box, yrange_bounding_box, vector_field_function, lotka_volterra_gound_truth, experimental_points, [], [], [])
out_of_domain_variability.computeGroundTruth(ood_analyzer)
out_of_domain_variability.computePoints(ood_analyzer)

out_of_domain_points = ood_analyzer.points

#get the distance between the training domain and the out of domain points
distances_from_training_set = out_of_domain_variability.getOutOfDomainDistance(ood_analyzer)

function model_simulation(θ, t, trajectory, initial_states, integrator=integrator, sensealg=sensealg, prob_uode_pred=prob_uode_pred)
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

  return Array(trajectory_sol)
end

times = training_data_structure.solution_dataframes[1].t
original_times = deepcopy(times)
times = times[1:1:end]

function get_Hessian_likelihood(parameters_to_consider, times, initial_states, training_data_structure, sigma)

  #first trajectory
  sensitivity_matrix_first_trajectory = Zygote.jacobian(p -> model_simulation(p, times, 1, initial_states), parameters_to_consider)[1]
  #second trajectory
  sensitivity_matrix_second_trajectory = Zygote.jacobian(p -> model_simulation(p, times, 2, initial_states), parameters_to_consider)[1]
  #third trajectory
  sensitivity_matrix_third_trajectory = Zygote.jacobian(p -> model_simulation(p, times, 3, initial_states), parameters_to_consider)[1]

  sensitivity_matrix = vcat(sensitivity_matrix_first_trajectory, sensitivity_matrix_second_trajectory, sensitivity_matrix_third_trajectory)

  multiplicative_factor_array = repeat(training_data_structure.max_oscillations[1].^2 * sigma^2, outer=size(times, 1))
  multiplicative_factor_array = vcat(multiplicative_factor_array, repeat(training_data_structure.max_oscillations[2].^2 * sigma^2, outer=size(times)))
  multiplicative_factor_array = vcat(multiplicative_factor_array, repeat(training_data_structure.max_oscillations[3].^2 * sigma^2, outer=size(times)))

  multiplicative_factor_matrix = Diagonal(multiplicative_factor_array)
  multiplicative_factor_matrix = inv(multiplicative_factor_matrix)

  hessian = sensitivity_matrix' * multiplicative_factor_matrix * sensitivity_matrix

  return Symmetric(2 .* hessian)
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

function costFunctionOnSingleTraj(par, i, integrator, sensealg, prob_uode_tmp)
  original_solutions = training_data_structure.solution_dataframes[i]
  original_times = original_solutions.t
  simulation = model_simulation(par, original_times, i, initial_states, integrator, sensealg, prob_uode_tmp)

  if simulation == Inf
    return Inf
  end

  simulation = simulation[:, 1:end]

  cost_trajectory = 1 / size(original_solutions, 1) * (sum((simulation[1, :] - original_solutions.x1) .^ 2 ./ training_data_structure.max_oscillations[i][1]^2) + sum((simulation[2, :] - original_solutions.x2) .^ 2 ./ training_data_structure.max_oscillations[i][2]^2))

  return cost_trajectory
end

function costFunction(par, integrator, sensealg, prob_uode_tmp)
  cost = 0.0
  for i in 1:3
    cost += costFunctionOnSingleTraj(par, i, integrator, sensealg, prob_uode_tmp)
  end
  return cost
end

function getValidationCost(pars, initial_states)
  cost = 0.0
  for i in 1:3
    #tmp_times = vcat(0, training_data_structure.validation_dataframes[i].t)
    tmp_times = training_data_structure.solution_dataframes[i].t
    simulation = model_simulation(pars, tmp_times, i, initial_states)

    if simulation == Inf
      return Inf
    end

    #simulation = simulation[:, 2:end]
    #cost_trajectory = 1 / size(training_data_structure.validation_dataframes[i], 1) * (sum((simulation[1, :] - training_data_structure.validation_dataframes[i].x1) .^ 2 ./ training_data_structure.max_oscillations[i][1]^2) + sum((simulation[2, :] - training_data_structure.validation_dataframes[i].x2) .^ 2 ./ training_data_structure.max_oscillations[i][2]^2))
    cost_trajectory = 1 / size(training_data_structure.solution_dataframes[i], 1) * (sum((simulation[1, :] - training_data_structure.solution_dataframes[i].x1) .^ 2 ./ training_data_structure.max_oscillations[i][1]^2) + sum((simulation[2, :] - training_data_structure.solution_dataframes[i].x2) .^ 2 ./ training_data_structure.max_oscillations[i][2]^2))
    cost += cost_trajectory
  end
  return cost
end

function getSigmaSquaredMse(pars, initial_states)
  cost = 0.0
  data_size = 0
  for i in 1:3
    #tmp_times = vcat(0, training_data_structure.validation_dataframes[i].t)
    tmp_times = training_data_structure.solution_dataframes[i].t
    simulation = model_simulation(pars, tmp_times, i, initial_states)

    if simulation == Inf
      return Inf
    end

    #simulation = simulation[:, 2:end]
    #cost_trajectory = 1 / size(training_data_structure.validation_dataframes[i], 1) * (sum((simulation[1, :] - training_data_structure.validation_dataframes[i].x1) .^ 2 ./ training_data_structure.max_oscillations[i][1]^2) + sum((simulation[2, :] - training_data_structure.validation_dataframes[i].x2) .^ 2 ./ training_data_structure.max_oscillations[i][2]^2))
    cost_trajectory = (sum((simulation[1, :] - training_data_structure.solution_dataframes[i].x1) .^ 2 ./ training_data_structure.max_oscillations[i][1]^2) + sum((simulation[2, :] - training_data_structure.solution_dataframes[i].x2) .^ 2 ./ training_data_structure.max_oscillations[i][2]^2))
    cost += cost_trajectory
    data_size += size(training_data_structure.solution_dataframes[i], 1) * 2
  end
  return cost / data_size
end

parameter_populations = [parameters .+ 0.0]
validation_cost_threshold = 1e-3

# Create a lock for thread-safe operations
population_to_add_candidates = []

#evaluate what's happening on the vector field

original_parameters = parameters .+ 0.0

initial_cost = getValidationCost(parameters, initial_states)
if initial_cost > validation_cost_threshold
  ensemble_interval_begin = (ensemble_selected - 1) * 5 + 1
  ensemble_interval_end = (ensemble_selected - 1) * 5 + 1 + 4
  remaining_indexes = collect(ensemble_interval_begin:ensemble_interval_end)

  for new_index in remaining_indexes
    global starting_point_index = new_index
    res = trained_ensemble[starting_point_index]
    global parameters = deepcopy(res.training_res.p)
    global initial_cost = getValidationCost(parameters, initial_states)

    if initial_cost <= validation_cost_threshold
      @info "The validation cost is below the threshold, I found a starting point"
      break
    end
  end
end

#MSE estimator of the σ of the noise on the data
sigma_squared_mse = getSigmaSquaredMse(parameters, initial_states)

# compute the squared norm of the parameters
parameters_squared_norm = sum(parameters .^ 2)

#compute the hessian of the likelihood
hessian = get_Hessian_likelihood(parameters, times, initial_states, training_data_structure, sqrt(sigma_squared_mse))

# maximizing the marginal likelihood
range_lambda = [1000, 10000, 100000, 1000000, 10000000, 100000000]

function epsilon(lambda)
  k = length(parameters)
  identity_matrix = I(k)
  F = cholesky(Symmetric(hessian .+ lambda * identity_matrix))
  logdet = 2.0 * sum(log, diag(F.L))
  return k/2*log(lambda) - 0.5*logdet - 0.5*parameters_squared_norm*lambda
end

optimal_lambda = range_lambda[argmax(epsilon.(range_lambda))]


#get the posterior covariance matrix
posterior_covariance_matrix = inv(hessian .+ optimal_lambda * I(length(parameters)))
posterior_covariance_matrix = Symmetric(posterior_covariance_matrix)

using Distributions
distribution_covariance_matrix = MvNormal(collect(parameters), posterior_covariance_matrix)

#sample from the distribution
number_of_samples = 1000
sampled_parameters = rand(distribution_covariance_matrix, number_of_samples)

#evaluate the cost of the sampled parameters and keep those that are below the threshold
population_to_add_candidates = []
for i in 1:number_of_samples
  pars = sampled_parameters[:, i]
  tmp_pars = deepcopy(parameters)
  tmp_pars .= pars
  cost = getValidationCost(tmp_pars, initial_states)
  println(cost)
    global population_to_add_candidates
    population_to_add_candidates = push!(population_to_add_candidates, tmp_pars)
end

result_file = result_folder * "/ensemble_laplace_approximation"*string(ensemble_selected)*".jld"

serialize(result_file, population_to_add_candidates)

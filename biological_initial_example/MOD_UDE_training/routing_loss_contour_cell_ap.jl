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
starting_point_index = 1
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

include("configurations_ca.jl")

################################### loads the data ##############################################
training_data_structure = deserialize("../data_generator/cell_apoptosis_training_data_structure.jld")


bounding_box_df = deserialize("../data_generator/cell_apoptosis_silico_data_bounding_box.jld")

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
original_parameters_ude = parameters.ode_par .* (upper_bounds .- lower_bounds) .+ lower_bounds
parameters.ode_par .= 1.0

naive_ensemble_reference = [res.training_res.p for res in trained_ensemble[ensemble_interval_begin:ensemble_interval_end]]

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

ranges = [(bounding_box_df[i, 2], bounding_box_df[i, 3]) for i in 1:size(bounding_box_df, 1)]

ood_analyzer = out_of_domain_variability_nd.out_of_domain_var_nd(ranges, 100, vector_field_function, ca_gound_truth, experimental_points, [], [], 8)
out_of_domain_variability_nd.computePoints(ood_analyzer)
out_of_domain_variability_nd.computeGroundTruth(ood_analyzer)

out_of_domain_points = ood_analyzer.points

#get the distance between the training domain and the out of domain points
distances_from_training_set = out_of_domain_variability_nd.getOutOfDomainDistance(ood_analyzer)

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


function get_Hessian(parameters_to_consider, times, initial_states, training_data_structure)

  #first trajectory
  sampled_times = sort(times[sample(eachindex(times), 10; replace=false)])
  sensitivity_matrix_first_trajectory = Zygote.jacobian(p -> model_simulation(p, sampled_times, 1, initial_states), parameters_to_consider)[1] .* parameters_to_consider'

  sensitivity_matrix = vcat(sensitivity_matrix_first_trajectory,)

  multiplicative_factor_array = repeat(training_data_structure.max_oscillations, outer=size(sampled_times, 1))

  multiplicative_factor_matrix = Diagonal(multiplicative_factor_array)

  hessian = sensitivity_matrix' * multiplicative_factor_matrix * sensitivity_matrix

  return hessian
end

function get_Hessian_not_proportional(parameters_to_consider, times, initial_states, training_data_structure)

  #first trajectory
  sampled_times = sort(times[sample(eachindex(times), 10; replace=false)])

  sensitivity_matrix_first_trajectory = Zygote.jacobian(p -> model_simulation(p, sampled_times, 1, initial_states), parameters_to_consider)[1]

  sensitivity_matrix = vcat(sensitivity_matrix_first_trajectory,)

  multiplicative_factor_array = repeat(training_data_structure.max_oscillations, outer=size(sampled_times, 1))

  multiplicative_factor_matrix = Diagonal(multiplicative_factor_array)

  hessian = sensitivity_matrix' * multiplicative_factor_matrix * sensitivity_matrix

  return hessian
end

function getEigenDempositionHessian(par, times, initial_states, training_data_structure, parameter_index)
  hessian = get_Hessian(par, times, initial_states, training_data_structure)
  #the rows and the columns corresponding to the parameters I shouldn't consider are put to zero
  hessian[Not(parameter_index), :] .= 0.0
  hessian[:, Not(parameter_index)] .= 0.0
  eigen_decomposition = eigen(Symmetric(hessian))
  return eigen_decomposition
end

function getEigenDempositionHessianNotProportional(par, times, initial_states, training_data_structure, parameter_index)
  hessian = get_Hessian_not_proportional(par, times, initial_states, training_data_structure)
  #the rows and the columns corresponding to the parameters I shouldn't consider are put to zero
  hessian[Not(parameter_index), :] .= 0.0
  hessian[:, Not(parameter_index)] .= 0.0
  eigen_decomposition = eigen(Symmetric(hessian))
  return eigen_decomposition
end
 
function getSampling(par, sample_number, times, initial_states, training_data_structure, parameter_index)
  eigenDecomposition = getEigenDempositionHessian(par, times, initial_states, training_data_structure, parameter_index)
  eigenvalues = eigenDecomposition.values
  eigenvalues = max.(eigenvalues, 1e-20)

  covariance_matrix = Diagonal(1 ./ eigenvalues)

  covariance_matrix[covariance_matrix.<10] .= 0.0

  covariance_matrix = min.(covariance_matrix, 1)

  #covariance_matrix[covariance_matrix .< 1] .= 0.0

  #sample the directions with a multivarate normal distribution
  directions = rand(rng, MvNormal(zeros(length(eigenvalues)), covariance_matrix), sample_number)
  #sample back to the original coordinate system of the parameters
  directions_original_system = eigenDecomposition.vectors * directions
  directions_original_system[Not(parameter_index), :] .= 0.0
  #define the epsilon to move on the directions
  epsilon = 0.01

  #move along the directions
  parameters_moved = [par .* (1 .+ epsilon .* directions_original_system[:, i]) for i in axes(directions_original_system, 2)]
  return parameters_moved[1]
end

function costFunctionOnSingleTraj(par, i)
  original_solutions = training_data_structure.solution_dataframes[i]
  original_times = original_solutions.t
  simulation = model_simulation(par, original_times, i, initial_states)

  if simulation == Inf
    return Inf
  end
  simulation = simulation[:, 1:end]

  cost_trajectory = 1 / size(original_solutions, 1) * (sum([sum(simulation[j, :] .- original_solutions[!, j+1]) .^ 2 ./ training_data_structure.max_oscillations[j]^2 for j in observables]))

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

  cost_trajectory = 1 / size(original_solutions, 1) * (sum([sum(simulation[j, :] .- original_solutions[!, j+1]) .^ 2 ./ training_data_structure.max_oscillations[j]^2 for j in observables]))

  return cost_trajectory
end

function costFunction(par, integrator, sensealg, prob_uode_tmp)
  cost = costFunctionOnSingleTraj(par, 1, integrator, sensealg, prob_uode_tmp)
  return cost
end

function getReoptimizedParameters(par, integrator, sensealg)

  @info "Re-optimizing the parameters"

  adtype = Optimization.AutoZygote()

  original_parameters_ude_fixed = original_parameters_ude .* par.ode_par

  uode_derivative_function_fixed = get_uode_fixed_model_function(approximating_neural_network, st, original_parameters_ude_fixed)
  prob_uode_pred_fixed = ODEProblem{true}(uode_derivative_function_fixed, Array(training_data_structure.solution_dataframes[1][1, 2:end]), tspan)

  optf = Optimization.OptimizationFunction((x, p) -> costFunction(x, integrator, sensealg, prob_uode_pred_fixed), adtype)
  optprob = Optimization.OptimizationProblem(optf, par)


  function print_and_save_cost(state, l, best_solution, best_objective)

    if l < best_objective[1]
      best_objective[1] = l
      best_solution[1] = deepcopy(state.u)
    end

    println("***************** Optimization Iteration: ", state.iter, " | Cost: ", l)
    return false  # Return false to allow optimization to continue
  end

  best_solution = [par]
  best_objective = [Inf]

  #opt = OptimizationOptimisers.Adam(0.0000001)
  opt = Optim.GradientDescent()

  callback_function = (θ, l) -> print_and_save_cost(θ, l, best_solution, best_objective)

  res = Optimization.solve(optprob, opt, callback=callback_function, maxiters=20)

  solution = deepcopy(best_solution[1])
  objective = deepcopy(best_objective[1])
  try
    optprob = Optimization.OptimizationProblem(optf, solution)
    res = Optimization.solve(optprob, Optim.LBFGS(), callback=callback_function, maxiters=20)
  catch e
    @warn "Error in the second optimization" e
  end

  solution = deepcopy(best_solution[1])
  objective = deepcopy(best_objective[1])

  @info "Optimization terminated with cost: " objective
  return solution
end

#################################################### arrivato qui #########################################################
function flatten(min_x_square, min_y_square, max_x_square, max_y_square, rows, cols)
  flattened_positions = []
  for x in min_x_square:max_x_square
    for y in min_y_square:max_y_square
      # Convert 2D (x, y) position to 1D flattened index
      index = (x - 1) * cols + y
      push!(flattened_positions, index)
    end
  end
  return flattened_positions
end

function getVarianceGradient(par, points, total_ensemble, ood_analyzer)

  selected_indexes = sample(1:size(points, 2), 25, replace=false)
  selected_tmp_points = points[:, selected_indexes]

  ensemble_predictions = out_of_domain_variability_nd.get_ensemble_predictions(ood_analyzer, total_ensemble, selected_tmp_points)
  current_prediction = out_of_domain_variability_nd.get_ensemble_predictions(ood_analyzer, [par], selected_tmp_points)[1]

  #get the mean of the prediction at each point
  stacked_predictions = cat(ensemble_predictions..., dims=3)
  mean_prediction = mean(stacked_predictions, dims=3)
  mean_prediction = mean_prediction[:, :, 1]

  #First variable of the ensemble
  vector_field = x -> vector_field_function(selected_tmp_points, x)
  gradient_vector_field = jacobian(x -> vector_field(x), par)[1]

  #@infiltrate

  #rows: gradient of the vector field in a point
  numpoints = size(selected_tmp_points, 2)
  num_variables = size(current_prediction, 1)
  gradient_in_single_points = [2 / size(total_ensemble, k) .* norm(current_prediction[k, :] .- mean_prediction[k, :]) .* gradient_vector_field[k:num_variables:end,:] for k in 1:size(current_prediction, 1)] 

  variances = [zeros(size(current_prediction, 2)) for _ in 1:size(current_prediction, 1)]
  for k in 1:size(current_prediction, 1)
    variances[k] = zeros(size(current_prediction[1, :]))
    for i in axes(stacked_predictions, 3)
      variances[k] .+= 1 / size(total_ensemble, 1) .* abs2.(stacked_predictions[k, :, i] .- mean_prediction[k, :])
    end
  end

  gradient_in_single_points_y = [gradient_in_single_points[k] ./ variances[k] for k in 1:size(current_prediction, 1)]

  gradient_variance = reduce(+, [mean(gradient_in_single_points_y[k], dims=1) for k in 1:size(current_prediction, 1)])

  return gradient_variance
end

function getCovarianceGradient_n(par, points, total_ensemble, ood_analyzer)

  selected_indexes = sample(1:length(points), 25, replace=false)
  selected_tmp_points = points[:, selected_indexes]

  ensemble_predictions = out_of_domain_variability.get_ensemble_predictions(ood_analyzer, total_ensemble, selected_tmp_points)
  current_prediction = out_of_domain_variability.get_ensemble_predictions(ood_analyzer, [par], selected_tmp_points)[1]
  # (n × Npts)

  # stack ensemble -> Y: (n × Npts × M)
  Y = cat(ensemble_preds...; dims=3)
  n, _, M = size(Y)

  # mean across ensemble: μ (n × Npts)
  μ = dropdims(mean(Y; dims=3); dims=3)

  # Jacobian of current prediction wrt parameters:
  # pred_flat(par) = vec(current_pred) of length n*Npts
  function pred_flat(pv)
    pred = out_of_domain_variability.get_ensemble_predictions(
      ood_analyzer, [pv], pts
    )[1]
    return vec(pred)
  end
  J = ForwardDiff.jacobian(pred_flat, par_vec)          # (n*Npts) × P
  G = reshape(J, n, Npts, P)                            # n × Npts × P

  α = (M - 1) / (M^2)   # matches your scaling

  # total_gradient: rows = points, cols = parameters
  total_gradient = zeros(eltype(J), Npts, P)
  determinant = zeros(eltype(Y), Npts)

  for j in 1:Npts
    # covariance Σ at point j
    Σ = zeros(eltype(Y), n, n)
    for i in 1:M
      δ = Y[:, j, i] .- μ[:, j]
      Σ .+= (δ * δ') / M
    end

    # regularize like a PSD covariance (helps avoid negative det from numerics)
    Σ = Symmetric(Σ + ϵ * I)

    detΣ = det(Matrix(Σ))
    determinant[j] = detΣ

    # factorization to compute Σ^{-1} * dΣ stably (avoid explicit inv)
    F = cholesky(Σ; check=false)

    δcur = current_pred[:, j] .- μ[:, j]  # (n,)

    # for each parameter k, build dΣ_k and use:
    # d(detΣ)/dθ_k = detΣ * tr(Σ^{-1} dΣ_k)
    for k in 1:P
      gk =  G[:, j, k]  # (n,)

      # dΣ_k ≈ α * (gk*δcur' + δcur*gk')  (your 2D cross-term generalization)
      dΣ = α .* (gk * δcur' .+ δcur * gk')

      X = F \ (F' \ dΣ)  # X = Σ^{-1} dΣ
      total_gradient[j, k] = detΣ * tr(X)
    end
  end

  @info "Volumes of ellipsoids " * string(determinant)
  @info "Mean volumes of ellipsoids " * string(mean(abs.(determinant)))

  # match your sign(det) behavior (broadcast over rows)
  total_gradient .*= sign.(determinant)

  # filter out rows with NaN/Inf (same as your code)
  mask = all(.!isinf.(total_gradient) .& .!isnan.(total_gradient), dims=2)
  total_gradient = total_gradient[vec(mask), :]

  # same output type/shape: 1×P row
  res = mean(total_gradient, dims=1)

  return res
end

function getNextPointDirection(par, times, initial_states, training_data_structure, out_of_domain_points, current_ensemble, ood_analyzer, parameter_index, step_size, trajectories, iterator)
  eigenDecomposition = getEigenDempositionHessianNotProportional(par, times, initial_states, training_data_structure, parameter_index)
  eigenvalues = eigenDecomposition.values
  eigenvalues = max.(eigenvalues, 1e-20)

  #get the sloppy directions
  sloppy_eigenvectors = eigenDecomposition.vectors[:, eigenvalues.<1e-1]

  gradient_variance = nothing
  if length(current_ensemble) < 8 || iterator < 100
    gradient_variance = getVarianceGradient(par, out_of_domain_points, current_ensemble, ood_analyzer)
  else
    gradient_variance = getCovarianceGradient_n(par, out_of_domain_points, current_ensemble, ood_analyzer)
  end

  gradient_variance = collect(vec(gradient_variance))
  #put to zero the non selected indexes (invert the selection)
  @info "Norm of the gradient variance " * string(sqrt(sum(abs2.(gradient_variance))))

  gradient_variance[Not(parameter_index)] .= 0.0

  @info "Norm of the gradient variance " * string(sqrt(sum(abs2.(gradient_variance))))

  #generate randomly 300 integrer between 1 and the length of the parameters
  #random_indexes = sort(rand(rng, 1:length(par), 300))
  #put to zero the non selected indexes (invert the selection)
  #to avoid that the gradient is based on just one direction
  #gradient_distance[Not(random_indexes)] .= 0.0

  #project the gradient on the sloppy directions
  projection = sloppy_eigenvectors * sloppy_eigenvectors' * gradient_variance
  projection[Not(parameter_index)] .= 0.0

  #normalize the direction
  #direction = projection ./ sqrt(sum(projection .^ 2))

  #gradient clipping 
  if sqrt(sum(projection .^ 2)) > 10
    projection = projection ./ sqrt(sum(projection .^ 2)) .* 10
  end

  return projection
end

function getValidationCost(pars, initial_states)
  cost = 0.0
  for i in 1:1
    #tmp_times = vcat(0, training_data_structure.validation_dataframes[i].t)
    tmp_times = training_data_structure.solution_dataframes[i].t
    simulation = model_simulation(pars, tmp_times, i, initial_states)

    #temp print the simulation against the solution dataframe 
    #= plt = Plots.plot(tmp_times, simulation', label=["Sim 1" "Sim 2" "Sim 3" "Sim 4" "Sim 5" "Sim 6" "Sim 7" "Sim 8"], title="Simulation vs Solution Traj " * string(i))
    #plot the solution dataframe 
    for j in 1:(size(training_data_structure.solution_dataframes[i], 2)-1)
      Plots.scatter!(plt, training_data_structure.solution_dataframes[i].t, training_data_structure.solution_dataframes[i][!, j+1], label="Data " * string(j))
    end =#

    #display(plt)

    if simulation == Inf
      return Inf
    end

    #simulation = simulation[:, 2:end]
    #cost_trajectory = 1 / size(training_data_structure.validation_dataframes[i], 1) * (sum((simulation[1, :] - training_data_structure.validation_dataframes[i].x1) .^ 2 ./ training_data_structure.max_oscillations[i][1]^2) + sum((simulation[2, :] - training_data_structure.validation_dataframes[i].x2) .^ 2 ./ training_data_structure.max_oscillations[i][2]^2))
    cost += 1 / size(training_data_structure.solution_dataframes[i], 1) * (sum([sum(simulation[j, :] .- training_data_structure.solution_dataframes[i][!, j+1]) .^ 2 ./ training_data_structure.max_oscillations[j]^2 for j in observables]))
  end
  return cost
end

#naive implementation of monte-carlo sampling
number_iterations_for_trajectory = 50
validation_cost_threshold = 1e-3

parameter_populations = [parameters .+ 0.0]

# Create a lock for thread-safe operations
population_to_add_candidates = []

#evaluate what's happening on the vector field

original_parameters = parameters .+ 0.0
trajectories = []
number_of_trajectories = 4
variances = []
total_cicps = []
reprojections = []
total_validation_costs = []

selected_points = out_of_domain_points
#selected_points = selected_points[:, 1:10]

#each trajectory moves along a specific subspace
total_parameter_indexes = 1:(length(parameters)-length(parameters.ode_par))
sampled_indexes_trajectories = [sort(sample(total_parameter_indexes, 200; replace=false)) for i in 1:number_of_trajectories]
#put always the last two indexes

sampled_indexes_trajectories = [vcat(sampled_indexes_trajectories[i], collect(length(parameters)-length(parameters.ode_par):length(parameters))) for i in 1:number_of_trajectories]


#check the validation cost before, if it's more than the threshold exit immediately

@info "Checking initial validation cost"

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


if initial_cost > validation_cost_threshold
  results = (
    status="failed",
    ensemble_original=nothing,
    ensemble_reprojected=nothing,
    naive_ensemble_number=ensemble_selected,
    naive_ensemble_reference=naive_ensemble_reference
  )
  serialize(result_folder * "/results.jld", results)
else
  for traj_number in 1:number_of_trajectories

    try

      iteration_performed = 0

      current_trajectory = []
      validation_costs = []
      current_ensemble = [parameters .+ 0.0]
      current_variances = []
      cicps = []
      reprojection_trajectory = 0

      while iteration_performed < 3

        current_trajectory = []
        validation_costs = []
        current_ensemble = [parameters .+ 0.0]
        current_variances = []
        cicps = []
        reprojection_trajectory = 0

        sampled_index_for_trajectory = sampled_indexes_trajectories[traj_number]
        step_size = 1.0
        min_step_size = 1e-5
        max_step_size = 10^2

        iteration_performed = 0

        for iterator in 1:number_iterations_for_trajectory

          iteration_performed += 1

          @info "Beginning iteration: " iterator
          iteration_original_parameters = deepcopy(current_ensemble[end])

          new_suggestion = nothing
          if iterator == 1
            new_suggestion = getSampling(current_ensemble[end], 1, times, initial_states, training_data_structure, sampled_index_for_trajectory)
          else

            #sample randomly 10 number over 1:100
            sampled_indexes = sample(2:100, 20, replace=false)
            push!(sampled_indexes, 1)
            sampled_indexes = sort(sampled_indexes)
            tmp_times = times
            new_suggestion_direction = getNextPointDirection(current_ensemble[end], tmp_times, initial_states, training_data_structure, selected_points, current_ensemble, ood_analyzer, sampled_index_for_trajectory, step_size, nothing, iterator)


            @info "Norm of th direction: " * string(sqrt(sum(new_suggestion_direction .^ 2)))

            #new_proposed_parameters = current_ensemble[end] .* (1 .+ step_size .* new_suggestion_direction)
            new_proposed_parameters = current_ensemble[end] .+ step_size .* new_suggestion_direction
            validation_cost = Inf
            try
              validation_cost = getValidationCost(new_proposed_parameters, initial_states)
            catch
              @warn "Error in integration with step " step_size
              validation_cost = Inf
            end

            previous_validation_cost = validation_costs[end]

            while validation_cost < validation_cost_threshold && (validation_cost - previous_validation_cost) < 0.000001
              @info "The validation cost is decreasing too much, I increase the step size: step_size=" step_size
              step_size = step_size * 2

              if step_size > max_step_size
                step_size = step_size / 2
                break
              end

              #move along the directions
              #new_proposed_parameters = current_ensemble[end] .* (1 .+ step_size .* new_suggestion_direction)
              new_proposed_parameters = current_ensemble[end] .+ step_size .* new_suggestion_direction

              vaidation_cost = Inf
              try
                validation_cost = getValidationCost(new_proposed_parameters, initial_states)
              catch
                @warn "Error in integration with step " step_size
                validation_cost = Inf
              end
            end

            while validation_cost > validation_cost_threshold || (validation_cost - previous_validation_cost) > 0.00001

              @info "The validation cost is increasing too much, I reduce the step size: step_size=" step_size
              step_size = step_size / 2
              #move along the directions
              #new_proposed_parameters = current_ensemble[end] .* (1 .+ step_size .* new_suggestion_direction)
              new_proposed_parameters = current_ensemble[end] .+ step_size .* new_suggestion_direction

              validation_cost = Inf
              try
                validation_cost = getValidationCost(new_proposed_parameters, initial_states)
              catch
                @warn "Error in integration with step " step_size
                validation_cost = Inf
              end

              if step_size < min_step_size
                step_size = min_step_size
                @error "The step size is too small, I stop the optimization"
                break
              end
            end

            @info "I move along the direction: step_size=" step_size
            new_suggestion = new_proposed_parameters

          end

          validation_cost = Inf
          try
            validation_cost = getValidationCost(new_suggestion, initial_states)
          catch
            @warn "Error in integration with step " step_size
            validation_cost = Inf
          end

          @info "Validation cost: " validation_cost
          if validation_cost > validation_cost_threshold
            @warn "The validation cost is above the threshold,i reoptimize the parameters"
            push!(validation_costs, validation_cost)
            original_validation_cost = validation_cost
            try
              new_suggestion = getReoptimizedParameters(iteration_original_parameters, integrator, sensealg)
              validation_cost = getValidationCost(new_suggestion, initial_states)
              push!(validation_costs, -1.0)
              step_size = 0.1
              reprojection_trajectory = reprojection_trajectory + 1
              if validation_cost > 0.9 * validation_cost_threshold
                @error "The validation cost is not low enough, I stop the optimization " * string(validation_cost)
                throw("Validation too high")
              end
              @info "Validation cost after reoptimization: " validation_cost
            catch e
              @error "Exception " * string(e)
              @warn "Error during parameter reprojection trying going backward in the trajectory"
              backward_looking = 1
              optimized = false

              while optimized == false
                @info "trying to go backward " * string(backward_looking) * " times"
                try

                  if backward_looking == length(current_trajectory) || backward_looking > 10
                    @error "I cannot go backward anymore, I stop the optimization"
                    push!(current_trajectory, iteration_original_parameters)
                    break
                  end

                  new_suggestion = getReoptimizedParameters(current_trajectory[end-backward_looking], integrator, sensealg)
                  validation_cost = getValidationCost(new_suggestion, initial_states)
                  push!(validation_costs, -1.0)
                  step_size = 0.1
                  reprojection_trajectory = reprojection_trajectory + 1
                  optimized = true
                  if validation_cost > 0.9 * validation_cost_threshold
                    @error "The validation cost is not low enough, I stop the optimization"
                    #throw an exception
                    throw("Validation too high")
                    #push!(current_trajectory, iteration_original_parameters)
                    #break
                  end
                catch
                  @warn "Error during parameter reprojection also with stiff configurations"
                  validation_cost = original_validation_cost
                  optimized = false
                  backward_looking += 1
                end
              end
            end


            if validation_cost > validation_cost_threshold
              @error "The validation cost is above the threshold, I stop the optimization"
              push!(current_trajectory, iteration_original_parameters)
              break
            end
          end

          push!(validation_costs, validation_cost)

          if iterator % 2 == 1 || iterator == number_iterations_for_trajectory
            @info "saving point in current trajectory"
            push!(current_trajectory, new_suggestion)
          end

          #select only the elements that increase the variance
          average_variance_before = nothing
          if length(current_ensemble) == 1
            average_variance_before = 0.0
          else
            #average_variance_before = out_of_domain_variability.getAverageVariance(ood_analyzer, current_ensemble)
            average_variance_before = out_of_domain_variability_nd.getVarianceInPoints(ood_analyzer, current_ensemble, selected_points)
          end

          @info "Average variance before: " average_variance_before


          #select the ensemble for the population
          current_ensemble = [original_parameters]
          for traj_counter in axes(trajectories, 1)
            current_ensemble = push!(current_ensemble, trajectories[traj_counter][end])
          end
          #current_ensemble = select_current_ensemble(current_ensemble, original_parameters)
          current_ensemble = push!(current_ensemble, new_suggestion)

          #distance from the original point
          distance = sqrt(sum((new_suggestion .- original_parameters) .^ 2))
          @info "Distance from the original point: " distance

          if iterator % 10 == 1
            @info "Distance from the original point: " distance
            abs2.(new_suggestion .- original_parameters)
            #plot the histogram 
            plt = Plots.histogram(log10.(abs2.(new_suggestion .- original_parameters)[sampled_index_for_trajectory]), bins=100, title="Distance from the original point", xlabel="Distance", ylabel="Frequency")
            distance_file = debug_folder * "/distance_" * string(traj_number) * "_" * string(iterator) * ".png"
            Plots.savefig(plt, distance_file)
          end

          #variance = out_of_domain_variability.getAverageVariance(ood_analyzer, current_ensemble)
          variance = out_of_domain_variability_nd.getVarianceInPoints(ood_analyzer, current_ensemble, selected_points)
          @info "Iteration: " iterator
          @info "Population size: " length(parameter_populations)
          @info "Current ensemble size: " length(current_ensemble)
          @info "Average variance: " variance
          @info "Physical parameters values: " * string(new_suggestion.ode_par)

          push!(current_variances, variance)

          #computes the analysis for out-of-domain generalization
          @debug "At the end of the iteration, I compute the summary statistcs over the Out of domain region"
          if size(current_ensemble, 1) > 8 && iterator % 10 == 1
            if loglevel <= Logging.Info
              global data_to_save = []
              out_of_domain_analysis = out_of_domain_variability_nd.getOutOfDomainAnalysis(ood_analyzer, current_ensemble)
              push!(cicps, out_of_domain_analysis.cicp)

              #= @info "Visualizing the OOD statistics "
              out_of_domain_plots = out_of_domain_variability.plotOutOfDomainAnalysis(ood_analyzer, out_of_domain_analysis)

              cicp_file_name = debug_folder * "/cicp_" * string(traj_number) * "_" * string(iterator) * ".png"
              Plots.savefig(out_of_domain_plots.cicp_out_of_domain, cicp_file_name)

              ellipse_file_name = debug_folder * "/ellipse_" * string(traj_number) * "_" * string(iterator) * ".png"
              Plots.savefig(out_of_domain_plots.areas_out_of_domain, ellipse_file_name)

              @info "Visualizing the training set trajectories"
              validation_traj_1_file_name = debug_folder * "/validation_traj_1_iter_" * string(iterator) * ".png"
              traj_training_plot = diagnostic_training_set.printValidationPlots(new_suggestion, original_times, 1, initial_states, prob_uode_pred, integrator, reltol, abstol, training_data_structure)
              Plots.savefig(traj_training_plot, validation_traj_1_file_name)


              validation_traj_2_file_name = debug_folder * "/validation_traj_2_iter_" * string(iterator) * ".png"
              traj_training_plot = diagnostic_training_set.printValidationPlots(new_suggestion, original_times, 2, initial_states, prob_uode_pred, integrator, reltol, abstol, training_data_structure)
              Plots.savefig(traj_training_plot, validation_traj_2_file_name)

              validation_traj_3_file_name = debug_folder * "/validation_traj_3_iter_" * string(iterator) * ".png"
              traj_training_plot = diagnostic_training_set.printValidationPlots(new_suggestion, original_times, 3, initial_states, prob_uode_pred, integrator, reltol, abstol, training_data_structure)
              Plots.savefig(traj_training_plot, validation_traj_3_file_name) =#

            end
          end
        end
      end

      push!(trajectories, current_trajectory)
      push!(variances, current_variances)
      push!(total_cicps, cicps)
      push!(reprojections, reprojection_trajectory)
      push!(total_validation_costs, validation_costs)
    catch e
      #print the error
      @error e
      @error "Error in trajectory " traj_number
    end
  end

  @info "Saving the resulting ensemble"

  #get the ensemble 
  ensemble = [parameters .+ 0.0]
  for trajectory in trajectories
    push!(ensemble, trajectory[end])
  end

  @info "Reprojecting the ensemble members"

  #reprojection of the ensemble if it is possible

  new_ensemble = []

  for ensemble_it in axes(ensemble, 1)
    new_member = deepcopy(ensemble[ensemble_it])
    if ensemble_it > 1
      try
        new_member = getReoptimizedParameters(ensemble[ensemble_it], integrator, sensealg)
        validation_cost = getValidationCost(new_member, initial_states)
        push!(total_validation_costs[ensemble_it-1], validation_cost)
      catch e
        @warn "Error in the reprojection of the ensemble member " * string(ensemble_it) * ": " * string(e)
        +
      end
    end
    push!(new_ensemble, new_member)
    #reoptimize
  end

  #saving the ensemble with the actual physicial parameters
  for i in axes(ensemble, 1)
    ensemble[i].ode_par = original_parameters_ude .* ensemble[i].ode_par
  end

  #saving the reprojected enesmble with the actual physical parameters
  for i in axes(new_ensemble, 1)
    new_ensemble[i].ode_par = original_parameters_ude .* new_ensemble[i].ode_par
  end

  #saving the original ensemble with the actual physical parameters
  for i in axes(naive_ensemble_reference, 1)
    naive_ensemble_reference[i].ode_par .= naive_ensemble_reference[i].ode_par .* (upper_bounds .- lower_bounds) .+ lower_bounds
  end
  @info "Saving the results"

  results = (
    status="success",
    ensemble_original=ensemble,
    ensemble_reprojected=new_ensemble,
    naive_ensemble_number=starting_point_index,
    reprojections=reprojections,
    validation_costs=total_validation_costs,
    naive_ensemble_reference=naive_ensemble_reference
  )

  #save the results
  serialize(result_folder * "/results.jld", results)
  serialize(result_folder * "/trajectories.jld", trajectories)
  serialize(result_folder * "/variances.jld", variances)
  serialize(result_folder * "/cicps.jld", total_cicps)
end

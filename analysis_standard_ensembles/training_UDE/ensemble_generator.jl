cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates

########################################################## reads the command line arguments ##########################################################
learning_rate_adam =  parse(Float64, ARGS[1])
neural_network_dimension = parse(Int64, ARGS[2])
activation_function = parse(Int64, ARGS[3])
#1: tanh, 2: relu, 3: sigmoid, 4: gelu
regularization = parse(Int64, ARGS[4])
#1: l1, 2: l2, 3: elastic_net, 4: early_stopping
regularization_coefficient_1 = parse(Float64, ARGS[5])
regularization_coefficient_2 = parse(Float64, ARGS[6])
#gain for initialization
gain = parse(Float64, ARGS[7])
# error level, 0: no error, 1: error
error_level = parse(Int64, ARGS[8])
number_ensembles = parse(Int64, ARGS[9])
number_threads = parse(Int64, ARGS[10])
output_folder = ARGS[11]
# 1: lotka volterra, 2: lorenz, 3: damped oscillator
model = parse(Int64, ARGS[12])
hyper_ms_segment = parse(Int64, ARGS[13])
hyper_ms_lambda = parse(Float64, ARGS[14])

#print the arguments to check their correctness
println("learning_rate_adam = ", learning_rate_adam)
println("neural_network_dimension = ", neural_network_dimension)
println("activation_function = ", activation_function)
println("regularization = ", regularization)
println("regularization_coefficient_1 = ", regularization_coefficient_1)
println("regularization_coefficient_2 = ", regularization_coefficient_2)
println("gain = ", gain)
println("error_level = ", error_level)
println("number_ensembles = ", number_ensembles)
println("number_threads = ", number_threads)
println("output_folder = ", output_folder)
println("model = ", model)
println("hyper_ms_segment = ", hyper_ms_segment)
println("hyper_ms_lambda = ", hyper_ms_lambda)
flush(stdout)

########################################################### general configurations ######################################################################
rng = Random.default_rng()
Random.seed!(rng, 0)

# numerical integrator
integrator = Tsit5()
abstol = 1e-7
reltol = 1e-6
sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))

#load the data and split into training and validation
datafile = "./data_generator/lotka_volterra_in_silico_data"
in_dim = 2
out_dim = 2
if model == 2
  in_dim = 3
  out_dim = 1
  println("Considering model lorenz")
  datafile = "./data_generator/lorenz_in_silico_data"
elseif model == 3
  in_dim = 2
  out_dim = 2
  println("Considering model damped oscillator")
  datafile = "./data_generator/damped_oscillator_in_silico_data"
end

if error_level == 1
  datafile = datafile * "_no_noise.jld"
else
  datafile = datafile * "_noisy.jld"
end

get_uode_model_function = nothing
#lotka volterra
if model == 1
  #model function
  get_uode_model_function = function (appr_neural_network, state, lower_boundaries, upper_boundaries)
    #generates the function with the parameters
    f(du, u, p, t) =
      let appr_neural_network = appr_neural_network, st = state, lower_boundaries=lower_boundaries, upper_boundaries=upper_boundaries
        #@infiltrate
        û = appr_neural_network(u, p.p_net, st)[1]
        @inbounds du[1] = (p.α*(upper_boundaries.α - lower_boundaries.α) + lower_boundaries.α)*u[1] + û[1]
        @inbounds du[2] = û[2] - (p.δ*(upper_boundaries.δ - lower_boundaries.δ) + lower_boundaries.δ)*u[2]
      end
  end
  original_ude_parameters = [1.3, 1.8]
  upper_parameter_boundaries_vec = original_ude_parameters * 2.0
  lower_parameter_boundaries_vec = original_ude_parameters * 0.5

  upper_parameter_boundaries = ComponentVector(α=upper_parameter_boundaries_vec[1], δ=upper_parameter_boundaries_vec[2])
  lower_parameter_boundaries = ComponentVector(α=lower_parameter_boundaries_vec[1], δ=lower_parameter_boundaries_vec[2])
elseif model == 2
  #model function
  get_uode_model_function = function (appr_neural_network, state, lower_boundaries, upper_boundaries)
    #generates the function with the parameters
    f(du, u, p, t) =
      let appr_neural_network = appr_neural_network, st = state, lower_boundaries=lower_boundaries, upper_boundaries=upper_boundaries
        û = appr_neural_network(u, p.p_net, st)[1]
        @inbounds du[1] = (p.σ * (upper_boundaries.σ - lower_boundaries.σ) + lower_boundaries.σ) *(u[2] - u[1]) 
        @inbounds du[2] = u[1] * ((p.r*(upper_boundaries.r - lower_boundaries.r) + lower_boundaries.r)-u[3])- u[2]
        @inbounds du[3] = û[1]
      end
  end
  original_ude_parameters = [10.0, 28.0]
  upper_parameter_boundaries_vec = original_ude_parameters * 2.0
  lower_parameter_boundaries_vec = original_ude_parameters * 0.5

  upper_parameter_boundaries = ComponentVector(σ=upper_parameter_boundaries_vec[1], r=upper_parameter_boundaries_vec[2])
  lower_parameter_boundaries = ComponentVector(σ=lower_parameter_boundaries_vec[1], r=lower_parameter_boundaries_vec[2])
else 
  #model function
  get_uode_model_function = function (appr_neural_network, state, lower_boundaries, upper_boundaries)
    #generates the function with the parameters
    f(du, u, p, t) =
      let appr_neural_network = appr_neural_network, st = state, lower_boundaries=lower_boundaries, upper_boundaries=upper_boundaries
        û = appr_neural_network(u, p.p_net, st)[1]
        @inbounds du[1] = -(p.α* (upper_boundaries.α - lower_boundaries.σ) + lower_boundaries.σ) * u[1]^3 + û[1]
        @inbounds du[2] = û[2] - (p.α * (upper_boundaries.α - lower_boundaries.σ) + lower_boundaries.σ) * u[2]^3
      end
  end
  original_ude_parameters = [0.1] 
  upper_parameter_boundaries_vec = original_ude_parameters * 2.0
  lower_parameter_boundaries_vec = original_ude_parameters * 0.5

  upper_parameter_boundaries = ComponentVector(α=upper_parameter_boundaries_vec[1])
  lower_parameter_boundaries = ComponentVector(α=lower_parameter_boundaries_vec[1])
end 

original_ude_parameters = ones(length(original_ude_parameters))

solutions_dataframe = deserialize(datafile)
trajectories = unique(solutions_dataframe.traj)

solution_dataframe_1 = solutions_dataframe[solutions_dataframe.traj .== trajectories[1], :]
size_df = size(solution_dataframe_1)[1]
size_validation = round(Int, 0.2 * size_df)

#generates the random mask for the training and the valudation data set
mask = shuffle(2:size_df)
validation_mask = mask[1:size_validation]
training_mask = pushfirst!(mask[size_validation+1:end], 1)

training_dataframe_1 = solution_dataframe_1[training_mask, :]
validation_dataframe_1 = solution_dataframe_1[validation_mask, :]

training_dataframe_1 = sort(training_dataframe_1, [:t])
validation_dataframe_1 = sort(validation_dataframe_1, [:t])

solution_dataframe_2 = solutions_dataframe[solutions_dataframe.traj .== trajectories[2], :]
size_df = size(solution_dataframe_2)[1]
size_validation = round(Int, 0.2 * size_df)

training_dataframe_2 = solution_dataframe_2[training_mask, :]
validation_dataframe_2 = solution_dataframe_2[validation_mask, :]
training_dataframe_2 = sort(training_dataframe_2, [:t])
validation_dataframe_2 = sort(validation_dataframe_2, [:t])

solution_dataframe_3 = solutions_dataframe[solutions_dataframe.traj .== trajectories[3], :]
size_df = size(solution_dataframe_3)[1]

size_validation = round(Int, 0.2 * size_df)

training_dataframe_3 = solution_dataframe_3[training_mask, :]
validation_dataframe_3 = solution_dataframe_3[validation_mask, :]
training_dataframe_3 = sort(training_dataframe_3, [:t])
validation_dataframe_3 = sort(validation_dataframe_3, [:t])

########################################## NODE neural network ############################################################
activation_function_fun = [tanh, relu, sigmoid, gelu][activation_function]
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=gain)
approximating_neural_network = Lux.Chain(
  Lux.Dense(in_dim, neural_network_dimension, activation_function_fun),
  Lux.Dense(neural_network_dimension, neural_network_dimension, activation_function_fun),
  Lux.Dense(neural_network_dimension, out_dim),
)


max_oscillations_1 = [maximum(training_dataframe_1[1:end, i]) - minimum(training_dataframe_1[1:end, i]) for i in 2:(size(training_dataframe_1, 2)-1)]

max_oscillations_2 = [maximum(training_dataframe_2[1:end, i]) - minimum(training_dataframe_2[1:end, i]) for i in 2:(size(training_dataframe_2, 2)-1)]

max_oscillations_3 = [maximum(training_dataframe_3[1:end, i]) - minimum(training_dataframe_3[1:end, i]) for i in 2:(size(training_dataframe_3, 2)-1)]

max_oscillations = [max_oscillations_1, max_oscillations_2, max_oscillations_3]

solution_dataframes = [solution_dataframe_1, solution_dataframe_2, solution_dataframe_3]
training_dataframes = [training_dataframe_1, training_dataframe_2, training_dataframe_3]
validation_dataframes = [validation_dataframe_1, validation_dataframe_2, validation_dataframe_3]

function train(approximating_neural_network, training_dataframes, validation_dataframes, solution_dataframes, rng, learning_rate_adam, integrator, abstol, reltol, sensealg, seed)
  
  current_time = Dates.now()

  local_rng = StableRNG(seed)

  tmp_neural_network = deepcopy(approximating_neural_network)
  p_net, st = Lux.setup(local_rng, tmp_neural_network)

  #extract randomly the intial points for the ude parameters
  ude_parameters = vec(rand(rng, length(original_ude_parameters)))
  
  #UDE derivative function
  tspan = extrema(solution_dataframes[1].t)
  uode_derivative_function = get_uode_model_function(approximating_neural_network, st, lower_parameter_boundaries, upper_parameter_boundaries)
  prob_uode_pred = ODEProblem{true}(uode_derivative_function, Array(solution_dataframes[1, 2:(end-1)]), tspan)

  training_data_1 = Array(training_dataframes[1][!, 2:(end-1)])'
  training_data_2 = Array(training_dataframes[2][!, 2:(end-1)])'
  training_data_3 = Array(training_dataframes[3][!, 2:(end-1)])'

  training_datas = [training_data_1, training_data_2, training_data_3]

  #loss function for the comparison among the parameters and the predictions
  function loss_function(data, pred, max_oscillation)
    original_cost = sum(abs2.(data .- pred) ./ abs2.(max_oscillation))
    return 1 / size(data, 2) * original_cost
  end

  function loss_on_trajectory(θ, hyperparameters, i)
    # Multiple shooting predictions
    sols = [
      solve(
        remake(
          prob_uode_pred;
          p=θ.p,
          tspan=(hyperparameters.tsteps[first(rg)], hyperparameters.tsteps[last(rg)]),
          #u0=θ.u0[:, first(rg)]
          #u0=θ.u0[i, :, first(rg)]
          u0=u0_original[i, :, first(rg)]
        ),
        integrator;
        saveat=tsteps[rg],
        reltol=reltol,
        abstol=abstol,
        sensealg=sensealg,
        verbose=true
      ) for rg in hyperparameters.ranges
    ]

    # Abort and return infinite loss if one of the integrations failed
    for k in 1:length(sols)
      if size(Array(sols[k]))[2] != length(hyperparameters.ranges[k])
        return Inf
      end
    end

    group_predictions = Array.(sols)
    # SE component of the cost function
    curr_loss = 0
    for (j, rg) in enumerate(ranges)
      training_data_range = training_datas[i][:, rg]
      model_prediction_range = group_predictions[j]
      max_oscillation = max_oscillations[i]
      curr_loss += loss_function(training_data_range, model_prediction_range, max_oscillation)
    end

    #continuity penalization
    for (j, rg) in enumerate(ranges)
      if j == 1
        continue
      end

      u0 = group_predictions[j-1][:, end]
      u1 = group_predictions[j][:, 1]
      curr_loss += hyperparameters.continuity_cost * sum(abs2.(u0 .- u1) ./ abs2.(max_oscillations[i]))
    end

    return curr_loss, hyperparameters
  end

  function loss(θ, hyperparameters)
    loss_1 = loss_on_trajectory(θ, hyperparameters, 1)
    loss_2 = loss_on_trajectory(θ, hyperparameters, 2)
    loss_3 = loss_on_trajectory(θ, hyperparameters, 3)
    curr_loss = loss_1[1] + loss_2[1] + loss_3[1]
    
    weights = vcat([vec(θ.p.p_net[layer_name].weight) for layer_name in keys(θ.p.p_net)]...)
    if regularization == 1
      curr_loss = curr_loss + regularization_coefficient_1 * sum(abs, weights)
    elseif regularization == 2
      curr_loss = curr_loss + regularization_coefficient_1 * sum(abs2, weights)
    elseif regularization == 3
      curr_loss = curr_loss + regularization_coefficient_1 * sum(abs2, weights) + regularization_coefficient_2 * sum(abs, weights)
    end
    
    return curr_loss, hyperparameters
  end

  function loss2_on_trajectory(θ, i)
    # Multiple shooting predictions

    tspan = extrema(training_dataframes[i].t)

    sol = Array(solve(
      remake(
        prob_uode_pred;
        p=θ.p,
        tspan=tspan,
        #u0=θ.u0[i, :, 1]
        u0=u0_original[i, :, 1]
      ),
      integrator;
      saveat=training_dataframes[1].t,
      reltol=reltol,
      abstol=abstol,
      sensealg=sensealg,
      verbose=true
    ))

    if size(sol) != size(training_datas[i])
      return Inf
    else
      max_oscillation = max_oscillations[i]
      return loss_function(training_datas[i], sol, max_oscillation)
    end    
    
  end

  function loss2(θ)
    loss_1 = loss2_on_trajectory(θ, 1)
    loss_2 = loss2_on_trajectory(θ, 2)
    loss_3 = loss2_on_trajectory(θ, 3)

    curr_loss = loss_1 + loss_2 + loss_3
    
    weights = vcat([vec(θ.p.p_net[layer_name].weight) for layer_name in keys(θ.p.p_net)]...)
    if regularization == 1
      curr_loss = curr_loss + regularization_coefficient_1 * sum(abs, weights)
    elseif regularization == 2
      curr_loss = curr_loss + regularization_coefficient_1 * sum(abs2, weights)
    elseif regularization == 3
      curr_loss = curr_loss + regularization_coefficient_1 * sum(abs2, weights) + regularization_coefficient_2 * sum(abs, weights)
    end
    
    return curr_loss
  end

  function callback2(θ, l, validation_losses, epochs, best_on_validation)
    epoch = maximum(epochs)+1
    epochs = push!(epochs, epoch)

    #check if it takes more than 20 minutes
    if Dates.now() - current_time > Dates.Minute(150)
      error("Time limit reached")
    end

    #validation prediction 
    prob_uode_pred_tmp_1 = remake(prob_uode_pred, u0=θ.u.u0[1, :, 1])
    model_prediction_1 = solve(prob_uode_pred_tmp_1, integrator, abstol=abstol, reltol=reltol, saveat=validation_dataframes[1].t, p=θ.u.p)
    max_oscillation = max_oscillations[1]
    val_loss_1 = loss_function(Array(validation_dataframes[1][:, 2:(end-1)])', model_prediction_1, max_oscillation)

    prob_uode_pred_tmp_2 = remake(prob_uode_pred, u0=θ.u.u0[2, :, 1])
    model_prediction_2 = solve(prob_uode_pred_tmp_2, integrator, abstol=abstol, reltol=reltol, saveat=validation_dataframes[1].t, p=θ.u.p)
    max_oscillation = max_oscillations[2]
    val_loss_2 = loss_function(Array(validation_dataframes[2][:, 2:(end-1)])', model_prediction_2, max_oscillation)

    prob_uode_pred_tmp_3 = remake(prob_uode_pred, u0=θ.u.u0[3, :, 1])
    model_prediction_3 = solve(prob_uode_pred_tmp_3, integrator, abstol=abstol, reltol=reltol, saveat=validation_dataframes[1].t, p=θ.u.p)
    max_oscillation = max_oscillations[3]
    val_loss_3 = loss_function(Array(validation_dataframes[3][:, 2:(end-1)])', model_prediction_3, max_oscillation)

    val_loss = val_loss_1 + val_loss_2 + val_loss_3

    push!(validation_losses, val_loss)

    #early early_stopping with patience
    if epoch > 300
      #minimum patience epoch ago 
      min_val_up_to_now = minimum(validation_losses)
      min_val_past = minimum(validation_losses[1:(epoch-300)])
      if min_val_up_to_now == min_val_past && regularization == 0
        println("Early stopping epoch ", epoch)
        flush(stdout)
        return true
      end
    end

    min_val_past = minimum(validation_losses)
    if epoch == 1 || min_val_past == val_loss
      best_on_validation[1] = θ.u
    end

    if epoch % 20 == 0

      regularization_loss = 0.0
      weights = vcat([vec(θ.u.p.p_net[layer_name].weight) for layer_name in keys(θ.u.p.p_net)]...)
      if regularization == 1
        regularization_loss = regularization_loss + regularization_coefficient_1 * sum(abs, weights)
      elseif regularization == 2
        regularization_loss = regularization_loss + regularization_coefficient_1 * sum(abs2, weights)
      elseif regularization == 3
        regularization_loss = regularization_loss + regularization_coefficient_1 * sum(abs2, weights) + regularization_coefficient_2 * sum(abs, weights)
      end

      println("Epoch: ", epoch, " Validation Loss: ", val_loss, " Loss: ", l, " Regularization Loss: ", regularization_loss)
      flush(stdout)
    end

    return false
  end

  function callback(θ, l, hyperparameters, validation_losses, epochs)

    epoch = maximum(epochs)+1
    epochs = push!(epochs, epoch)

    #check if it takes more than 20 minutes
    if Dates.now() - current_time > Dates.Minute(150)
      error("Time limit reached")
    end

    #validation prediction 
    prob_uode_pred_tmp_1 = remake(prob_uode_pred, u0=θ.u.u0[1, :, 1])
    model_prediction_1 = solve(prob_uode_pred_tmp_1, integrator, abstol=abstol, reltol=reltol, saveat=validation_dataframes[1].t, p=θ.u.p)
    max_oscillation = max_oscillations[1]
    val_loss_1 = loss_function(Array(validation_dataframes[1][:, 2:(end-1)])', model_prediction_1, max_oscillation)

    prob_uode_pred_tmp_2 = remake(prob_uode_pred, u0=θ.u.u0[2, :, 1])
    model_prediction_2 = solve(prob_uode_pred_tmp_2, integrator, abstol=abstol, reltol=reltol, saveat=validation_dataframes[1].t, p=θ.u.p)
    max_oscillation = max_oscillations[2]
    val_loss_2 = loss_function(Array(validation_dataframes[2][:, 2:(end-1)])', model_prediction_2, max_oscillation)

    prob_uode_pred_tmp_3 = remake(prob_uode_pred, u0=θ.u.u0[3, :, 1])
    model_prediction_3 = solve(prob_uode_pred_tmp_3, integrator, abstol=abstol, reltol=reltol, saveat=validation_dataframes[1].t, p=θ.u.p)
    max_oscillation = max_oscillations[3]
    val_loss_3 = loss_function(Array(validation_dataframes[3][:, 2:(end-1)])', model_prediction_3, max_oscillation)

    val_loss = val_loss_1 + val_loss_2 + val_loss_3

    push!(validation_losses, val_loss)

    if epoch % 20 == 0

      regularization_loss = 0.0
      weights = vcat([vec(θ.u.p.p_net[layer_name].weight) for layer_name in keys(θ.u.p.p_net)]...)
      if regularization == 1
        regularization_loss = regularization_loss + regularization_coefficient_1 * sum(abs, weights)
      elseif regularization == 2
        regularization_loss = regularization_loss + regularization_coefficient_1 * sum(abs2, weights)
      elseif regularization == 3
        regularization_loss = regularization_loss + regularization_coefficient_1 * sum(abs2, weights) + regularization_coefficient_2 * sum(abs, weights)
      end

      println("Epoch: ", epoch, " Validation Loss: ", val_loss, " Loss: ", l, " Regularization Loss: ", regularization_loss)
      flush(stdout)
    end

    return false
  end

  #optimization
  #defining the optimization procedures
  adtype = Optimization.AutoZygote()

  u0_1=deepcopy(Array(training_dataframes[1][:, 2:(end-1)]))'
  u0_2=deepcopy(Array(training_dataframes[2][:, 2:(end-1)]))'
  u0_3=deepcopy(Array(training_dataframes[3][:, 2:(end-1)]))'

  u0 = zeros(3, size(u0_1, 1), size(u0_1, 2))
  u0[1, :, :] = u0_1
  u0[2, :, :] = u0_2
  u0[3, :, :] = u0_3

  u0_original = deepcopy(u0)

  par = nothing
  if model == 1
    par = ComponentVector(
      p_net=p_net,
      α=ude_parameters[1],
      δ=ude_parameters[2],
    )
  elseif model == 2
    par = ComponentVector(
      p_net=p_net,
      σ=ude_parameters[1],
      r=ude_parameters[2],
    )
  else
    par = ComponentVector(
      p_net=p_net,
      α=ude_parameters[1],
    )
  end
  starting_point_in = ComponentVector{Float64}(p=par, u0=u0)

  optf = Optimization.OptimizationFunction((x, p) -> loss(x, p), adtype)
  optf_2 = Optimization.OptimizationFunction((x, p) -> loss2(x), adtype)

  #set up for multiple shooting
  ranges = DiffEqFlux.group_ranges(size(training_dataframes[1], 1), hyper_ms_segment)
  continuity_cost = hyper_ms_lambda
  tsteps = training_dataframes[1].t
  ms_hyperparameters = (ranges=ranges, continuity_cost=continuity_cost, tsteps=tsteps)

  optprob = Optimization.OptimizationProblem(optf, starting_point_in, ms_hyperparameters)

  opt = OptimizationOptimisers.Adam(learning_rate_adam)

  validation_losses = []
  epochs = [0]

  ##################### ADAM ###########
  res = Optimization.solve(optprob, opt, callback=(θ, l, hyperparameters) -> callback(θ, l,hyperparameters, validation_losses, epochs), maxiters=1000)
  
  optprob2 = Optimization.OptimizationProblem(optf_2, res.u)
  best_on_validation = [res.u]
  validation_losses = [validation_losses[end]]
  epochs = [0]
  res = Optimization.solve(optprob2, Optim.LBFGS(), callback=(θ, l) -> callback2(θ, l, validation_losses, epochs, best_on_validation), maxiters=500)

  likelihood = validation_losses[end]

  elapsed_time = Dates.now() - current_time

  final_solution = res.u
  if regularization == 0
    likelihood = minimum(validation_losses)
    final_solution = best_on_validation[1]
  end

  #saves the results  
  result = (
    original_physical_parameters =  ude_parameters,
    lower_parameter_boundaries = lower_parameter_boundaries,
    upper_parameter_boundaries = upper_parameter_boundaries,
    elapsed_time = elapsed_time,
    training_res=final_solution,
    net_status=st,
    learning_rate_adam=learning_rate_adam,
    neural_network_dimension=neural_network_dimension,
    activation_function=activation_function,
    #1: tanh, 2: relu, 3: sigmoid, 4: gelu
    regularization=regularization,
    #1: l1, 2: l2, 3: elastic_net, 4: early_stopping
    regularization_coefficient_1=regularization_coefficient_1,
    regularization_coefficient_2=regularization_coefficient_2,
    #gain for initialization
    gain=gain,
    # error level, 0: no error, 1: error
    error_level=error_level,
    validation_likelihood=likelihood,
    status="success"
  )

  return result
end

#run the training for the number_ensembles
ensemble_results = []
lock_results = ReentrantLock()
global process_launched = Set()
Threads.@threads for iterator in 1:number_ensembles
  random_seed = nothing
  try
    println("******************************** Starting ensemble ", iterator)
    flush(stdout)
    random_seed = abs(rand(rng, Int))
    global process_launched
    result = train(approximating_neural_network, training_dataframes, validation_dataframes, solution_dataframes, rng, learning_rate_adam, integrator, abstol, reltol, sensealg, random_seed)
    lock(lock_results)
    push!(ensemble_results, result)
    unlock(lock_results)
  catch e 
    println("Error in ensemble ", iterator, " ", e)
    lock(lock_results)
    push!(ensemble_results, (status = "failed",))
    unlock(lock_results)
  end
end

#create the output folder if it does not exist
if !isdir(output_folder)
  mkdir(output_folder)
end

println("Saving the results")
try 
  filename = output_folder * "/ensemble_results_model_" * string(model) * "_gain_"*string(gain)* "_reg_"*string(regularization)*"_"*string(regularization_coefficient_1)*".jld"
  serialize(filename, ensemble_results)
  println(filename)
catch e
  println("Error in saving the results", e)
end
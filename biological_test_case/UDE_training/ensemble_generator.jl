cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, DiffEqFlux

#= ########################################################## reads the command line arguments ##########################################################
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
hyper_ms_lambda = parse(Float64, ARGS[14]) =#

learning_rate_adam =  0.005
neural_network_dimension = 32
activation_function = 4
#1: tanh, 2: relu, 3: sigmoid, 4: gelu
regularization = 4
#1: l1, 2: l2, 3: elastic_net, 4: early_stopping
regularization_coefficient_1 = 0.0
regularization_coefficient_2 = 0.0
#gain for initialization
gain = 1.0
# error level, 0: no error, 1: error
number_ensembles = 5
number_threads = 5
output_folder = "cell_apop_UDE_results"
hyper_ms_segment = 20
hyper_ms_lambda = 0.01

#print the arguments to check their correctness
println("learning_rate_adam = ", learning_rate_adam)
println("neural_network_dimension = ", neural_network_dimension)
println("activation_function = ", activation_function)
println("regularization = ", regularization)
println("regularization_coefficient_1 = ", regularization_coefficient_1)
println("regularization_coefficient_2 = ", regularization_coefficient_2)
println("gain = ", gain)
println("number_ensembles = ", number_ensembles)
println("number_threads = ", number_threads)
println("output_folder = ", output_folder)
println("hyper_ms_segment = ", hyper_ms_segment)
println("hyper_ms_lambda = ", hyper_ms_lambda)
flush(stdout)

########################################################### general configurations ######################################################################
rng = Random.default_rng()
Random.seed!(rng, 0)

# numerical integrator
integrator = TRBDF2(autodiff=true);
abstol = 1e-7
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

#load the data and split into training and validation
datafile = "../data_generator/cell_apoptosis_silico_data.jld"
in_dim = 6
out_dim = 1

include("../cell_apoptosis_settings/cell_apop_model_functions.jl")
include("../cell_apoptosis_settings/cell_apop_model_settings.jl")

upper_parameter_boundaries = original_parameters * 2.0
lower_parameter_boundaries = original_parameters * 0.5
original_ude_parameters = ones(length(original_parameters))

solutions_dataframe = deserialize(datafile)


size_df = size(solutions_dataframe)[1]

size_validation = round(Int, 0.2 * size_df)


mask = shuffle(2:size_df)
validation_mask = mask[1:size_validation]
training_mask = pushfirst!(mask[size_validation+1:end], 1)

training_dataframe = solutions_dataframe[training_mask, :]
validation_dataframe = solutions_dataframe[validation_mask, :]

training_dataframe = sort(training_dataframe, [:t])
validation_dataframe = sort(validation_dataframe, [:t])

########################################## NODE neural network ############################################################
activation_function_fun = gelu
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=gain)
approximating_neural_network = Lux.Chain(
  Lux.Dense(in_dim, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(neural_network_dimension, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(neural_network_dimension, out_dim; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)

max_oscillations = [maximum(training_dataframe[1:end, i]) - minimum(training_dataframe[1:end, i]) for i in 2:(size(training_dataframe, 2))]


solution_dataframes = [solutions_dataframe,]
training_dataframes = [training_dataframe,]
validation_dataframes = [validation_dataframe,]
max_oscillations = [max_oscillations,]

observables = [4,]


###################
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
  prob_uode_pred = ODEProblem{true}(uode_derivative_function, Array(solution_dataframes[1, 2:(end)]), tspan)

  training_data = Array(training_dataframes[1][!, 2:(end)])'
  training_datas = [training_data,]

  #loss function for the comparison among the parameters and the predictions
  function loss_function(data, pred, max_oscillation)
    original_cost = sum(abs2.(data[observables, :] .- pred[observables, :]) ./ abs2.(max_oscillation[observables]))
    return 1 / size(data, 2) * original_cost
  end

  function loss_on_trajectory(θ, hyperparameters, i)

    #.u0[1, :, 1] = u0_original[1, :, 1]

    # Multiple shooting predictions
    sols = [
      solve(
        remake(
          prob_uode_pred;
          p=θ.p,
          tspan=(hyperparameters.tsteps[first(rg)], hyperparameters.tsteps[last(rg)]),
          #u0=θ.u0[:, first(rg)]
          u0=θ.u0[i, :, first(rg)]
          #u0=u0_original[i, :, first(rg)]
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

    curr_loss = loss_1[1]
    
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

  function loss2_on_trajectory(θ, i)
    # Multiple shooting predictions

    tspan = extrema(training_dataframes[i].t)
    
    sol = Array(solve(
      remake(
        prob_uode_pred;
        p=θ.p,
        tspan=tspan,
        u0=u0_original[1, :, 1]
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
    curr_loss = loss_1
    
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
    #if Dates.now() - current_time > Dates.Minute(150)
    #  error("Time limit reached")
    #end

    θ.u.u0[1, :, 1] = u0_original[1, :, 1]


    #validation prediction 
    prob_uode_pred_tmp_1 = remake(prob_uode_pred, u0=max.(θ.u.u0[1, :, 1], 0.0))
    model_prediction_1 = solve(prob_uode_pred_tmp_1, integrator, abstol=abstol, reltol=reltol, saveat=validation_dataframes[1].t, p=θ.u.p)
    max_oscillation = max_oscillations[1]
    val_loss_1 = loss_function(Array(validation_dataframes[1][:, 2:(end)])', model_prediction_1, max_oscillation)

    val_loss = val_loss_1

    #plot the simulations 
    prob_uode_pred_tmp_plot = remake(prob_uode_pred, u0=θ.u.u0[1, :, 1])
    model_prediction_plot = solve(prob_uode_pred_tmp_1, integrator, abstol=abstol, reltol=reltol, saveat=0.01, p=θ.u.p)
    model_prediction_plot_as_array = Array(model_prediction_plot)

    variable_plots = []
    for var_index in 1:(size(training_dataframes[1], 2)-1)
      p = Plots.plot(title="Training Data vs Model Prediction - Variable " * string(var_index), xlabel="Time", ylabel="Value", legend = nothing)
      scatter!(p, training_dataframes[1].t, training_dataframes[1][!, var_index+1], label="Training Data", markersize=3)
      #plot the experimental data
      Plots.scatter!(p, model_prediction_plot.t, model_prediction_plot_as_array[var_index,:], label="Experimental Data", lw=2, linecolor=:black)
      push!(variable_plots, p)
    end

    #put them together
    plot_layout = @layout [a b; c d; e f; g h]
    combined_plot = Plots.plot(variable_plots..., layout=plot_layout, size=(1200, 800))

    #display 
    display(combined_plot)

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

    println("Iteration " * string(maximum(epochs)+1) * " Loss: " * string(l))

    epoch = maximum(epochs)+1
    epochs = push!(epochs, epoch)


    θ.u.u0[1, :, 1] = u0_original[1, :, 1]

    #check the simulations 
    i = 1
    sols = [
      solve(
        remake(
          prob_uode_pred;
          p=θ.u.p,
          tspan=(hyperparameters.tsteps[first(rg)], hyperparameters.tsteps[last(rg)]),
          #u0=θ.u0[:, first(rg)]
          #u0=θ.u0[i, :, first(rg)]
          u0=max.(θ.u.u0[i, :, first(rg)], 0.0)
        ),
        integrator;
        saveat=tsteps[rg],
        reltol=reltol,
        abstol=abstol,
        sensealg=sensealg,
        verbose=true
      ) for rg in hyperparameters.ranges
    ]

    # plot against the training data
    variable_plots = []
    for var_index in 1:(size(training_dataframes[1], 2)-1)
      p = Plots.plot(title="Training Data vs Model Prediction - Variable " * string(var_index), xlabel="Time", ylabel="Value", legend = nothing)
      scatter!(p, training_dataframes[1].t, training_dataframes[1][!, var_index+1], label="Training Data", markersize=3)
      for (j, rg) in enumerate(hyperparameters.ranges)
        model_prediction_range = Array(sols[j])
        Plots.plot!(p, hyperparameters.tsteps[rg], model_prediction_range[var_index, :], label="Model Prediction Segment " * string(j))
      end
      #plot the experimental data
      Plots.scatter!(p, solution_dataframes[1].t, solution_dataframes[1][!, var_index+1], label="Experimental Data", lw=2, linecolor=:black)
      push!(variable_plots, p)
    end

    #put them together
    plot_layout = @layout [a b; c d; e f; g h]
    combined_plot = Plots.plot(variable_plots..., layout=plot_layout, size=(1200, 800))

    #display 
    display(combined_plot)
    
    val_loss = Inf 
    push!(validation_losses, val_loss)

    #= #check if it takes more than 20 minutes
    if Dates.now() - current_time > Dates.Minute(150)
      error("Time limit reached")
    end

    #validation prediction 
    prob_uode_pred_tmp_1 = remake(prob_uode_pred, u0=θ.u.u0[1, :, 1])
    model_prediction_1 = solve(prob_uode_pred_tmp_1, integrator, abstol=abstol, reltol=reltol, saveat=validation_dataframes[1].t, p=θ.u.p)
    max_oscillation = max_oscillations[1]
    val_loss_1 = loss_function(Array(validation_dataframes[1][:, 2:(end)])', model_prediction_1, max_oscillation)

    val_loss = val_loss_1
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
    end =#

    return false
  end

  #optimization
  #defining the optimization procedures
  adtype = Optimization.AutoZygote()

  u0_1=deepcopy(Array(training_dataframes[1][:, 2:(end)]))'

  u0 = zeros(1, size(u0_1, 1), size(u0_1, 2))
  u0[1, :, :] = u0_1

  #take just the initial point for the trajectories not observables
  u0[1, setdiff(1:size(u0, 2), observables), :] .= u0_1[setdiff(1:size(u0_1, 1), observables), 1][:, 1]

  u0_original = deepcopy(u0)

  par = ComponentVector(
      p_net=p_net,
      ode_par=ude_parameters
  )
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
  res = Optimization.solve(optprob, opt, callback=(θ, l) -> callback(θ, l, ms_hyperparameters, validation_losses, epochs), maxiters=100)
  
  optprob2 = Optimization.OptimizationProblem(optf_2, res.u)
  best_on_validation = [res.u]
  validation_losses = [validation_losses[end]]
  epochs = [0]
  res = Optimization.solve(optprob2, Optim.LBFGS(), callback=(θ, l) -> callback2(θ, l, validation_losses, epochs, best_on_validation), maxiters=50)

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
    #error_level=error_level,
    validation_likelihood=likelihood,
    status="success"
  )

  return result
end

#run the training for the number_ensembles
ensemble_results = []
#lock_results = ReentrantLock()
#global process_launched = Set()
#Threads.@threads for iterator in 1:
number_ensembles = 53
for iterator in 1:number_ensembles
  random_seed = nothing
  try
    println("******************************** Starting ensemble ", iterator)
    flush(stdout)
    random_seed = abs(rand(rng, Int))

    global process_launched
    result = train(approximating_neural_network, training_dataframes, validation_dataframes, solution_dataframes, rng, learning_rate_adam, integrator, abstol, reltol, sensealg, random_seed)
    #lock(lock_results)
    push!(ensemble_results, result)
    #unlock(lock_results)
  catch e 
    println("Error in ensemble ", iterator, " ", e)
    #lock(lock_results)
    push!(ensemble_results, (status = "failed",))
    #unlock(lock_results)
  end
end

#create the output folder if it does not exist
if !isdir(output_folder)
  mkdir(output_folder)
end

println("Saving the results")
try 
  filename = output_folder * "/ensemble_results.jld"
  serialize(filename, ensemble_results)
  println(filename)
catch e
  println("Error in saving the results", e)
end
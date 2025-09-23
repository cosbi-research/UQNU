cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates

########################################################## reads the command line arguments ##########################################################
learning_rate_adam = 0.05
neural_network_dimension = 32
activation_function = 4
regularization = 0
regularization_coefficient_1 = 0.0
regularization_coefficient_2 = 0.0
gain = 1.0
error_level = 1
number_ensembles = 1
number_threads = 1
output_folder = "initialization_difference"
model = 1
hyper_ms_segment = 15
hyper_ms_lambda = 0.1

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
datafile = "../../data_generator/lotka_volterra_in_silico_data"
in_dim = 2
out_dim = 2
if model == 2
  in_dim = 3
  out_dim = 1
  println("Considering model lorenz")
  datafile = "../../data_generator/lorenz_in_silico_data"
elseif model == 3
  in_dim = 2
  out_dim = 2
  println("Considering model damped oscillator")
  datafile = "../../data_generator/damped_oscillator_in_silico_data"
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
  get_uode_model_function = function (appr_neural_network, state)
    #generates the function with the parameters
    f(du, u, p, t) =
      let appr_neural_network = appr_neural_network, st = state
        û = appr_neural_network(u, p.p_net, st)[1]
        α = p.α
        δ = p.δ
        @inbounds du[1] = α*u[1] + û[1]
        @inbounds du[2] = û[2] - δ*u[2]
      end
  end
  parameters_ground_truth = [1.3, 1.8]
elseif model == 2
  #model function
  get_uode_model_function = function (appr_neural_network, state)
    #generates the function with the parameters
    f(du, u, p, t) =
      let appr_neural_network = appr_neural_network, st = state
        σ= p.σ
        r = p.r
        û = appr_neural_network(u, p.p_net, st)[1]
        @inbounds du[1] = σ*(u[2] - u[1]) 
        @inbounds du[2] = u[1]*(r-u[3])- u[2]
        @inbounds du[3] = û[1]
      end
  end
  parameters_ground_truth = [10.0, 28.0]
else 
  #model function
  get_uode_model_function = function (appr_neural_network, state)
    #generates the function with the parameters
    f(du, u, p, t) =
      let appr_neural_network = appr_neural_network, st = state
        α = p.α
        û = appr_neural_network(u, p.p_net, st)[1]
        @inbounds du[1] = -α*u[1]^3 + û[1]
        @inbounds du[2] = û[2] - α*u[2]^3
      end
  end
  parameters_ground_truth = [0.1,]
end 

lower_parameter_boundary = parameters_ground_truth .* 0.1
upper_parameter_boundary = parameters_ground_truth .* 10.0

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
  Lux.Dense(in_dim, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform),
  Lux.Dense(neural_network_dimension, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform),
  Lux.Dense(neural_network_dimension, out_dim; init_weight=my_glorot_uniform),
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
  #UDE derivative function
  tspan = extrema(solution_dataframes[1].t)
  uode_derivative_function = get_uode_model_function(approximating_neural_network, st)
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


    weights = vcat([vec(θ.p[layer_name].weight) for layer_name in keys(θ.p)]...)
    if regularization == 1
      curr_loss = curr_loss + regularization_coefficient_1 * sum(abs, weights)
    elseif regularization == 2
      curr_loss = curr_loss + regularization_coefficient_1 * sum(abs2, weights)
    elseif regularization == 3
      curr_loss = curr_loss + regularization_coefficient_1 * sum(abs2, weights) + regularization_coefficient_2 * sum(abs, weights)
    end
    return curr_loss, hyperparameters
  end

  function loss(θ, hyperparameters)
    loss_1 = loss_on_trajectory(θ, hyperparameters, 1)
    loss_2 = loss_on_trajectory(θ, hyperparameters, 2)
    loss_3 = loss_on_trajectory(θ, hyperparameters, 3)
    return loss_1[1] + loss_2[1] + loss_3[1], hyperparameters
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
    return loss_1 + loss_2 + loss_3
  end

  function callback2(θ, l, validation_losses, epochs)
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
    if epoch > 10000
      #minimum patience epoch ago 
      min_val_up_to_now = minimum(validation_losses)
      min_val_past = minimum(validation_losses[1:(epoch-200)])
      if min_val_up_to_now == min_val_past
        println("Early stopping epoch ", epoch)
        flush(stdout)
        return false
      end
    end

    #plot 
    if epoch % 20 == 0

      model_unique_prediction_1 = solve(remake(
        prob_uode_pred;
        p=θ.u.p,
        tspan=(solution_dataframes[1].t[1],solution_dataframes[1].t[end]),
        #u0=θ.u0[:, first(rg)]
        u0=θ.u.u0[1, :, 1]
      ), integrator, abstol=abstol, reltol=reltol, saveat=0.01, p=θ.u.p)

      model_unique_prediction_2 = solve(remake(
        prob_uode_pred;
        p=θ.u.p,
        tspan=(solution_dataframes[2].t[1],solution_dataframes[2].t[end]),
        #u0=θ.u0[:, first(rg)]
        u0=θ.u.u0[2, :, 1]
      ), integrator, abstol=abstol, reltol=reltol, saveat=0.01, p=θ.u.p)

      model_unique_prediction_3 = solve(remake(
        prob_uode_pred;
        p=θ.u.p,
        tspan=(solution_dataframes[3].t[1],solution_dataframes[3].t[end]),
        #u0=θ.u0[:, first(rg)]
        u0=θ.u.u0[3, :, 1]
      ), integrator, abstol=abstol, reltol=reltol, saveat=0.01, p=θ.u.p)

      plts = [Plots.scatter(solution_dataframes[1].t, [solution_dataframes[1].x1, solution_dataframes[1].x2], label=["ground truth x1" "ground truth x2"]),
              Plots.scatter(solution_dataframes[2].t, [solution_dataframes[2].x1, solution_dataframes[2].x2], label=["ground truth x1" "ground truth x2"]),
              Plots.scatter(solution_dataframes[3].t, [solution_dataframes[3].x1, solution_dataframes[3].x2], label=["ground truth x1" "ground truth x2"]),
              ]

      Plots.plot!(plts[1], model_unique_prediction_1.t, Array(model_unique_prediction_1)', label=["pred x1" "pred x2"])
      title!(plts[1], "Total Epoch: $epoch")
      Plots.plot!(plts[1], legend=false)

      Plots.plot!(plts[2], model_unique_prediction_2.t, Array(model_unique_prediction_2)', label=["pred x1" "pred x2"])
      title!(plts[2], "Total Epoch: $epoch")
      Plots.plot!(plts[2], legend=false)

      Plots.plot!(plts[3], model_unique_prediction_3.t, Array(model_unique_prediction_3)', label=["pred x1" "pred x2"])
      title!(plts[3], "Total Epoch: $epoch")
      Plots.plot!(plts[3], legend=false)

      plt = Plots.plot(plts..., layout=(1, 3))
      display(plt)
      
      rounded_val_loss = round(val_loss, digits=6)
      rounded_l = round(l, digits=6)
      println("Epoch: ", epoch, " Validation Loss: ", rounded_val_loss, " Loss: ", rounded_l)
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

    #early early_stopping with patience
    if epoch > 10000
      #minimum patience epoch ago 
      min_val_up_to_now = minimum(validation_losses)
      min_val_past = minimum(validation_losses[1:(epoch-200)])
      if min_val_up_to_now == min_val_past
        println("Early stopping epoch ", epoch)
        flush(stdout)
        return true
      end
    end

    #plot 
    if epoch % 20 == 0

      model_predictions_1 = [
        solve(
          remake(
            prob_uode_pred;
            p=θ.u.p,
            tspan=(hyperparameters.tsteps[first(rg)], hyperparameters.tsteps[last(rg)]),
            #u0=θ.u0[:, first(rg)]
            u0=θ.u.u0[1,:, first(rg)]
          ),
          integrator;
          saveat=tsteps[rg],
          reltol=reltol,
          abstol=abstol,
          sensealg=sensealg,
          verbose=true
        ) for rg in hyperparameters.ranges
      ]

      model_predictions_2 = [
        solve(
          remake(
            prob_uode_pred;
            p=θ.u.p,
            tspan=(hyperparameters.tsteps[first(rg)], hyperparameters.tsteps[last(rg)]),
            #u0=θ.u0[:, first(rg)]
            u0=θ.u.u0[2,:, first(rg)]
          ),
          integrator;
          saveat=tsteps[rg],
          reltol=reltol,
          abstol=abstol,
          sensealg=sensealg,
          verbose=true
        ) for rg in hyperparameters.ranges
      ]

      model_predictions_3 = [
        solve(
          remake(
            prob_uode_pred;
            p=θ.u.p,
            tspan=(hyperparameters.tsteps[first(rg)], hyperparameters.tsteps[last(rg)]),
            #u0=θ.u0[:, first(rg)]
            u0=θ.u.u0[3,:, first(rg)]
          ),
          integrator;
          saveat=tsteps[rg],
          reltol=reltol,
          abstol=abstol,
          sensealg=sensealg,
          verbose=true
        ) for rg in hyperparameters.ranges
      ]

      model_unique_prediction_1 = solve(remake(
        prob_uode_pred;
        p=θ.u.p,
        tspan=(solution_dataframes[1].t[1],solution_dataframes[1].t[end]),
        #u0=θ.u0[:, first(rg)]
        u0=θ.u.u0[1, :, 1]
      ), integrator, abstol=abstol, reltol=reltol, saveat=0.01, p=θ.u.p)

      model_unique_prediction_2 = solve(remake(
        prob_uode_pred;
        p=θ.u.p,
        tspan=(solution_dataframes[2].t[1],solution_dataframes[2].t[end]),
        #u0=θ.u0[:, first(rg)]
        u0=θ.u.u0[2, :, 1]
      ), integrator, abstol=abstol, reltol=reltol, saveat=0.01, p=θ.u.p)

      model_unique_prediction_3 = solve(remake(
        prob_uode_pred;
        p=θ.u.p,
        tspan=(solution_dataframes[3].t[1],solution_dataframes[3].t[end]),
        #u0=θ.u0[:, first(rg)]
        u0=θ.u.u0[3, :, 1]
      ), integrator, abstol=abstol, reltol=reltol, saveat=0.01, p=θ.u.p)

      plts = [Plots.scatter(training_dataframes[1].t, [training_dataframes[1].x1, training_dataframes[1].x2], label=["ground truth x1" "ground truth x2"], alpha=.3),
              Plots.scatter(training_dataframes[2].t, [training_dataframes[2].x1, training_dataframes[2].x2], label=["ground truth x1" "ground truth x2"], alpha=.3),
              Plots.scatter(training_dataframes[3].t, [training_dataframes[3].x1, training_dataframes[3].x2], label=["ground truth x1" "ground truth x2"], alpha=.3),
              Plots.scatter(solution_dataframes[1].t, [solution_dataframes[1].x1, solution_dataframes[1].x2], label=["ground truth x1" "ground truth x2"]),
              Plots.scatter(solution_dataframes[2].t, [solution_dataframes[2].x1, solution_dataframes[2].x2], label=["ground truth x1" "ground truth x2"]),
              Plots.scatter(solution_dataframes[3].t, [solution_dataframes[3].x1, solution_dataframes[3].x2], label=["ground truth x1" "ground truth x2"]),
              ]

      Plots.plot!(plts[4], model_unique_prediction_1.t, Array(model_unique_prediction_1)', label=["pred x1" "pred x2"])
      title!(plts[4], "Total Epoch: $epoch")
      Plots.plot!(plts[4], legend=false)

      Plots.plot!(plts[5], model_unique_prediction_2.t, Array(model_unique_prediction_2)', label=["pred x1" "pred x2"])
      title!(plts[5], "Total Epoch: $epoch")
      Plots.plot!(plts[5], legend=false)

      Plots.plot!(plts[6], model_unique_prediction_3.t, Array(model_unique_prediction_3)', label=["pred x1" "pred x2"])
      title!(plts[6], "Total Epoch: $epoch")
      Plots.plot!(plts[6], legend=false)

      for count_rg in 1:length(model_predictions_1)
        model_prediction = model_predictions_1[count_rg]
        Plots.plot!(plts[1], model_prediction.t, Array(model_prediction)', label=["pred x1" "pred x2"])
        rg = hyperparameters.ranges[count_rg]
        Plots.scatter!(plts[1], [training_dataframes[1].t[first(rg)]], [[training_dataframes[1].x1[first(rg)]], [training_dataframes[1].x2[first(rg)]]], color="red")
        Plots.plot!(plts[1], legend=false)
      end
      title!(plts[1], "MS Epoch: $epoch")

      for count_rg in 1:length(model_predictions_2)
        model_prediction = model_predictions_2[count_rg]
        Plots.plot!(plts[2], model_prediction.t, Array(model_prediction)', label=["pred x1" "pred x2"])
        rg = hyperparameters.ranges[count_rg]
        Plots.scatter!(plts[2], [training_dataframes[2].t[first(rg)]], [[training_dataframes[2].x1[first(rg)]], [training_dataframes[2].x2[first(rg)]]], color="red")
        Plots.plot!(plts[2], legend=false)
      end
      title!(plts[2], "MS Epoch: $epoch")

      for count_rg in 1:length(model_predictions_3)
        model_prediction = model_predictions_3[count_rg]
        Plots.plot!(plts[3], model_prediction.t, Array(model_prediction)', label=["pred x1" "pred x2"])
        rg = hyperparameters.ranges[count_rg]
        Plots.scatter!(plts[3], [training_dataframes[3].t[first(rg)]], [[training_dataframes[3].x1[first(rg)]], [training_dataframes[3].x2[first(rg)]]], color="red")
        Plots.plot!(plts[3], legend=false)
      end
      title!(plts[3], "MS Epoch: $epoch")

      plt = Plots.plot(plts..., layout=(2, 3), legend=false)
      display(plt)

      #println("Epoch: ", epoch)
      #println("Validation Loss: ", val_loss)
      #println("Loss: ", l)
      rounded_val_loss = round(val_loss, digits=6)
      rounded_l = round(l, digits=6)
      println("Epoch: ", epoch, " Validation Loss: ", rounded_val_loss, " Loss: ", rounded_l)
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

  starting_point_in = ComponentVector{Float64}(p=p_net, u0=u0)

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
  res = Optimization.solve(optprob, opt, callback=(θ, l, hyperparameters) -> callback(θ, l, hyperparameters, validation_losses, epochs), maxiters=1000)
  optprob2 = Optimization.OptimizationProblem(optf_2, res.u)
  res = Optimization.solve(optprob2, Optim.LBFGS(), callback=(θ, l) -> callback2(θ, l, validation_losses, epochs), maxiters=500)

  likelihood = validation_losses[end]

  elapsed_time = Dates.now() - current_time

  #saves the results  
  result = (
    elapsed_time = elapsed_time,
    training_res=res.u,
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
for iterator in 1:number_ensembles
  random_seed = nothing
  try
    println("******************************** Starting ensemble ", iterator)
    flush(stdout)
    random_seed = abs(rand(rng, Int))
    global process_launched
    #push!(process_launched, random_seed)
    #println("******************************** Active processes ", string(process_launched))
    result = train(approximating_neural_network, training_dataframes, validation_dataframes, solution_dataframes, rng, learning_rate_adam, integrator, abstol, reltol, sensealg, random_seed)
    lock(lock_results)
    push!(ensemble_results, result)
    unlock(lock_results)
    #delete!(process_launched, random_seed)
    #println("******************************** Active processes ", string(process_launched))
    #flush(stdout)
  catch e 
    println("Error in ensemble ", iterator, " ", e)
    push!(ensemble_results, (status = "failed",))
    unlock(lock_results)
  end
end

#create the output folder if it does not exist
if !isdir(output_folder)
  mkdir(output_folder)
end
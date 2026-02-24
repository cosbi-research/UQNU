cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, OrdinaryDiffEq, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes, Dates, DiffEqFlux, Distributions


neural_network_dimension = 32
activation_function = 4

########################################################### general configurations ######################################################################
rng = Random.default_rng()
Random.seed!(rng, 0)

# numerical integrator
integrator = TRBDF2(autodiff=true);
abstol = 1e-4
reltol = 1e-5
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

#load the data and split into training and validation
datafile = "../../data_generator/cell_apoptosis_silico_data.jld"
in_dim = 6
out_dim = 1

observables = [4,]

include("../../cell_apoptosis_settings/cell_apop_model_functions.jl")
include("../../cell_apoptosis_settings/cell_apop_model_settings.jl")

cd(@__DIR__)

upper_parameter_boundaries = original_parameters * 2.0
lower_parameter_boundaries = original_parameters * 0.5
original_ude_parameters = ones(length(original_parameters))

solutions_dataframe = deserialize(datafile)
training_data = Array(solutions_dataframe[!, 2:(end)])'
training_datas = [training_data,]

variable_plots = []
losses = []

original_u0_survival = [1.34 * 10^5 / 10^5, 1.0 * 10^5 / 10^5, 2.67 * 10^5 / 10^5, 0.0 / 10^5, 0.0 / 10^5, 0.0 / 10^5, 2.9 * 10^3 / 10^5, 0.0 / 10^5]
original_u0_death = [1.34 * 10^5 / 10^5, 1.0 * 10^5 / 10^5, 2.67 * 10^5 / 10^5, 0.0 / 10^5, 0.0 / 10^5, 0.0 / 10^5, 2.9 * 10^4 / 10^5, 0.0 / 10^5]

max_oscillations = [maximum(solutions_dataframe[1:end, i]) - minimum(solutions_dataframe[1:end, i]) for i in 2:(size(solutions_dataframe, 2))]

solution_dataframes = [solutions_dataframe,]
max_oscillations = [max_oscillations,]

#if the result folder doesn't exist, create it
if !isdir("cell_apop_UDE_results")
  mkdir("cell_apop_UDE_results")
end

datafile = "../../data_generator/cell_apoptosis_silico_data_no_noise.jld"
solutions_dataframe = deserialize(datafile)

death_trajectory_contained = []
MOD_death_trajectory_contained = []
#### DEATH CONDITION ANALYSIS  ######
for ensemble_index in 1:4

  global solution_dataframes
  global max_oscillations


  #load the training results 
  results = deserialize("result_ca$ensemble_index/results.jld")
  MOD_ensemble = results.ensemble_original
  initial_states = results.initial_states
  initial_states = vec(initial_states[1, :, 1]) #take the death trajectory initial states


  if MOD_ensemble == nothing
    println("No MOD ensemble found for ensemble index $ensemble_index")
    continue
  end

  naive_ensemble = results.naive_ensemble_reference
  naive_ensemble_reference_initial_states = results.naive_ensemble_reference_initial_states

  variable_plots = []

  predictions = []
  MOD_predictions = []

  for index in axes(naive_ensemble,1)

    res = naive_ensemble[index]
    initial_states_tmp = naive_ensemble_reference_initial_states[index]
    initial_states_tmp = vec(initial_states_tmp[:, :, 1]) #take the death trajectory initial states

    global variable_plots

    ########################################## NODE neural network ############################################################
    activation_function_fun = gelu
    my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=1.0)
    approximating_neural_network = Lux.Chain(
      Lux.Dense(in_dim, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(neural_network_dimension, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(neural_network_dimension, out_dim; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    )

    observables = [4,]

    local_rng = StableRNG(0)
    p_net, st = Lux.setup(local_rng, approximating_neural_network)

    tspan = extrema(solutions_dataframe.t)
    uode_derivative_function = get_uode_model_function_not_scaled(approximating_neural_network, st)
    prob_uode_pred = ODEProblem{true}(uode_derivative_function, Array(solutions_dataframe[1, 2:(end)]), tspan)

    #loss function for the comparison among the parameters and the predictions
    function loss_function(data, pred, max_oscillation)
      original_cost = sum(abs2.(data[observables, :] .- pred[observables, :]) ./ abs2.(max_oscillation[observables]))
      return 1 / size(data, 2) * original_cost
    end

    θ = res
    sol = Array(solve(
      remake(
        prob_uode_pred;
        p=θ,
        tspan=tspan,
        u0=initial_states_tmp
        #u0=u0_original[i, :, 1]
      ),
      integrator;
      saveat=solutions_dataframe.t,
      reltol=reltol,
      abstol=abstol,
      sensealg=sensealg,
      verbose=true
    ))

    if size(sol, 2) != length(solutions_dataframe.t)
      println("Simulation failed for this result.")
      continue
    end

    push!(predictions, sol)

    #plot the results towards the training data 
    if length(variable_plots) == 0
      for var_index in 1:size(training_datas[1], 1)

        p = Plots.scatter(
          solutions_dataframe.t,
          solutions_dataframe[:, var_index+1],
          label="training data",
          lw=2,
          xlabel="time",
          ylabel="concentration",
          title="Variable $(var_index)",
        )
        plot!(p, solutions_dataframe.t, sol[var_index, :], label="UDE prediction", lw=2, ls=:dash, color=:red, legend=nothing)
        #range 0-1 for y axis
        y_min = minimum(solutions_dataframe[:, var_index+1])
        y_max = maximum(solutions_dataframe[:, var_index+1])
        y_range = y_max - y_min
        y_lower = y_min - 1 * y_range
        y_upper = y_max + 1 * y_range
        Plots.plot!(p, ylims=(y_lower, y_upper))
        push!(variable_plots, p)
      end
    else
      for var_index in 1:size(training_datas[1], 1)
        p = variable_plots[var_index]
        plot!(p, solutions_dataframe.t, sol[var_index, :], label="UDE prediction", lw=2, ls=:dash, color=:red, legend=nothing)
        y_min = minimum(solutions_dataframe[:, var_index+1])
        y_max = maximum(solutions_dataframe[:, var_index+1])
        y_range = y_max - y_min
        y_lower = y_min - 1 * y_range
        y_upper = y_max + 1 * y_range
        Plots.plot!(p, ylims=(y_lower, y_upper))
      end
    end
  end

  #plot in a grid layout
  p = Plots.plot(variable_plots..., layout=(4, 2), size=(900, 900))
  #circle the forth variable which is the observable
  savefig(p, "cell_apop_UDE_results/v_se_death_$ensemble_index.png")


  global variable_plots = []
  for res in MOD_ensemble

    global variable_plots

    ########################################## NODE neural network ############################################################
    activation_function_fun = gelu
    my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=1.0)
    approximating_neural_network = Lux.Chain(
      Lux.Dense(in_dim, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(neural_network_dimension, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(neural_network_dimension, out_dim; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    )

    observables = [4,]

    local_rng = StableRNG(0)
    p_net, st = Lux.setup(local_rng, approximating_neural_network)

    tspan = extrema(solutions_dataframe.t)
    uode_derivative_function = get_uode_model_function_not_scaled(approximating_neural_network, st)
    prob_uode_pred = ODEProblem{true}(uode_derivative_function, initial_states, tspan)

    #loss function for the comparison among the parameters and the predictions
    function loss_function(data, pred, max_oscillation)
      original_cost = sum(abs2.(data[observables, :] .- pred[observables, :]) ./ abs2.(max_oscillation[observables]))
      return 1 / size(data, 2) * original_cost
    end

    θ = res
    sol = Array(solve(
      remake(
        prob_uode_pred;
        p=θ,
        tspan=tspan,
        u0=initial_states
        #u0=u0_original[i, :, 1]
      ),
      integrator;
      saveat=solutions_dataframe.t,
      reltol=reltol,
      abstol=abstol,
      sensealg=sensealg,
      verbose=true
    ))

    if size(sol, 2) != length(solutions_dataframe.t)
      println("Simulation failed for this result.")
      continue
    end

    push!(MOD_predictions, sol)

    #plot the results towards the training data 
    if length(variable_plots) == 0
      for var_index in 1:size(training_datas[1], 1)

        p = Plots.scatter(
          solutions_dataframe.t,
          solutions_dataframe[:, var_index+1],
          label="training data",
          lw=2,
          xlabel="time",
          ylabel="concentration",
          title="Variable $(var_index)",
        )
        plot!(p, solutions_dataframe.t, sol[var_index, :], label="UDE prediction", lw=2, ls=:dash, color=:red, legend=nothing)
        #range 0-1 for y axis
        y_min = minimum(solutions_dataframe[:, var_index+1])
        y_max = maximum(solutions_dataframe[:, var_index+1])
        y_range = y_max - y_min
        y_lower = y_min - 1 * y_range
        y_upper = y_max + 1 * y_range
        Plots.plot!(p, ylims=(y_lower, y_upper))
        push!(variable_plots, p)
      end
    else
      for var_index in 1:size(training_datas[1], 1)
        p = variable_plots[var_index]
        plot!(p, solutions_dataframe.t, sol[var_index, :], label="UDE prediction", lw=2, ls=:dash, color=:red, legend=nothing)
        y_min = minimum(solutions_dataframe[:, var_index+1])
        y_max = maximum(solutions_dataframe[:, var_index+1])
        y_range = y_max - y_min
        y_lower = y_min - 1 * y_range
        y_upper = y_max + 1 * y_range
        Plots.plot!(p, ylims=(y_lower, y_upper))
      end
    end
  end

  #plot in a grid layout
  p = Plots.plot(variable_plots..., layout=(4, 2), size=(900, 900))
  #circle the forth variable which is the observable
  savefig(p, "cell_apop_UDE_results/v_se_death_MOD_$ensemble_index.png")

  #for each time, i compute the mean and std of the predictions
  mean_prediction = zeros(size(predictions[1]))
  std_prediction = zeros(size(predictions[1]))

  MOD_mean_prediction = zeros(size(MOD_predictions[1]))
  MOD_std_prediction = zeros(size(MOD_predictions[1]))

  for t_index in 1:size(predictions[1], 2)
    #collect all predictions at this time
    preds_at_t = [predictions[i][:, t_index] for i in 1:length(predictions)]
    #compute mean and std
    mean_prediction[:, t_index] = mean(hcat(preds_at_t...), dims=2)[:]
    std_prediction[:, t_index] = std(hcat(preds_at_t...), dims=2)[:]

    MOD_preds_at_t = [MOD_predictions[i][:, t_index] for i in 1:length(MOD_predictions)]
    #compute mean and std
    MOD_mean_prediction[:, t_index] = mean(hcat(MOD_preds_at_t...), dims=2)[:]
    MOD_std_prediction[:, t_index] = std(hcat(MOD_preds_at_t...), dims=2)[:]
  end

  #try and visualize the results with 1.96 * std as confidence interval
  variable_plots = []
  variable_contained_in_ci = []
  MOD_variable_contained_in_ci = []
  for var_index in 1:size(training_datas[1], 1)

    p = Plots.scatter(
      solutions_dataframe.t,
      solutions_dataframe[:, var_index+1],
      label="training data",
      lw=2,
      xlabel="time",
      ylabel="concentration",
      title="Variable $(var_index)",
    )
    plot!(p, solutions_dataframe.t, mean_prediction[var_index, :], label="UDE mean prediction", lw=2, ls=:dash, color=:red)

    n_ensemble = 5
    t_value = quantile(TDist(n_ensemble - 1), 0.975)

    #plot the confidence interval
    upper_bound = mean_prediction[var_index, :] .+ t_value * sqrt(1 + 1 / n_ensemble) * std_prediction[var_index, :]
    lower_bound = mean_prediction[var_index, :] .- t_value * sqrt(1 + 1 / n_ensemble) * std_prediction[var_index, :]
    upper_bound_MOD = MOD_mean_prediction[var_index, :] .+ t_value * sqrt(1 + 1 / n_ensemble) * MOD_std_prediction[var_index, :]
    lower_bound_MOD = MOD_mean_prediction[var_index, :] .- t_value * sqrt(1 + 1 / n_ensemble) * MOD_std_prediction[var_index, :]
    plot!(p, solutions_dataframe.t, upper_bound, lw=1, ls=:dot, color=:orange, label="95% CI")
    plot!(p, solutions_dataframe.t, lower_bound, lw=1, ls=:dot, color=:orange, label="")
    plot!(p, solutions_dataframe.t, upper_bound_MOD, lw=1, ls=:dashdot, color=:green, label="95% CI MOD")
    plot!(p, solutions_dataframe.t, lower_bound_MOD, lw=1, ls=:dashdot, color=:green, label="")
    #fill between upper and lower bound
    Plots.plot!(p, solutions_dataframe.t, upper_bound, fillrange=lower_bound, fillalpha=0.2, fillcolor=:orange, label="")
    Plots.plot!(p, solutions_dataframe.t, upper_bound_MOD, fillrange=lower_bound_MOD, fillalpha=0.2, fillcolor=:green, label="")
    Plots.plot!(p, legend=nothing)

    #check if the variable is inside the confidence interval
    observations = solutions_dataframe[:, var_index+1]
    epsilon = 1e-4
    inside_ci = ((observations .>= (lower_bound .- epsilon)) .& (observations .<= (upper_bound .+ epsilon)))
    inside_ci_MOD = ((observations .>= (lower_bound_MOD .- epsilon)) .& (observations .<= (upper_bound_MOD .+ epsilon)))

    push!(variable_contained_in_ci, inside_ci)
    push!(MOD_variable_contained_in_ci, inside_ci_MOD)

    #range 0-1 for y axis
    y_min = minimum(solutions_dataframe[:, var_index+1])
    y_max = maximum(solutions_dataframe[:, var_index+1])
    y_range = y_max - y_min
    y_lower = y_min - 1 * y_range
    y_upper = y_max + 1 * y_range
    Plots.plot!(p, ylims=(y_lower, y_upper))
    push!(variable_plots, p)
  end

  #make the % between the same positions of the variable contained in ci
  contained_points = zeros(size(variable_contained_in_ci[1]))
  for t_index in 1:length(contained_points)
    contained = 1
    for var_index in observables
      if !variable_contained_in_ci[var_index][t_index]
        contained = 0
        break
      end
    end
    contained_points[t_index] = contained
  end
  contained_points = sum(contained_points) / length(contained_points)

  push!(death_trajectory_contained, contained_points)

  contained_points_MOD = zeros(size(MOD_variable_contained_in_ci[1]))
  for t_index in 1:length(contained_points_MOD)
    contained = 1
    for var_index in observables
      if !MOD_variable_contained_in_ci[var_index][t_index]
        contained = 0
        break
      end
    end
    contained_points_MOD[t_index] = contained
  end

  contained_points_MOD = sum(contained_points_MOD) / length(contained_points_MOD)

  push!(MOD_death_trajectory_contained, contained_points_MOD)

  #plot in a grid layout
  p = Plots.plot(variable_plots..., layout=(4, 2), size=(900, 900))
  #circle the forth variable which is the observable
  savefig(p, "cell_apop_UDE_results/v_se_ci_ensemble_death_$ensemble_index.png")
end

serialize("cell_apop_UDE_results/trajectory_contained_in_ci_death.jld", death_trajectory_contained)
serialize("cell_apop_UDE_results/MOD_trajectory_contained_in_ci_death.jld", MOD_death_trajectory_contained)

datafile = "../../data_generator/cell_apoptosis_silico_data_survival.jld"
solutions_dataframe = deserialize(datafile)

#####SURVIVAL CONDITION ANALYSIS #########
survival_trajectory_contained = []
MOD_survival_trajectory_contained = []

for ensemble_index in 1:4

  global solution_dataframes
  global max_oscillations


  variable_plots = []

  predictions = []


  #load the training results 
  results = deserialize("result_ca$ensemble_index/results.jld")
  MOD_ensemble = results.ensemble_original
  naive_ensemble = results.naive_ensemble_reference

  initial_states = results.initial_states
  initial_states = vec(initial_states[:, :, 1]) #take the death trajectory initial states


  if MOD_ensemble == nothing
    println("No MOD ensemble found for ensemble index $ensemble_index")
    continue
  end

  variable_plots = []

  predictions = []
  MOD_predictions = []

  for res in naive_ensemble

    global variable_plots

    ########################################## NODE neural network ############################################################
    activation_function_fun = gelu
    my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=1.0)
    approximating_neural_network = Lux.Chain(
      Lux.Dense(in_dim, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(neural_network_dimension, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(neural_network_dimension, out_dim; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    )

    observables = [4,]

    local_rng = StableRNG(0)
    p_net, st = Lux.setup(local_rng, approximating_neural_network)

    tspan = extrema(solutions_dataframe.t)
    uode_derivative_function = get_uode_model_function_not_scaled(approximating_neural_network, st)
    prob_uode_pred = ODEProblem{true}(uode_derivative_function, Array(solutions_dataframe[1, 2:(end)]), tspan)

    #loss function for the comparison among the parameters and the predictions
    function loss_function(data, pred, max_oscillation)
      original_cost = sum(abs2.(data[observables, :] .- pred[observables, :]) ./ abs2.(max_oscillation[observables]))
      return 1 / size(data, 2) * original_cost
    end

    θ = res
    initial_states_survival = deepcopy(initial_states)
    initial_states_survival[end-1] = initial_states_survival[end-1] / 10.0 #reduce the initial caspase 3 concentration for survival

    sol = Array(solve(
      remake(
        prob_uode_pred;
        p=θ,
        tspan=tspan,
        u0=initial_states_survival
        #u0=u0_original[i, :, 1]
      ),
      integrator;
      saveat=solutions_dataframe.t,
      reltol=reltol,
      abstol=abstol,
      sensealg=sensealg,
      verbose=true
    ))

    if size(sol, 2) != length(solutions_dataframe.t)
      println("Simulation failed for this result.")
      continue
    end

    push!(predictions, sol)

    #plot the results towards the training data 
    if length(variable_plots) == 0
      for var_index in 1:size(training_datas[1], 1)

        p = Plots.scatter(
          solutions_dataframe.t,
          solutions_dataframe[:, var_index+1],
          label="training data",
          lw=2,
          xlabel="time",
          ylabel="concentration",
          title="Variable $(var_index)",
        )
        plot!(p, solutions_dataframe.t, sol[var_index, :], label="UDE prediction", lw=2, ls=:dash, color=:red, legend=nothing)
        #range 0-1 for y axis
        y_min = minimum(solutions_dataframe[:, var_index+1])
        y_max = maximum(solutions_dataframe[:, var_index+1])
        y_range = y_max - y_min
        y_lower = y_min - 0.1 * y_range
        y_upper = y_max + 0.1 * y_range
        Plots.plot!(p, ylims=(y_lower, y_upper))
        push!(variable_plots, p)
      end
    else
      for var_index in 1:size(training_datas[1], 1)
        p = variable_plots[var_index]
        plot!(p, solutions_dataframe.t, sol[var_index, :], label="UDE prediction", lw=2, ls=:dash, color=:red, legend=nothing)
        y_min = minimum(solutions_dataframe[:, var_index+1])
        y_max = maximum(solutions_dataframe[:, var_index+1])
        y_range = y_max - y_min
        y_lower = y_min - 0.1 * y_range
        y_upper = y_max + 0.1 * y_range
        Plots.plot!(p, ylims=(y_lower, y_upper))
      end
    end
  end

  #plot in a grid layout
  p = Plots.plot(variable_plots..., layout=(4, 2), size=(900, 900))
  #circle the forth variable which is the observable
  savefig(p, "cell_apop_UDE_results/v_se_survival_$ensemble_index.png")

  global variable_plots = []
  for res in MOD_ensemble

    global variable_plots

    ########################################## NODE neural network ############################################################
    activation_function_fun = gelu
    my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims..., gain=1.0)
    approximating_neural_network = Lux.Chain(
      Lux.Dense(in_dim, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(neural_network_dimension, neural_network_dimension, activation_function_fun; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(neural_network_dimension, out_dim; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    )

    observables = [4,]

    local_rng = StableRNG(0)
    p_net, st = Lux.setup(local_rng, approximating_neural_network)

    tspan = extrema(solutions_dataframe.t)
    uode_derivative_function = get_uode_model_function_not_scaled(approximating_neural_network, st)
    prob_uode_pred = ODEProblem{true}(uode_derivative_function, Array(solutions_dataframe[1, 2:(end)]), tspan)

    #loss function for the comparison among the parameters and the predictions
    function loss_function(data, pred, max_oscillation)
      original_cost = sum(abs2.(data[observables, :] .- pred[observables, :]) ./ abs2.(max_oscillation[observables]))
      return 1 / size(data, 2) * original_cost
    end

    θ = res

    initial_states_survival = deepcopy(initial_states)
    initial_states_survival[end-1] = initial_states_survival[end-1] / 10.0 #reduce the initial caspase 3 concentration for survival


    sol = Array(solve(
      remake(
        prob_uode_pred;
        p=θ,
        tspan=tspan,
        u0=initial_states_survival
        #u0=u0_original[i, :, 1]
      ),
      integrator;
      saveat=solutions_dataframe.t,
      reltol=reltol,
      abstol=abstol,
      sensealg=sensealg,
      verbose=true
    ))

    if size(sol, 2) != length(solutions_dataframe.t)
      println("Simulation failed for this result.")
      continue
    end

    push!(MOD_predictions, sol)

    #plot the results towards the training data 
    if length(variable_plots) == 0
      for var_index in 1:size(training_datas[1], 1)

        p = Plots.scatter(
          solutions_dataframe.t,
          solutions_dataframe[:, var_index+1],
          label="training data",
          lw=2,
          xlabel="time",
          ylabel="concentration",
          title="Variable $(var_index)",
        )
        plot!(p, solutions_dataframe.t, sol[var_index, :], label="UDE prediction", lw=2, ls=:dash, color=:red, legend=nothing)
        #range 0-1 for y axis
        y_min = minimum(solutions_dataframe[:, var_index+1])
        y_max = maximum(solutions_dataframe[:, var_index+1])
        y_range = y_max - y_min
        y_lower = y_min - 0.1 * y_range
        y_upper = y_max + 0.1 * y_range
        Plots.plot!(p, ylims=(y_lower, y_upper))
        push!(variable_plots, p)
      end
    else
      for var_index in 1:size(training_datas[1], 1)
        p = variable_plots[var_index]
        plot!(p, solutions_dataframe.t, sol[var_index, :], label="UDE prediction", lw=2, ls=:dash, color=:red, legend=nothing)
        y_min = minimum(solutions_dataframe[:, var_index+1])
        y_max = maximum(solutions_dataframe[:, var_index+1])
        y_range = y_max - y_min
        y_lower = y_min - 0.1 * y_range
        y_upper = y_max + 0.1 * y_range
        Plots.plot!(p, ylims=(y_lower, y_upper))
      end
    end
  end

  #plot in a grid layout
  p = Plots.plot(variable_plots..., layout=(4, 2), size=(900, 900))
  #circle the forth variable which is the observable
  savefig(p, "cell_apop_UDE_results/v_se_survival_MOD_$ensemble_index.png")

  #for each time, i compute the mean and std of the predictions
  mean_prediction = zeros(size(predictions[1]))
  std_prediction = zeros(size(predictions[1]))

  mean_prediction_MOD = zeros(size(MOD_predictions[1]))
  std_prediction_MOD = zeros(size(MOD_predictions[1]))

  for t_index in 1:size(predictions[1], 2)
    #collect all predictions at this time
    preds_at_t = [predictions[i][:, t_index] for i in 1:length(predictions)]
    #compute mean and std
    mean_prediction[:, t_index] = mean(hcat(preds_at_t...), dims=2)[:]
    std_prediction[:, t_index] = std(hcat(preds_at_t...), dims=2)[:]

    preds_at_t_MOD = [MOD_predictions[i][:, t_index] for i in 1:length(MOD_predictions)]
    #compute mean and std
    mean_prediction_MOD[:, t_index] = mean(hcat(preds_at_t_MOD...), dims=2)[:]
    std_prediction_MOD[:, t_index] = std(hcat(preds_at_t_MOD...), dims=2)[:]
  end

  #try and visualize the results with 1.96 * std as confidence interval
  variable_plots = []
  variable_contained_in_ci = []
  MOD_variable_contained_in_ci = []

  for var_index in 1:size(training_datas[1], 1)

    p = Plots.scatter(
      solutions_dataframe.t,
      solutions_dataframe[:, var_index+1],
      label="training data",
      lw=2,
      xlabel="time",
      ylabel="concentration",
      title="Variable $(var_index)",
    )
    plot!(p, solutions_dataframe.t, mean_prediction[var_index, :], label="UDE mean prediction", lw=2, ls=:dash, color=:red)

    n_ensemble = 5
    t_value = quantile(TDist(n_ensemble - 1), 0.975)

    #plot the confidence interval
    upper_bound = mean_prediction[var_index, :] .+ t_value * sqrt(1 + 1 / n_ensemble) * std_prediction[var_index, :]
    lower_bound = mean_prediction[var_index, :] .- t_value * sqrt(1 + 1 / n_ensemble) * std_prediction[var_index, :]

    lower_bound_MOD = mean_prediction_MOD[var_index, :] .- t_value * sqrt(1 + 1 / n_ensemble) * std_prediction_MOD[var_index, :]
    upper_bound_MOD = mean_prediction_MOD[var_index, :] .+ t_value * sqrt(1 + 1 / n_ensemble) * std_prediction_MOD[var_index, :]

    plot!(p, solutions_dataframe.t, upper_bound, lw=1, ls=:dot, color=:orange, label="95% CI")
    plot!(p, solutions_dataframe.t, lower_bound, lw=1, ls=:dot, color=:orange, label="")

    plot!(p, solutions_dataframe.t, upper_bound_MOD, lw=1, ls=:dot, color=:green, label="95% CI MOD")
    plot!(p, solutions_dataframe.t, lower_bound_MOD, lw=1, ls=:dot, color=:green, label="")

    #fill between upper and lower bound
    Plots.plot!(p, solutions_dataframe.t, upper_bound, fillrange=lower_bound, fillalpha=0.2, fillcolor=:orange, label="")
    Plots.plot!(p, solutions_dataframe.t, upper_bound_MOD, fillrange=lower_bound_MOD, fillalpha=0.2, fillcolor=:green, label="")
    Plots.plot!(p, legend=nothing)

    #check if the variable is inside the confidence interval
    observations = solutions_dataframe[:, var_index+1]
    epsilon = 1e-4
    inside_ci = ((observations .>= (lower_bound .- epsilon)) .& (observations .<= (upper_bound .+ epsilon)))

    MOD_inside_ci = ((observations .>= (lower_bound_MOD .- epsilon)) .& (observations .<= (upper_bound_MOD .+ epsilon)))

    push!(variable_contained_in_ci, inside_ci)
    push!(MOD_variable_contained_in_ci, MOD_inside_ci)

    #range 0-1 for y axis
    y_min = minimum(solutions_dataframe[:, var_index+1])
    y_max = maximum(solutions_dataframe[:, var_index+1])
    y_range = y_max - y_min
    y_lower = y_min - 1 * y_range
    y_upper = y_max + 1 * y_range
    Plots.plot!(p, ylims=(y_lower, y_upper))
    push!(variable_plots, p)
  end

  #make the % between the same positions of the variable contained in ci
  contained_points = zeros(size(variable_contained_in_ci[1]))
  for t_index in 1:length(contained_points)
    contained = 1
    for var_index in observables
      if !variable_contained_in_ci[var_index][t_index]
        contained = 0
        break
      end
    end
    contained_points[t_index] = contained
  end
  contained_points = sum(contained_points) / length(contained_points)

  contained_points_MOD = zeros(size(MOD_variable_contained_in_ci[1]))
  for t_index in 1:length(contained_points_MOD)
    contained = 1
    for var_index in observables
      if !MOD_variable_contained_in_ci[var_index][t_index]
        contained = 0
        break
      end
    end
    contained_points_MOD[t_index] = contained
  end
  contained_points_MOD = sum(contained_points_MOD) / length(contained_points_MOD)

  push!(survival_trajectory_contained, contained_points)
  push!(MOD_survival_trajectory_contained, contained_points_MOD)

  #plot in a grid layout
  p = Plots.plot(variable_plots..., layout=(4, 2), size=(900, 900))
  #circle the forth variable which is the observable
  savefig(p, "cell_apop_UDE_results/v_se_ci_ensemble_survival_$ensemble_index.png")

end
#trying and simulate the model on the other initial conditions (cell death)
#save the contained in ci 
serialize("cell_apop_UDE_results/trajectory_contained_in_ci_survival.jld", survival_trajectory_contained)
serialize("cell_apop_UDE_results/MOD_trajectory_contained_in_ci_survival.jld", MOD_survival_trajectory_contained)

# put together death and survival trajectory contained in ci and write in a txt file
total_trajectory_contained = vcat(death_trajectory_contained, survival_trajectory_contained)
open("cell_apop_UDE_results/total_trajectory_contained_in_ci.txt", "w") do io
  println(io, "Trajectory contained in CI for death and survival conditions:")
  for value in death_trajectory_contained
    println(io, value)
  end
  println(io, "Trajectory contained in CI for survival conditions:")
  for value in survival_trajectory_contained
    println(io, value)
  end
  println(io, "Trajectory contained in CI for MOD death conditions:")
  for value in MOD_death_trajectory_contained
    println(io, value)
  end
  println(io, "Trajectory contained in CI for MOD survival conditions:")
  for value in MOD_survival_trajectory_contained
    println(io, value)
  end
end





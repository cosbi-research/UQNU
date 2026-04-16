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

#take the ensemble 5 for 5
#= for res in not_failed_results

  global variable_plots

  if res.status != "success"
    push!(losses, 100.0)
    continue
  end

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

  tspan = extrema(solution_dataframes[1].t)
  uode_derivative_function = get_uode_model_function(approximating_neural_network, st, lower_parameter_boundaries, upper_parameter_boundaries)
  prob_uode_pred = ODEProblem{true}(uode_derivative_function, Array(solution_dataframes[1, 2:(end)]), tspan)

  #loss function for the comparison among the parameters and the predictions
  function loss_function(data, pred, max_oscillation)
    original_cost = sum(abs2.(data[observables, :] .- pred[observables, :]) ./ abs2.(max_oscillation[observables]))
    return 1 / size(data, 2) * original_cost
  end

  θ = res.training_res
  sol = Array(solve(
    remake(
      prob_uode_pred;
      p=θ.p,
      tspan=tspan,
      u0=original_u0_death
      #u0=u0_original[i, :, 1]
    ),
    integrator;
    saveat=solutions_dataframe.t,
    reltol=reltol,
    abstol=abstol,
    sensealg=sensealg,
    verbose=true
  ))

  loss = loss_function(training_datas[1], sol, max_oscillations[1])
  push!(losses, loss)

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
end =#

#losses_to_accept = losses .< 0.1

#plot in a grid layout
#Plots.plot(variable_plots..., layout=(4, 2), size=(900, 900))
#circle the forth variable which is the observable
#savefig("cell_apop_UDE_results/variable_plots_training.png")

#not_failed_results = not_failed_results[losses_to_accept]

datafile = "../../data_generator/cell_apoptosis_silico_data.jld"
solutions_dataframe = deserialize(datafile)

training_data_structure = deserialize("../../data_generator/cell_apoptosis_training_data_structure.jld")


datafile = "../../data_generator/cell_apoptosis_silico_data_survival.jld"
solutions_dataframe_survival = deserialize(datafile)


death_trajectory_contained = []
MOD_death_trajectory_contained = []
sigma_naive_ensembles = []
sigma_mod_ensembles = []

function costFunctionOnSingleTraj(simulation)
  original_solutions = training_data_structure.solution_dataframes[1]

  cost_trajectory = 1 / size(original_solutions, 1) * (sum([sum((simulation[j, :] .- original_solutions[!, j+1]) .^ 2) ./ training_data_structure.max_oscillations[j]^2 for j in observables]))

  return cost_trajectory
end


cost_naive_ensembles = []
for ensemble_index in 1:10

  #load the training results 
  results = deserialize("../MOD_results/result_ca$ensemble_index/results.jld")
  MOD_ensemble = results.ensemble_original
  initial_states = results.initial_states
  initial_states = vec(initial_states[1, :, 1]) #take the death trajectory initial states

  naive_ensemble = results.naive_ensemble_reference
  naive_ensemble_reference_initial_states = results.naive_ensemble_reference_initial_states


  predictions = []

  for index in axes(naive_ensemble, 1)

    res = naive_ensemble[index]
    initial_states_tmp = naive_ensemble_reference_initial_states[index]
    initial_states_tmp = vec(initial_states_tmp[:, :, 1]) #take the death trajectory initial states

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
        u0=initial_states_tmp[:, 1]
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

    sol_as_array = Array(sol)
    tmp_cost = costFunctionOnSingleTraj(sol_as_array)

    push!(cost_naive_ensembles, tmp_cost)

    push!(predictions, sol)
  end
end

#find the extrema of the cost 
cost_extrema = extrema(cost_naive_ensembles)


plt_death = Plots.plot()
plt_survival = Plots.plot()

candidates = []
for ensemble_index in 1:5

  #load the training results 
  results = deserialize("result_ca$ensemble_index/results.jld")
  trajectories = deserialize("result_ca$ensemble_index/trajectories.jld")
  MOD_ensemble = results.ensemble_original
  initial_states = results.initial_states
  initial_states = vec(initial_states[1, :, 1]) #take the death trajectory initial states

  #initial parameters for the MOD ensemble
  physical_parameters = MOD_ensemble[1].ode_par

  cost_along_trajectories = []

  traj_number = 0
  for trajectory in trajectories

    traj_number += 1
    iter = 0
    for res in trajectory[1:(end-1)]

      iter += 1

      tmp_parameters = deepcopy(res)
      tmp_parameters.ode_par = tmp_parameters.ode_par .* physical_parameters

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

      θ = tmp_parameters
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

      sol_as_array = Array(sol)
      tmp_cost = costFunctionOnSingleTraj(sol_as_array)

      #println("Cost for this trajectory: $tmp_cost")

      push!(cost_along_trajectories, tmp_cost)

      if (tmp_cost < (cost_extrema[2] +cost_extrema[1])/2 && iter > 100) || (iter == 1 && traj_number == 1)

      #if  (iter == 1 && traj_number == 1)

        #attempt to optimize the parameters starting from this point 

        if length(candidates) == 0
          plot!(plt_death, solutions_dataframe.t, sol[4, :], label="UDE model", lw=1, color=:blue)
        else
          plot!(plt_death, solutions_dataframe.t, sol[4, :], label="", lw=1, color=:blue)
        end
        #range 0-1 for y axis
        y_min = minimum(solutions_dataframe[:, 4+1])
        y_max = maximum(solutions_dataframe[:, 4+1])
        y_range = y_max - y_min
        y_lower = 0.0
        y_upper = y_max + 0.1 * y_range
        Plots.plot!(plt_death, ylims=(y_lower, y_upper))
        
        #simulation of survival 
        initial_states_survival = deepcopy(initial_states)
        initial_states_survival[end-1] = initial_states_survival[end-1]/ 10.0
        θ = tmp_parameters
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

        if length(candidates) == 0
          plot!(plt_survival, solutions_dataframe.t, sol[4, :], label="UDE model", lw=1, color=:blue)
        else
          plot!(plt_survival, solutions_dataframe.t, sol[4, :], label="", lw=1, color=:blue)
        end

        #range 0-1 for y axis
        y_min = minimum(solutions_dataframe_survival[:, 4+1])
        y_max = maximum(solutions_dataframe_survival[:, 4+1])
        y_range = y_max - y_min
        y_lower = 0.0
        y_upper = y_max + 4.0 * y_range
        Plots.plot!(plt_survival, ylims=(y_lower, y_upper))

        println("Seleziono " * string(iter))

        push!(candidates, (parameters=deepcopy(tmp_parameters), cost=tmp_cost))
        #println("Cost $tmp_cost is out of the extrema range, skipping.")
        continue
      end
    end
  end
end

Plots.scatter!(plt_death,
  solutions_dataframe.t,
  solutions_dataframe[:, 4+1],
  label="training data",
  markersize=6,
  color=:red,
  xguidefont=font(18),    # Increase x-axis label font size
  yguidefont=font(18),
  titlefont=font(16),
  xtickfont=font(12),     # Increase x-axis tick font size
  ytickfont=font(12),     # Increase y-axis label font size
  legendfont=font(10),
)

Plots.plot!(plt_survival,
  xlabel="Time (hours)",
  ylabel="y4 (10⁵ molecules/cell)",
  title="Cell death",
  xguidefont=font(18),    # Increase x-axis label font size
  yguidefont=font(18),
  titlefont=font(16),
  xtickfont=font(12),     # Increase x-axis tick font size
  ytickfont=font(12),     # Increase y-axis label font size
  legendfont=font(10),
)

Plots.plot!(plt_death,
  xlabel="Time (hours)",
  ylabel="y4 (10⁵ molecules/cell)",
  title="Cell survival",
  xguidefont=font(18),    # Increase x-axis label font size
  yguidefont=font(18),
  titlefont=font(16),
  xtickfont=font(12),     # Increase x-axis tick font size
  ytickfont=font(12),     # Increase y-axis label font size
  legendfont=font(10),
)

datafile = "../../data_generator/cell_apoptosis_silico_data_survival.jld"
solutions_dataframe = deserialize(datafile)

Plots.plot!(plt_survival,
  solutions_dataframe.t,
  solutions_dataframe[:, 4+1],
  label="ground truth",
  lw=3,
  color=:red
)

#Save the plots 
Plots.savefig(plt_death, "death_trajectory_candidates.png")
Plots.savefig(plt_survival, "survival_trajectory_candidates.png")